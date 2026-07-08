"""Annotated-video recorder — raw-BGR pipe to an encoder subprocess.

Same encoder strategy as SeekerCtrl's in-flight recorder: on the Jetson the
H.264 encode runs on the Tegra NVENC block (gst-launch fdsrc → nvvidconv →
nvv4l2h264enc → mp4), so the vision loop never stalls on x264 pipe
backpressure; boxes without the HW encoder fall back to ffmpeg libx264
ultrafast. Exists as a standalone class so the test harnesses (which drive
Seeker.run() directly, without MAVLink) can record too.
"""

from __future__ import annotations
import datetime
import subprocess
import time


class VideoRecorder:
    def __init__(self):
        self._proc    = None
        self._hw_enc  = None   # tri-state: None = not probed yet
        self.path     = None
        self._fps     = 0.0
        self._t0      = 0.0
        self._written = 0

    def _has_hw_encoder(self) -> bool:
        """True if the Tegra HW H.264 encoder (nvv4l2h264enc) is available.
        Probed once via gst-inspect-1.0 and cached — keeps the per-record path
        cheap and lets the dev box fall back to ffmpeg automatically."""
        if self._hw_enc is None:
            try:
                r = subprocess.run(
                    ["gst-inspect-1.0", "nvv4l2h264enc"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._hw_enc = (r.returncode == 0)
            except Exception:
                self._hw_enc = False
        return self._hw_enc

    def _encoder_cmd(self, w: int, h: int, fps: float, path: str):
        """Build the encoder subprocess argv reading raw bgr24 from stdin."""
        if self._has_hw_encoder():
            from fractions import Fraction
            fr = Fraction(max(fps, 1.0)).limit_denominator(1000)
            return ([
                "gst-launch-1.0", "-q", "-e",
                "fdsrc", "fd=0", "!",
                "rawvideoparse", "format=bgr",
                f"width={w}", f"height={h}",
                f"framerate={fr.numerator}/{fr.denominator}", "!",
                "videoconvert", "!", "video/x-raw,format=BGRx", "!",
                "nvvidconv", "!", "video/x-raw(memory:NVMM),format=NV12", "!",
                "nvv4l2h264enc", "bitrate=8000000", "maxperf-enable=1", "!",
                "h264parse", "!", "qtmux", "!", "filesink", f"location={path}",
            ], "nvv4l2h264enc (HW)")
        return ([
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            path,
        ], "libx264 ultrafast (CPU)")

    def open(self, frame_shape, fps: float, label: str = "tracking"):
        h, w = frame_shape[:2]
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = f"{label}_{ts}.mp4"
        cmd, enc  = self._encoder_cmd(w, h, fps, self.path)
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        self._fps     = max(fps, 1.0)
        self._t0      = time.monotonic()
        self._written = 0
        print(f"[REC] {label} recording → {self.path}  fps={fps:.1f}  enc={enc}")

    def write(self, frame):
        """Write `frame`, locked to the wall clock: the file is constant-frame-
        rate at the fps given to open(), so a frame is duplicated when the
        caller's loop falls behind that rate and dropped when it runs ahead.
        Playback is therefore real-time regardless of loop speed."""
        if self._proc is None:
            return
        due = int((time.monotonic() - self._t0) * self._fps) + 1
        try:
            while self._written < due:
                self._proc.stdin.write(frame.tobytes())
                self._written += 1
        except BrokenPipeError:
            pass

    def close(self, label: str = "tracking"):
        proc = self._proc
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()
        self._proc = None
        elapsed = time.monotonic() - self._t0
        print(f"[REC] {label} recording stopped — {self._written} frames "
              f"in {elapsed:.1f}s (file plays {self._written / self._fps:.1f}s "
              f"@ {self._fps:.1f} fps)")
