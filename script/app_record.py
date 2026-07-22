"""app_record.py — record raw camera/stream input gated by RC channel 6.

Based on test_camera.py (same capture backends: Picamera2, GStreamer udpsrc,
OpenCV) plus a MAVLink listener.  While RC channel 6 sits in the MIDDLE
position the raw frames coming off the source are piped to ffmpeg and written
to rec_<timestamp>.mp4 (libx264, like main.py).  Moving ch6 out of the middle
stops the recording; moving it back in starts a fresh file.

  ch6 low  (< 1400)       : idle
  ch6 mid  (1400..1700)   : RECORDING -> rec_<ts>.mp4
  ch6 high (>= 1700)      : idle

Examples:
  python3 app_record.py --udpsrc 5600
  python3 app_record.py 0 --connection /dev/ttyAMA0 --baud 921600
  python3 app_record.py --udpsrc 5600 --record   # always record, ignore ch6 (no MAVLink)
"""

import argparse
import collections
import datetime
import os
import shutil
import statistics
import subprocess
import sys
import time

import cv2
from pymavlink import mavutil


# ch6 "middle" band (matches seekerctrl.py's active-switch window).
CH6_MID_LOW  = 1400
CH6_MID_HIGH = 1700

# Min frame-interval samples before trusting the measured fps for the recording
# tag. The first interval includes camera/pipeline startup (~0.8 s); without a
# warm-up the recorder gets tagged at a garbage rate (e.g. 1.3 fps → plays 23× slow).
_FPS_WARMUP = 8


class Picamera2Capture:
    """cv2.VideoCapture drop-in using picamera2/libcamera (RPi CSI camera)."""
    def __init__(self, width: int = 1280, height: int = 720, flip: bool = False):
        from picamera2 import Picamera2
        from libcamera import Transform
        self._w = width
        self._h = height
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            # Picamera2 format names are byte-order reversed: "RGB888" yields a
            # B,G,R array (OpenCV BGR), "BGR888" yields R,G,B. capture_array() is
            # fed straight to OpenCV, so use "RGB888" to get true BGR — else R/B
            # swap and the image looks red.
            main={"size": (self._w, self._h), "format": "RGB888"},
            transform=Transform(hflip=1, vflip=1) if flip else Transform(),
        )
        self._cam.configure(cfg)
        self._cam.start()
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop_id: int, value: float) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        return 0.0

    def read(self):
        if not self._opened:
            return False, None
        try:
            return True, self._cam.capture_array()
        except Exception:
            return False, None

    def release(self):
        self._cam.stop()
        self._opened = False


def _build_udpsrc_pipeline(port: int, codec: str) -> str:
    if codec == "mjpeg":
        return (
            f"udpsrc port={port} "
            "! application/x-rtp,encoding-name=JPEG "
            "! rtpjpegdepay ! jpegdec ! videoconvert "
            "! appsink drop=1 max-buffers=1"
        )
    return (
        f"udpsrc port={port} "
        "! application/x-rtp,payload=96 "
        "! rtph264depay ! avdec_h264 ! videoconvert "
        "! appsink drop=1 max-buffers=1"
    )


def _resolve_crop(crop, fw: int, fh: int):
    """Resolve (offset_x, offset_y, w, h) against a frame, clamped to bounds.

    X/Y are the crop's top-left offset; W/H the size (None = 'rest of dimension
    after the offset'). The offset is clamped into the frame and the size trimmed
    so the window never runs off the edge."""
    cx, cy, cw, ch = crop
    cx = min(max(0, cx), max(0, fw - 1))
    cy = min(max(0, cy), max(0, fh - 1))
    cw = fw - cx if cw is None else min(cw, fw - cx)
    ch = fh - cy if ch is None else min(ch, fh - cy)
    return cx, cy, cw, ch


def _apply_crop(frame, crop):
    if crop is None:
        return frame
    fh, fw = frame.shape[:2]
    cx, cy, cw, ch = _resolve_crop(crop, fw, fh)
    return frame[cy:cy + ch, cx:cx + cw]


def _apply_outres(frame, outres):
    """Scale *frame* to (w, h) from --outres. Applied BEFORE crop, so --crop
    coords refer to the SCALED frame. INTER_AREA shrinking, INTER_LINEAR growing."""
    if outres is None:
        return frame
    ow, oh = outres
    fh, fw = frame.shape[:2]
    if (fw, fh) == (ow, oh):
        return frame
    interp = cv2.INTER_AREA if (ow * oh) < (fw * fh) else cv2.INTER_LINEAR
    return cv2.resize(frame, (ow, oh), interpolation=interp)


def open_capture(source, w, h, flip=False):
    """Open camera source.
    - GStreamer pipeline string (contains ' ! '): uses cv2.CAP_GSTREAMER
    - Integer: tries Picamera2 first (flip via libcamera Transform), falls back to OpenCV
    - String path/device: OpenCV
    """
    if isinstance(source, str) and " ! " in source:
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(
                f"GStreamer pipeline failed to open. "
                "Check OpenCV was built with GStreamer support:\n"
                "  python3 -c \"import cv2; print(cv2.getBuildInformation())\" | grep GStreamer"
            )
        print(f"[app_record] Using GStreamer backend  pipeline={source!r}")
        return cap

    if isinstance(source, int):
        try:
            cap = Picamera2Capture(w, h, flip=flip)
            print(f"[app_record] Using Picamera2 backend  {w}x{h}  flip={flip}")
            return cap
        except Exception:
            pass

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    print(f"[app_record] Using OpenCV backend  source={source!r}  "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


# ── ffmpeg recorder (raw bgr24 frames -> libx264 mp4, like main.py) ───────────

def _resolve_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError("ffmpeg not found. Install it and ensure it is on PATH.")


class Recorder:
    """Pipes raw BGR frames to ffmpeg, encoding to rec_<timestamp>.mp4."""
    def __init__(self):
        self._proc = None
        self.path  = None

    @property
    def active(self) -> bool:
        return self._proc is not None

    def start(self, frame_shape, fps: float):
        if self._proc is not None:
            return
        h, w = frame_shape[:2]
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = f"rec_{ts}.mp4"
        cmd = [
            _resolve_ffmpeg(), "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}",
            "-i", "pipe:0",
            "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            self.path,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[REC] recording -> {self.path}  {w}x{h}  fps={fps:.1f}")

    def write(self, frame):
        if self._proc is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                pass

    def stop(self):
        proc = self._proc
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()
        self._proc = None
        print(f"[REC] recording stopped -> {self.path}")


# ── MAVLink ───────────────────────────────────────────────────────────────────

def connect_mavlink(connection_string: str, baud: int):
    print(f"[MAV] connecting to {connection_string} ...")
    master = mavutil.mavlink_connection(connection_string, baud=baud)
    master.wait_heartbeat()
    print(f"[MAV] heartbeat (system {master.target_system}, "
          f"component {master.target_component})")
    # Ask the FC to stream RC_CHANNELS so we can watch ch6.
    try:
        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 10, 1,
        )
    except Exception:
        pass
    return master


def poll_ch6(master, last_pwm: int) -> int:
    """Drain pending RC_CHANNELS messages; return latest ch6 PWM (or last)."""
    pwm = last_pwm
    while True:
        msg = master.recv_match(type="RC_CHANNELS", blocking=False)
        if msg is None:
            break
        pwm = getattr(msg, "chan6_raw", pwm)
    return pwm


# ── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Record raw camera/stream input while RC ch6 is in the middle.")
parser.add_argument("source", nargs="?", default="0",
                    help="Camera index (e.g. 0), video file, or GStreamer pipeline string")
parser.add_argument("--connection", default="udpin:0.0.0.0:14560",
                    help="MAVLink connection string (default: udpin:0.0.0.0:14560)")
parser.add_argument("--baud", type=int, default=57600,
                    help="Baud rate for serial connections (default: 57600)")
parser.add_argument("--width",  type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=float, default=None,
                    help="Recording fps. Default: measured from the live source.")
parser.add_argument("--udpsrc", type=int, default=None, metavar="PORT",
                    help="Receive H.264/MJPEG stream from UDP port (e.g. --udpsrc 5600). "
                         "Overrides positional source.")
parser.add_argument("--udpsrc-codec", default="h264", choices=["h264", "mjpeg"],
                    metavar="CODEC", help="Codec for --udpsrc: h264 (default) or mjpeg")
parser.add_argument("--flip", action="store_true", default=False,
                    help="Flip frames 180 degrees. Uses libcamera Transform for Picamera2 sources.")
parser.add_argument("--outres", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="Scale each frame to this size (e.g. --outres 640 360). "
                         "Applied BEFORE --crop; downscaling cuts CPU/heat.")
parser.add_argument("--crop", type=str, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"),
                    help="Crop each frame to this ROI before recording "
                         "(e.g. --crop 320 180 640 360). Use - for W or H to mean "
                         "'rest of dimension after offset'.")
parser.add_argument("--no-display", action="store_true", default=False,
                    help="Headless: do not open a preview window.")
parser.add_argument("--record", action="store_true", default=False,
                    help="Always record, ignoring the RC ch6 gating (recording starts "
                         "immediately and never stops). Skips the MAVLink connection.")
args = parser.parse_args()

crop = (tuple(None if v == '-' else int(v) for v in args.crop)
        if args.crop else None)
outres = tuple(args.outres) if args.outres else None

if args.udpsrc is not None:
    source = _build_udpsrc_pipeline(args.udpsrc, args.udpsrc_codec)
else:
    source = int(args.source) if args.source.isdigit() else args.source

cap = open_capture(source, args.width, args.height, flip=args.flip)
# Picamera2 flips in hardware (libcamera Transform); for OpenCV/GStreamer
# sources do the 180-degree flip in software here.
sw_flip = args.flip and not isinstance(cap, Picamera2Capture)
if crop is not None:
    print(f"[app_record] crop ROI (applied before recording): "
          f"X={args.crop[0]} Y={args.crop[1]} W={args.crop[2]} H={args.crop[3]}")
if args.record:
    print("[app_record] --record: always recording, ignoring RC ch6 (no MAVLink).")
    master = None
else:
    master = connect_mavlink(args.connection, args.baud)
recorder = Recorder()

show = not args.no_display
if show:
    cv2.namedWindow("app_record", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("app_record", args.width, args.height)

prev_time   = time.time()
frame_times = collections.deque(maxlen=30)
ch6_pwm     = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame — exiting.")
            break

        if sw_flip:
            frame = cv2.flip(frame, -1)
        frame = _apply_outres(frame, outres)   # scale BEFORE crop
        frame = _apply_crop(frame, crop)

        curr_time = time.time()
        frame_times.append(curr_time - prev_time)
        prev_time = curr_time
        # Median interval → robust to the startup/stall outliers that skew a mean.
        _med = statistics.median(frame_times)
        fps = 1.0 / _med if _med > 0 else 0.0

        if args.record:
            in_middle = True          # --record: always on, ch6 ignored
        else:
            ch6_pwm = poll_ch6(master, ch6_pwm)
            in_middle = CH6_MID_LOW <= ch6_pwm < CH6_MID_HIGH

        # Edge-trigger the recorder on the ch6 middle band (or force-on with --record).
        if in_middle and not recorder.active:
            # Robust rec_fps: explicit --fps wins; else the median-based fps once we
            # have enough samples, clamped to a plausible range (default 30). Until
            # warmed up, defer the start (drops a few frames) rather than tag garbage.
            if args.fps:
                rec_fps = float(args.fps)
            elif len(frame_times) >= _FPS_WARMUP:
                rec_fps = fps if 1.0 <= fps <= 120.0 else 30.0
            else:
                rec_fps = None
            if rec_fps is not None:
                recorder.start(frame.shape, rec_fps)
        elif not in_middle and recorder.active:
            recorder.stop()

        # Record the RAW frame (no overlays) before drawing the preview HUD.
        recorder.write(frame)

        if show:
            disp = frame.copy()
            status = "REC" if recorder.active else "idle"
            color  = (0, 0, 255) if recorder.active else (0, 255, 0)
            gate   = "ch6:off" if args.record else f"ch6:{ch6_pwm}"
            cv2.putText(disp, f"FPS:{fps:.1f}  {gate}  {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if recorder.active:
                cv2.circle(disp, (disp.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.imshow("app_record", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    recorder.stop()
    cap.release()
    if show:
        cv2.destroyAllWindows()
