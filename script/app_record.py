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
"""

import argparse
import collections
import datetime
import os
import shutil
import subprocess
import sys
import time

import cv2
from pymavlink import mavutil


# ch6 "middle" band (matches seekerctrl.py's active-switch window).
CH6_MID_LOW  = 1400
CH6_MID_HIGH = 1700


class Picamera2Capture:
    """cv2.VideoCapture drop-in using picamera2/libcamera (RPi CSI camera)."""
    def __init__(self, width: int = 1280, height: int = 720, flip: bool = False):
        from picamera2 import Picamera2
        from libcamera import Transform
        self._w = width
        self._h = height
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={"size": (self._w, self._h), "format": "BGR888"},
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
parser.add_argument("--no-display", action="store_true", default=False,
                    help="Headless: do not open a preview window.")
args = parser.parse_args()

if args.udpsrc is not None:
    source = _build_udpsrc_pipeline(args.udpsrc, args.udpsrc_codec)
else:
    source = int(args.source) if args.source.isdigit() else args.source

cap = open_capture(source, args.width, args.height, flip=args.flip)
# Picamera2 flips in hardware (libcamera Transform); for OpenCV/GStreamer
# sources do the 180-degree flip in software here.
sw_flip = args.flip and not isinstance(cap, Picamera2Capture)
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

        curr_time = time.time()
        frame_times.append(curr_time - prev_time)
        prev_time = curr_time
        fps = 1.0 / (sum(frame_times) / len(frame_times))

        ch6_pwm = poll_ch6(master, ch6_pwm)
        in_middle = CH6_MID_LOW <= ch6_pwm < CH6_MID_HIGH

        # Edge-trigger the recorder on the ch6 middle band.
        if in_middle and not recorder.active:
            rec_fps = args.fps if args.fps else max(1.0, fps)
            recorder.start(frame.shape, rec_fps)
        elif not in_middle and recorder.active:
            recorder.stop()

        # Record the RAW frame (no overlays) before drawing the preview HUD.
        recorder.write(frame)

        if show:
            disp = frame.copy()
            status = "REC" if recorder.active else "idle"
            color  = (0, 0, 255) if recorder.active else (0, 255, 0)
            cv2.putText(disp, f"FPS:{fps:.1f}  ch6:{ch6_pwm}  {status}",
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
