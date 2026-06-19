import argparse
import collections
import cv2
import time


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
        print(f"[test_camera] Using GStreamer backend  pipeline={source!r}")
        return cap

    if isinstance(source, int):
        try:
            cap = Picamera2Capture(w, h, flip=flip)
            print(f"[test_camera] Using Picamera2 backend  {w}x{h}  flip={flip}")
            return cap
        except Exception:
            pass

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    print(f"[test_camera] Using OpenCV backend  source={source!r}  "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


parser = argparse.ArgumentParser(description="Camera / GStreamer UDP stream viewer")
parser.add_argument("source", nargs="?", default="0",
                    help="Camera index (e.g. 0), video file, or GStreamer pipeline string")
parser.add_argument("--width",  type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--udpsrc", type=int, default=None, metavar="PORT",
                    help="Receive H.264/MJPEG stream from UDP port (e.g. --udpsrc 5600). "
                         "Overrides positional source.")
parser.add_argument("--udpsrc-codec", default="h264", choices=["h264", "mjpeg"],
                    metavar="CODEC", help="Codec for --udpsrc: h264 (default) or mjpeg")
parser.add_argument("--flip", action="store_true", default=False,
                    help="Flip frames 180 degrees. Uses libcamera Transform for Picamera2 sources.")
args = parser.parse_args()

if args.udpsrc is not None:
    source = _build_udpsrc_pipeline(args.udpsrc, args.udpsrc_codec)
else:
    source = int(args.source) if args.source.isdigit() else args.source
cap = open_capture(source, args.width, args.height, flip=args.flip)
# Picamera2 flips in hardware (libcamera Transform); for OpenCV/GStreamer
# sources do the 180-degree flip in software here.
sw_flip = args.flip and not isinstance(cap, Picamera2Capture)

newh = args.height
neww = args.width
print(f"display size: {neww}x{newh}")

prev_time = time.time()
frame_times = collections.deque(maxlen=30)

cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("Camera", neww, newh)
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame — exiting.")
        break

    if sw_flip:
        frame = cv2.flip(frame, -1)

    frame = cv2.resize(frame, (neww, newh))
    curr_time = time.time()
    frame_times.append(curr_time - prev_time)
    prev_time = curr_time
    fps = 1.0 / (sum(frame_times) / len(frame_times))

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
