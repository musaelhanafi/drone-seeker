import argparse
import collections
import cv2
import time


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


def open_capture(source, w, h):
    """Open camera source.
    - GStreamer pipeline string (contains ' ! '): uses cv2.CAP_GSTREAMER
    - Integer: tries Picamera2 first, falls back to OpenCV
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
            from picamera2 import Picamera2
            cam = Picamera2()
            cfg = cam.create_video_configuration(
                main={"size": (w, h), "format": "BGR888"}
            )
            cam.configure(cfg)
            cam.start()

            class _Cap:
                def read(self):
                    try:
                        return True, cam.capture_array()
                    except Exception:
                        return False, None
                def release(self):
                    cam.stop()

            print(f"[test_camera] Using Picamera2 backend  {w}x{h}")
            return _Cap()
        except ImportError:
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
args = parser.parse_args()

if args.udpsrc is not None:
    source = _build_udpsrc_pipeline(args.udpsrc, args.udpsrc_codec)
else:
    source = int(args.source) if args.source.isdigit() else args.source
cap = open_capture(source, args.width, args.height)

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
