import argparse
import collections
import cv2
import time


def open_capture(source, w, h):
    """Open camera source. Uses Picamera2 on Pi 5 (integer source), OpenCV otherwise."""
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


parser = argparse.ArgumentParser()
parser.add_argument("source", nargs="?", default="0",
                    help="Camera index (e.g. 0) or video file path")
parser.add_argument("--width",  type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args = parser.parse_args()

source = int(args.source) if args.source.isdigit() else args.source
cap = open_capture(source, args.width, args.height)

newh = 480
neww = newh * args.width // args.height
print(f"display size: {neww}x{newh}")

prev_time = time.time()
frame_times = collections.deque(maxlen=30)

cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
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
