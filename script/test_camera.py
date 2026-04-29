import argparse
import collections
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument("source", nargs="?", default="0",
                    help="Camera index (e.g. 0) or video file path")
args = parser.parse_args()

source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)
w = 1280    
h = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
newh = 480
neww = newh*w//h

print(f"new width={neww} new height={newh}")

prev_time = time.time()
frame_times = collections.deque(maxlen=30)

while True:
    ret, frame = cap.read()
    if not ret:
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
