import argparse
import collections
import cv2
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("source", nargs="?", default="0",
                    help="Camera index (e.g. 0) or video file path")
args = parser.parse_args()

source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)
w, h = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
newh = h
neww = w

# Pink HSV range
PINK_LOWER = np.array([140, 50, 100])
PINK_UPPER = np.array([170, 255, 255])

# CAMShift state
track_window = None
roi_hist = None
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

prev_time = time.time()
frame_times = collections.deque(maxlen=30)


def init_camshift(hsv, contour):
    x, y, cw, ch = cv2.boundingRect(contour)
    roi_hsv = hsv[y:y + ch, x:x + cw]
    mask_roi = cv2.inRange(roi_hsv, PINK_LOWER, PINK_UPPER)
    hist = cv2.calcHist([roi_hsv], [0], mask_roi, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return (x, y, cw, ch), hist


while True:
    curr_time = time.time()
    frame_times.append(curr_time - prev_time)
    prev_time = curr_time
    fps = 1.0 / (sum(frame_times) / len(frame_times))

    ret, frame = cap.read()
    if not ret:
        break

    if neww != w or newh != h:
        frame = cv2.resize(frame, (neww, newh))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PINK_LOWER, PINK_UPPER)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= 500]

    if valid:
        # Re-init CAMShift on the largest detected region each frame
        largest = max(valid, key=cv2.contourArea)
        track_window, roi_hist = init_camshift(hsv, largest)

    cx, cy = None, None

    if roi_hist is not None and track_window is not None:
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        back_proj &= mask

        ret_cs, track_window = cv2.CamShift(back_proj, track_window, term_crit)
        pts = cv2.boxPoints(ret_cs).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 165, 255), 2)

        cx, cy = int(ret_cs[0][0]), int(ret_cs[0][1])
        cv2.line(frame, (0, cy), (neww, cy), (0, 255, 255), 1)
        cv2.line(frame, (cx, 0), (cx, newh), (0, 255, 255), 1)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    # Center corner-bracket crosshair
    cx_img, cy_img = neww // 2, newh // 2
    box, arm = 80, 16
    for x, y, dx, dy in [
        (cx_img - box, cy_img - box, +1, +1),
        (cx_img + box, cy_img - box, -1, +1),
        (cx_img + box, cy_img + box, -1, -1),
        (cx_img - box, cy_img + box, +1, -1),
    ]:
        cv2.line(frame, (x, y), (x + arm * dx, y), (255, 255, 255), 2)
        cv2.line(frame, (x, y), (x, y + arm * dy), (255, 255, 255), 2)
    # small cross at center
    cs = 10
    cv2.line(frame, (cx_img - cs, cy_img), (cx_img + cs, cy_img), (255, 255, 255), 1)
    cv2.line(frame, (cx_img, cy_img - cs), (cx_img, cy_img + cs), (255, 255, 255), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detect Pink", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
