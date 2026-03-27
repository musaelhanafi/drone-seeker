import cv2
import numpy as np


# ── Pink HSV ranges (OpenCV hue 0-179) ───────────────────────────────────────
# Two bands cover hot-pink/magenta and light-pink/rose.
_PINK_RANGES = [
    (np.array([145, 50,  80]),  np.array([179, 255, 255])),  # magenta / hot-pink
    (np.array([0,   50,  80]),  np.array([10,  150, 255])),  # rose / light-pink
]

# Minimum contour area to accept as a valid pink blob (pixels²)
_MIN_BLOB_AREA = 400


def _pink_mask(hsv: np.ndarray) -> np.ndarray:
    """Return a binary mask of all pink pixels in *hsv* frame."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _PINK_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    # small morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask


def _largest_blob_rect(mask: np.ndarray):
    """Return the bounding rect (x, y, w, h) of the largest pink blob,
    or None if no blob meets the minimum area threshold."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < _MIN_BLOB_AREA:
        return None
    return cv2.boundingRect(largest)


class Seeker:
    def __init__(
        self,
        source: int | str = 0,
        window_name: str = "Seeker",
    ):
        """
        source      : camera index (int) or video / image file path (str)
        window_name : OpenCV display window title
        """
        self.source      = source
        self.window_name = window_name

        self.cap         = None
        self._roi_hist   = None   # hue histogram of the tracked ROI
        self._track_win  = None   # current CamShift window (x, y, w, h)
        self._term_crit  = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self):
        """Open the video source and create the display window."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source!r}")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        print(f"[Seeker] Opened source={self.source!r}")

    def close(self):
        """Release the capture device and destroy the display window."""
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyWindow(self.window_name)

    # ── Tracking helpers ──────────────────────────────────────────────────────

    def _init_camshift(self, frame: np.ndarray, rect):
        """Build the hue histogram for the detected pink ROI and seed CamShift."""
        x, y, w, h = rect
        hsv_roi  = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        mask_roi = _pink_mask(hsv_roi)

        self._roi_hist = cv2.calcHist(
            [hsv_roi], [0], mask_roi, [180], [0, 180]
        )
        cv2.normalize(self._roi_hist, self._roi_hist, 0, 255, cv2.NORM_MINMAX)
        self._track_win = rect
        print(f"[Seeker] CamShift initialised on pink blob at {rect}")

    def track(self, frame: np.ndarray):
        """Run one tracking step.

        Returns (annotated_frame, cx, cy) where cx/cy is the tracked centre,
        or (annotated_frame, None, None) when no target is locked.
        """
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = _pink_mask(hsv)
        out  = frame.copy()

        # ── Acquire / re-acquire target ───────────────────────────────────────
        if self._roi_hist is None or self._track_win is None:
            rect = _largest_blob_rect(mask)
            if rect is None:
                cv2.putText(out, "Searching for pink ...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (203, 192, 255), 2)
                return out, None, None
            self._init_camshift(frame, rect)

        # ── CamShift step ─────────────────────────────────────────────────────
        back_proj = cv2.calcBackProject(
            [hsv], [0], self._roi_hist, [0, 180], 1
        )
        back_proj &= mask  # restrict to pink pixels only

        ret, self._track_win = cv2.CamShift(
            back_proj, self._track_win, self._term_crit
        )

        # ── Validate: drop lock if window collapsed ───────────────────────────
        x, y, w, h = self._track_win
        if w < 5 or h < 5:
            self._roi_hist  = None
            self._track_win = None
            return out, None, None

        # ── Draw rotated bounding box ─────────────────────────────────────────
        pts = cv2.boxPoints(ret).astype(np.intp)
        cv2.polylines(out, [pts], True, (203, 192, 255), 2)

        cx = int(ret[0][0])
        cy = int(ret[0][1])
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)

        return out, cx, cy

    # ── Error computation ─────────────────────────────────────────────────────

    def error_xy(self, cx, cy, frame_shape):
        """Convert tracked centre (cx, cy) to normalised error [-1, 1].

        errorx: positive = target right of frame centre
        errory: positive = target above  frame centre
        Returns (None, None) when cx/cy are None.
        """
        if cx is None:
            return None, None
        h, w = frame_shape[:2]
        errorx =  (cx - w / 2.0) / (w / 2.0)
        errory = -(cy - h / 2.0) / (h / 2.0)
        return float(errorx), float(errory)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def read_frame(self):
        """Return (ok, frame) from the capture device."""
        return self.cap.read()

    def run(self):
        """Open source, run CamShift tracking loop; press 'q' to quit,
        'r' to reset the tracker."""
        self.open()
        try:
            while True:
                ok, frame = self.read_frame()
                if not ok:
                    print("[Seeker] End of stream.")
                    break

                annotated, cx, cy = self.track(frame)
                errorx, errory    = self.error_xy(cx, cy, frame.shape)

                if errorx is not None:
                    label = f"ex={errorx:+.3f}  ey={errory:+.3f}"
                    cv2.putText(annotated, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow(self.window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[Seeker] Quit.")
                    break
                elif key == ord("r"):
                    self._roi_hist  = None
                    self._track_win = None
                    print("[Seeker] Tracker reset.")
        finally:
            self.close()
