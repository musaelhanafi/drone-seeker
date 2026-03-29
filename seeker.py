import collections
import cv2
import numpy as np
import time


# ── Pink HSV ranges (OpenCV hue 0-179) ───────────────────────────────────────
# Two bands cover hot-pink/magenta and light-pink/rose.
_PINK_RANGES = [
    (np.array([145, 50,  40]),  np.array([179, 255, 255])),  # magenta / hot-pink (incl. dark)
    (np.array([0,   50,  40]),  np.array([10,  255, 255])),  # rose / light-pink  (incl. dark)
]

# Minimum contour area to accept as a valid blob (pixels²)
_MIN_BLOB_AREA = 50

# Normalised error threshold (±) within which the target is considered centred
_CENTER_THRESHOLD = 0.1


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


def _nearest_blob_rect(mask: np.ndarray, frame_shape=None):
    """Return the bounding rect (x, y, w, h) of the largest blob by area.
    Returns None if no blob meets the minimum area threshold.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    valid = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= _MIN_BLOB_AREA]
    if not valid:
        return None
    best, _ = max(valid, key=lambda item: item[1])
    return cv2.boundingRect(best)


_CAL_HISTOGRAM_FILE = "color_histogram.txt"
_GAUSS_SIGMA        = 2.5   # confidence window: ±_GAUSS_SIGMA * std


def _load_histogram(path: str) -> np.ndarray | None:
    """Load a calibration histogram saved by calibrate_color.py, or return None."""
    try:
        hist = np.loadtxt(path).reshape(180, 1).astype(np.float32)
        print(f"[Seeker] Loaded calibration histogram from {path!r}")
        return hist
    except FileNotFoundError:
        print(f"[Seeker] No histogram file at {path!r} — falling back to HSV ranges")
        return None
    except Exception as e:
        print(f"[Seeker] WARNING: could not load histogram {path!r}: {e}")
        return None


def _fit_gaussian(hist: np.ndarray) -> tuple[float, float]:
    """Fit a weighted Gaussian to a 180-bin hue histogram.

    Returns (mean_hue, std_hue).  Handles circular wrap so the mean stays
    in [0, 179] even for colours that straddle the 0/179 boundary.
    """
    bins = np.arange(180, dtype=np.float32)
    h    = hist.flatten().astype(np.float32)
    total = h.sum()
    if total == 0:
        return 90.0, 30.0

    prob = h / total

    # Circular mean via unit-vector averaging
    angles = bins * (2.0 * np.pi / 180.0)
    cx = float(np.sum(np.cos(angles) * prob))
    cy = float(np.sum(np.sin(angles) * prob))
    mean_rad = np.arctan2(cy, cx)
    if mean_rad < 0:
        mean_rad += 2.0 * np.pi
    mean_hue = float(mean_rad * 180.0 / (2.0 * np.pi))

    # Circular std: wrap bins to [-90, 90] relative to mean, then compute variance
    diff = bins - mean_hue
    diff = ((diff + 90.0) % 180.0) - 90.0   # wrap to [-90, 90]
    var  = float(np.sum(diff ** 2 * prob))
    return mean_hue, float(np.sqrt(max(var, 1.0)))


def _confidence_hist(hist: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Return a copy of *hist* with bins outside mean ± _GAUSS_SIGMA*std zeroed.

    The resulting histogram carries the original hue weights for the confident
    bins only.  Back-projecting it onto an HSV frame produces non-zero values
    only for pixels whose hue falls within the 3-sigma confidence window.
    """
    bins = np.arange(180, dtype=np.float32)
    diff = np.abs(bins - mean)
    diff = np.minimum(diff, 180.0 - diff)          # circular wrap
    conf = hist.flatten().copy().astype(np.float32)
    conf[diff >= _GAUSS_SIGMA * std] = 0.0
    return conf.reshape(hist.shape)


class Seeker:
    def __init__(
        self,
        source: int | str = 0,
        window_name: str = "Seeker",
        capture_width: int | None = None,
        capture_height: int | None = None,
        crop: tuple[int | None, int | None, int | None, int | None] | None = None,
        histogram_file: str = _CAL_HISTOGRAM_FILE,
        show_histogram: bool = False,
        show_mask: bool = False,
    ):
        """
        source          : camera index (int) or video / image file path (str)
        window_name     : OpenCV display window title
        capture_width   : request this width from the camera before any crop
        capture_height  : request this height from the camera before any crop
        crop            : (offset_x, offset_y, width, height) ROI to cut from each
                          raw frame before any processing.  None = full frame.
        histogram_file  : path to ASCII histogram saved by calibrate_color.py;
                          if present it replaces the hardcoded HSV detection.
        show_histogram  : if True and a histogram is loaded, display it in a
                          separate window named "<window_name> — Histogram".
        show_mask       : if True, display the detection mask each frame in a
                          separate window named "<window_name> — Mask".
        """
        self.source          = source
        self.window_name     = window_name
        self.capture_width   = capture_width
        self.capture_height  = capture_height
        self.crop            = crop   # (x, y, w, h) or None
        self._show_histogram = show_histogram
        self._show_mask      = show_mask

        self._cal_hist        = _load_histogram(histogram_file)
        if self._cal_hist is not None:
            self._gauss_mean, self._gauss_std = _fit_gaussian(self._cal_hist)
            self._conf_hist = _confidence_hist(self._cal_hist, self._gauss_mean, self._gauss_std)
            kept = int((self._conf_hist.flatten() > 0).sum())
            print(f"[Seeker] Confidence hist: mean={self._gauss_mean:.1f}  "
                  f"std={self._gauss_std:.1f}  bins={kept}/180")
        else:
            self._gauss_mean = self._gauss_std = None
            self._conf_hist  = None
        self.cap              = None
        self._roi_hist        = self._conf_hist  # fixed: always use conf histogram
        self._track_win       = None   # current CamShift window (x, y, w, h)
        self._detect_count    = 0      # consecutive successful detections
        self._res_logged      = False
        self._term_crit  = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self):
        """Open the video source and create the display window."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source!r}")
        if self.capture_width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.capture_width)
        if self.capture_height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Seeker] Opened source={self.source!r}  capture={actual_w}x{actual_h}")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self._show_histogram and self._cal_hist is not None:
            self._hist_window = f"{self.window_name} — Histogram"
            cv2.namedWindow(self._hist_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._hist_window, 360, 230)
        else:
            self._hist_window = None
        if self._show_mask:
            self._mask_window = f"{self.window_name} — Mask"
            cv2.namedWindow(self._mask_window, cv2.WINDOW_NORMAL)
        else:
            self._mask_window = None

    def close(self):
        """Release the capture device and destroy the display window."""
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyWindow(self.window_name)
        if getattr(self, "_hist_window", None):
            cv2.destroyWindow(self._hist_window)
            self._hist_window = None
        if getattr(self, "_mask_window", None):
            cv2.destroyWindow(self._mask_window)
            self._mask_window = None

    # ── Tracking helpers ──────────────────────────────────────────────────────

    def _detection_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Return a binary detection mask.

        When a calibration histogram is loaded, back-projects the 3-sigma
        confidence histogram: only pixels whose hue bin has a non-zero weight
        in the confidence window are accepted, weighted by the actual histogram
        shape.  Saturation and value floors reject grey/dark pixels.
        Falls back to hardcoded HSV ranges otherwise.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if self._conf_hist is not None:
            bp  = cv2.calcBackProject([hsv], [0], self._conf_hist, [0, 180], 1)
            in_sat = hsv[:, :, 1] > 40
            in_val = hsv[:, :, 2] > 40
            mask = ((bp > 0) & in_sat & in_val).astype(np.uint8) * 255
        else:
            mask = _pink_mask(hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        return mask

    def track(self, frame: np.ndarray):
        """Run one tracking step.

        Returns (annotated_frame, cx, cy) where cx/cy is the tracked centre,
        or (annotated_frame, None, None) when no target is locked.
        """
        h_frame, w_frame = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._detection_mask(hsv)
        if getattr(self, "_mask_window", None):
            cv2.imshow(self._mask_window, mask)
        out  = frame.copy()

        rect = _nearest_blob_rect(mask, frame.shape)
        if rect is not None:
            self._track_win     = rect
            self._detect_count  = min(self._detect_count + 1, 5)
        else:
            self._detect_count  = 0
            self._track_win     = None

        if self._roi_hist is None or self._track_win is None or self._detect_count < 5:
            self._draw_center_cross(out, w_frame, h_frame)
            self._update_histogram_window()
            return out, None, None

        # ── CamShift step ─────────────────────────────────────────────────────
        back_proj = cv2.calcBackProject(
            [hsv], [0], self._roi_hist, [0, 180], 1
        )
        back_proj &= mask

        ret, self._track_win = cv2.CamShift(
            back_proj, self._track_win, self._term_crit
        )

        # ── Validate: drop lock if window collapsed ───────────────────────────
        _, _, w, h = self._track_win
        if w < 2 or h < 2:
            self._track_win    = None
            self._detect_count = 0
            self._draw_center_cross(out, w_frame, h_frame)
            self._update_histogram_window()
            return out, None, None

        cx = int(ret[0][0])
        cy = int(ret[0][1])

        # ── Draw rotated bounding box (green when centred, pink otherwise) ────
        ex = (cx - w_frame / 2.0) / (w_frame / 2.0)
        ey = (cy - h_frame / 2.0) / (h_frame / 2.0)
        centred    = abs(ex) < _CENTER_THRESHOLD and abs(ey) < _CENTER_THRESHOLD
        box_colour = (0, 255, 0) if centred else (203, 192, 255)
        pts = cv2.boxPoints(ret).astype(np.intp)
        cv2.polylines(out, [pts], True, box_colour, 2)

        # centroid crosshair
        cv2.line(out, (0, cy), (w_frame, cy), (0, 255, 255), 1)
        cv2.line(out, (cx, 0), (cx, h_frame), (0, 255, 255), 1)
        cv2.circle(out, (cx, cy), 5, (0, 255, 255), -1)

        self._draw_center_cross(out, w_frame, h_frame)
        self._update_histogram_window()
        return out, cx, cy

    @staticmethod
    def _render_histogram(hist, width=360, height=200) -> np.ndarray:
        """Render a hue histogram as a BGR image with hue-coloured bars and axis labels."""
        canvas = np.zeros((height + 30, width, 3), dtype=np.uint8)
        max_v  = float(hist.max()) or 1.0
        bar_w  = width / 180.0
        for i, v in enumerate(hist.flatten()):
            bar_h  = int(v / max_v * height)
            colour = cv2.cvtColor(
                np.uint8([[[int(i), 220, 200]]]), cv2.COLOR_HSV2BGR
            )[0][0].tolist()
            x0 = int(i * bar_w)
            x1 = max(x0 + 1, int((i + 1) * bar_w))
            cv2.rectangle(canvas, (x0, height - bar_h), (x1, height), colour, -1)
        # hue axis ticks every 30°
        for hue in range(0, 181, 30):
            x = int(hue * bar_w)
            cv2.line(canvas, (x, height), (x, height + 5), (200, 200, 200), 1)
            cv2.putText(canvas, str(hue), (max(0, x - 8), height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(canvas, "Hue (0-179)", (width // 2 - 35, height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        return canvas

    def _update_histogram_window(self):
        """Push the calibration histogram to its dedicated window (if open)."""
        if not getattr(self, "_hist_window", None) or self._cal_hist is None:
            return
        cv2.imshow(self._hist_window, self._render_histogram(self._cal_hist))

    @staticmethod
    def _draw_center_cross(frame, w, h, box=80, arm=16, color=(0, 0, 255), thickness=3):
        """Draw a corner-bracket crosshair (L-shapes at 4 corners of a box) with a small center cross."""
        cx, cy = w // 2, h // 2
        corners = [
            (cx - box, cy - box, +1, +1),
            (cx + box, cy - box, -1, +1),
            (cx + box, cy + box, -1, -1),
            (cx - box, cy + box, +1, -1),
        ]
        for x, y, dx, dy in corners:
            cv2.line(frame, (x, y), (x + arm * dx, y), color, thickness)
            cv2.line(frame, (x, y), (x, y + arm * dy), color, thickness)
        # center cross
        cs = 24
        cv2.line(frame, (cx - cs, cy), (cx + cs, cy), (0, 0, 255), 3)
        cv2.line(frame, (cx, cy - cs), (cx, cy + cs), (0, 0, 255), 3)

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
        errory =  -(cy - h / 2.0) / (h / 2.0)
        return float(errorx), float(errory)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def read_frame(self):
        """Return (ok, frame) from the capture device, cropped if configured."""
        ok, frame = self.cap.read()
        if ok and self.crop is not None:
            x, y, w, h = self.crop
            fh, fw = frame.shape[:2]
            if w is None:
                w = fw - x
            if h is None:
                h = fh - y
            frame = frame[y:y + h, x:x + w]
        if ok and not self._res_logged:
            fh, fw = frame.shape[:2]
            print(f"[Seeker] Resolution: {fw}x{fh}"
                  + (f"  (cropped from raw, offset {self.crop[:2]})" if self.crop else ""))
            self._res_logged = True
        return ok, frame

    def run(self):
        """Open source, run CamShift tracking loop; press 'q' to quit,
        'r' to reset the tracker."""
        self.open()
        prev_time = time.time()
        frame_times = collections.deque(maxlen=30)
        try:
            while True:
                curr_time = time.time()
                frame_times.append(curr_time - prev_time)
                prev_time = curr_time
                fps = 1.0 / (sum(frame_times) / len(frame_times))

                ok, frame = self.read_frame()
                if not ok:
                    print("[Seeker] End of stream.")
                    break

                annotated, cx, cy = self.track(frame)
                errorx, errory    = self.error_xy(cx, cy, frame.shape)

                if errorx is not None:
                    label = f"ex={errorx:+.3f}  ey={errory:+.3f}"
                    cv2.putText(annotated, label, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(self.window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[Seeker] Quit.")
                    break
                elif key == ord("r"):
                    self._track_win    = None
                    self._detect_count = 0
                    print("[Seeker] Tracker reset.")
        finally:
            self.close()
