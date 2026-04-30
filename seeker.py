import collections
import cv2
import numpy as np
import threading
import time

from hud_display import HudDisplay


# ── Picamera2 drop-in replacement for cv2.VideoCapture ───────────────────────
# On Raspberry Pi 5 the rp1-cfe CSI driver exposes /dev/video0-7 as media
# pipeline sub-devices, NOT V4L2 capture devices.  cv2.VideoCapture therefore
# always fails.  Picamera2Capture wraps picamera2/libcamera to give Seeker the
# same isOpened / set / get / read / release interface it expects.
class Picamera2Capture:
    def __init__(self, width: int = 1280, height: int = 720):
        from picamera2 import Picamera2
        self._w = width
        self._h = height
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={"size": (self._w, self._h), "format": "BGR888"}
        )
        self._cam.configure(cfg)
        self._cam.start()
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self._w = int(value)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self._h = int(value)
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
            frame = self._cam.capture_array()
            return True, frame
        except Exception:
            return False, None

    def release(self):
        if self._opened:
            self._cam.stop()
            self._opened = False


# ── Hot-pink HSV range (OpenCV hue 0-179) ────────────────────────────────────
# Hot pink sits at H ≈ 130-173 (≈300-330° on the standard wheel).
# High saturation floor (100) rejects pastel / pale pinks.
_PINK_RANGES = [
    (np.array([130, 40, 80]), np.array([173, 233, 233])),  # hot pink / magenta
]

# Minimum contour area to accept as a valid blob (pixels²)
_MIN_BLOB_AREA = 9

# Normalised error threshold (±) within which the target is considered centred
_CENTER_THRESHOLD = 0.1


def _pink_mask(hsv: np.ndarray) -> np.ndarray:
    """Return a binary mask of all pink pixels in *hsv* frame."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _PINK_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    # small morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask


_MIN_EXTENT   = 0.30   # minimum contour/bbox fill ratio
_MIN_SOLIDITY = 0.45   # minimum contour/convex-hull fill ratio (rejects L-shapes, noise)
_MIN_DIM      = 3      # minimum blob width AND height in pixels
_MAX_ASPECT   = 6.0    # maximum long/short side ratio (rejects thin slivers)


def _nearest_blob_rect(mask: np.ndarray, frame_shape=None,
                       box_filter: bool = True,
                       prefer_pt: tuple | None = None):
    """Return the bounding rect (x, y, w, h) of the best blob.

    When box_filter=True, candidates must pass four shape tests:
      - minimum area (_MIN_BLOB_AREA)
      - minimum dimension in both axes (_MIN_DIM)
      - extent  = contour_area / bbox_area        >= _MIN_EXTENT
      - solidity = contour_area / convex_hull_area >= _MIN_SOLIDITY
      - aspect ratio (long/short) <= _MAX_ASPECT
    Survivors are scored by solidity × extent × area so compact, filled,
    large blobs rank highest.  When prefer_pt=(px, py) is given the score is
    divided by (1 + dist/100) so blobs closer to the reference win ties.

    When box_filter=False, the largest blob by raw contour area is returned
    with no shape constraint.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best_score = -1.0
    best_rect  = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < _MIN_BLOB_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        if box_filter:
            # Dimension gate
            if w < _MIN_DIM or h < _MIN_DIM:
                continue
            # Aspect ratio gate
            aspect = max(w, h) / min(w, h)
            if aspect > _MAX_ASPECT:
                continue
            # Extent (fill of bounding box)
            extent = area / (w * h)
            if extent < _MIN_EXTENT:
                continue
            # Solidity (fill of convex hull)
            hull      = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity  = area / hull_area if hull_area > 0 else 0.0
            if solidity < _MIN_SOLIDITY:
                continue
            score = solidity * extent * area
        else:
            score = area
        # Soft bias toward reference point when provided
        if prefer_pt is not None:
            bcx  = x + w * 0.5
            bcy  = y + h * 0.5
            dist = ((bcx - prefer_pt[0]) ** 2 + (bcy - prefer_pt[1]) ** 2) ** 0.5
            score /= (1.0 + dist / 100.0)
        if score > best_score:
            best_score = score
            best_rect  = (x, y, w, h)
    return best_rect


_CAL_HISTOGRAM_FILE = "color_histogram.txt"
_GAUSS_SIGMA        = 2.0   # confidence window: ±2.0σ

# ── Kalman filter tuning (position tracking) ──────────────────────────────────
_KF_Q_POS    = 2.0    # process noise — position  (px^2/s)
_KF_Q_VEL    = 80.0   # process noise — velocity  (px^2/s^3)
_KF_R        = 30.0   # measurement noise         (px^2)
_KF_MISS_MAX = 5      # predict this many frames after lock loss, then give up


def _make_tracker(name: str):
    """Create a cv2 MIL tracker.

    Handles three API variants across OpenCV versions:
      - free function  cv2.TrackerMIL_create()  (OpenCV ≤ 4.4 / contrib)
      - legacy ns      cv2.legacy.TrackerMIL_create()
      - class method   cv2.TrackerMIL.create()  (OpenCV 4.5+)
    """
    if name.lower() != "mil":
        raise ValueError(f"Unknown tracker '{name}'. Only 'mil' is supported.")
    try:
        if hasattr(cv2, "TrackerMIL_create"):
            return cv2.TrackerMIL_create()
        legacy = getattr(cv2, "legacy", None)
        if legacy and hasattr(legacy, "TrackerMIL_create"):
            return legacy.TrackerMIL_create()
        if hasattr(cv2, "TrackerMIL"):
            return cv2.TrackerMIL.create()
        raise RuntimeError(
            f"TrackerMIL not available in this OpenCV build (cv2 {cv2.__version__}). "
            f"Try: pip install opencv-contrib-python"
        )
    except cv2.error as exc:
        raise RuntimeError(f"TrackerMIL failed to create: {exc}") from exc


def _kf1d(x0, x1, P00, P01, P10, P11, meas, dt):
    """One step of a 1-D constant-velocity Kalman filter.

    State [pos, vel], observation = pos only.
    Returns (filtered_pos, nx0, nx1, nP00, nP01, nP10, nP11).
    """
    # ── Predict ───────────────────────────────────────────────────────────────
    px0  = x0 + x1 * dt
    px1  = x1
    pp00 = P00 + dt * (P10 + P01) + dt * dt * P11 + _KF_Q_POS
    pp01 = P01 + dt * P11
    pp10 = P10 + dt * P11
    pp11 = P11 + _KF_Q_VEL
    # ── Update ────────────────────────────────────────────────────────────────
    S_inv = 1.0 / (pp00 + _KF_R)
    K0    = pp00 * S_inv
    K1    = pp10 * S_inv
    innov = meas - px0
    nx0   = px0 + K0 * innov
    nx1   = px1 + K1 * innov
    nP00  = (1.0 - K0) * pp00
    nP01  = (1.0 - K0) * pp01
    nP10  = pp10 - K1 * pp00
    nP11  = pp11 - K1 * pp01
    return nx0, nx1, nP00, nP01, nP10, nP11


def _kf1d_pred(x0, x1, P00, P01, P10, P11, dt):
    """Kalman predict-only step (no measurement).  Propagates state and covariance."""
    px0  = x0 + x1 * dt
    px1  = x1
    pp00 = P00 + dt * (P10 + P01) + dt * dt * P11 + _KF_Q_POS
    pp01 = P01 + dt * P11
    pp10 = P10 + dt * P11
    pp11 = P11 + _KF_Q_VEL
    return px0, px1, pp00, pp01, pp10, pp11



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
    """Return a copy of *hist* with bins outside mean ± 2.5σ zeroed.

    The resulting histogram carries the original hue weights for the confident
    bins only.  Back-projecting it onto an HSV frame produces non-zero values
    only for pixels whose hue falls within the 2.5σ confidence window.
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
        mask_algo: str = "all",
        use_camshift: bool = True,
        shift_algo: str = "camshift",   # "camshift" | "meanshift"
        box_filter: bool = True,
        use_kalman: bool = True,
        tracker: str = "",
        pitch_offset_norm: float = 0.0,
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
        mask_algo       : which detection algorithm(s) to use when a calibration
                          histogram is loaded.  One of:
                            "gaussian"  — Gaussian back-projection only (Method 1)
                            "adaptive"  — Adaptive hue threshold only   (Method 2)
                            "inrange"   — Dual inRange only              (Method 3)
                            "all"       — 2-of-3 majority vote (default)
        box_filter      : if True, reject blobs whose extent (contour/bbox area)
                          is below _MIN_EXTENT — prefers rectangular targets.
        """
        self.source          = source
        self.window_name     = window_name
        self.capture_width   = capture_width
        self.capture_height  = capture_height
        self.crop            = crop   # (x, y, w, h) or None
        self._show_histogram = show_histogram
        self._show_mask      = show_mask
        self._mask_algo      = mask_algo
        self._use_camshift   = use_camshift
        self._shift_algo     = shift_algo
        self._box_filter     = box_filter
        self._use_kalman     = use_kalman
        self._tracker_name      = tracker.lower() if tracker else ""
        self._tracker_obj       = None
        self.pitch_offset_norm  = pitch_offset_norm

        self._cal_hist        = _load_histogram(histogram_file)
        if self._cal_hist is not None:
            self._gauss_mean, self._gauss_std = _fit_gaussian(self._cal_hist)
            self._conf_hist = _confidence_hist(self._cal_hist, self._gauss_mean, self._gauss_std)
            kept = int((self._conf_hist.flatten() > 0).sum())
            print(f"[Seeker] Confidence hist: mean={self._gauss_mean:.1f}  "
                  f"std={self._gauss_std:.1f}  bins={kept}/180")
            # Wider histogram for CamShift tracking (3σ) — more signal than 2σ conf_hist
            # but still selective enough to avoid false positives in back-projection.
            _roi = self._cal_hist.flatten().copy().astype(np.float32)
            _d   = np.abs(np.arange(180, dtype=np.float32) - self._gauss_mean)
            _d   = np.minimum(_d, 180.0 - _d)
            _roi[_d >= 3.0 * self._gauss_std] = 0.0
            self._roi_hist = _roi.reshape(180, 1)
        else:
            self._gauss_mean = self._gauss_std = None
            self._conf_hist  = None
            self._roi_hist   = None
        self.cap              = None
        self._track_win       = None   # current CamShift window (x, y, w, h)
        self._detect_count    = 0      # consecutive successful detections
        self._res_logged      = False
        self._term_crit  = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.5
        )
        self._win_w_ema  = 0.0   # EMA of CamShift window width
        self._win_h_ema  = 0.0   # EMA of CamShift window height
        self._EMA_ALPHA  = 0.3   # EMA smoothing factor (lower = smoother)
        # Kalman filter state (independent x/y axes, constant-velocity model)
        self._kf_x0 = 0.0;  self._kf_x1 = 0.0   # x: [position, velocity]
        self._kf_Px = [1.0, 0.0, 0.0, 1.0]       # x: P row-major [P00,P01,P10,P11]
        self._kf_y0 = 0.0;  self._kf_y1 = 0.0   # y: [position, velocity]
        self._kf_Py = [1.0, 0.0, 0.0, 1.0]       # y: P row-major
        self._kf_initialized = False
        self._kf_last_t      = 0.0
        self._miss_count     = 0     # consecutive predict-only frames since lock lost
        # Precompute morphological kernels — avoids per-frame allocation
        self._kern3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kern5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Precompute hue LUT and inRange bounds (fixed once calibrated)
        if self._gauss_mean is not None and self._gauss_std is not None:
            bins = np.arange(180, dtype=np.float32)
            d    = np.abs(bins - self._gauss_mean)
            self._hue_dist_lut = np.minimum(d, 180.0 - d).astype(np.float32)
            # Boolean LUT: True where hue bin is within ±GAUSS_SIGMA*std
            sigma_thresh = _GAUSS_SIGMA * self._gauss_std
            self._hue_gate_lut = (self._hue_dist_lut < sigma_thresh).astype(np.uint8) * 255
            # Precompute inRange band bounds (core = ±1σ, outer = ±Nσ)
            self._inrange_core  = self._gauss_mean, self._gauss_std
            self._inrange_outer = self._gauss_mean, _GAUSS_SIGMA * self._gauss_std
            # Precompute np.array bounds for inRange calls (avoids per-frame allocation)
            self._precomp_bands = self._build_inrange_bounds()
        else:
            self._hue_dist_lut  = None
            self._hue_gate_lut  = None
            self._inrange_core  = None
            self._inrange_outer = None
            self._precomp_bands = None
        # Persistent buffer for H channel copy in _mask_gaussian
        self._h_buf = None
        # Persistent output and HSV buffers — allocated on first frame, reused after
        self._out_buf = None
        self._hsv_buf = None
        # LUT: maps inRange output (binary 0/255) to half-weight (0/128) in one call
        _lut = np.zeros(256, dtype=np.uint8)
        _lut[255] = 128
        self._outer_lut = _lut

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self):
        """Open the video source and create the display window."""
        if isinstance(self.source, int):
            w = self.capture_width  or 1280
            h = self.capture_height or 720
            self.cap = Picamera2Capture(w, h)
        else:
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
        # Start background capture thread so read_frame() never blocks on I/O.
        self._cap_lock  = threading.Lock()
        self._cap_stop  = False
        self._cap_ok    = False
        self._cap_frame = None
        self._cap_seq   = 0
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, actual_w, actual_h)
        if self._show_histogram and self._cal_hist is not None:
            self._hist_window = f"{self.window_name} — Histogram"
            cv2.namedWindow(self._hist_window, cv2.WINDOW_NORMAL|cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self._hist_window, 360, 230)
        else:
            self._hist_window = None
        if self._show_mask:
            self._mask_window = f"{self.window_name} — Mask"
            cv2.namedWindow(self._mask_window, cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._mask_window, actual_w, actual_h)
        else:
            self._mask_window = None

    def _capture_loop(self):
        """Background thread: continuously read frames, always keep the latest."""
        while not self._cap_stop:
            ok, frame = self.cap.read()
            with self._cap_lock:
                self._cap_ok    = ok
                self._cap_frame = frame
                self._cap_seq  += 1

    def close(self):
        """Release the capture device and destroy the display window."""
        if getattr(self, "_cap_thread", None):
            self._cap_stop = True
            self._cap_thread.join(timeout=1.0)
            self._cap_thread = None
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

    def _build_inrange_bounds(self) -> dict:
        """Precompute all np.array bounds for inRange calls — called once at init."""
        result = {}
        for key, (mean, hw) in [("core",  self._inrange_core),
                                 ("outer", self._inrange_outer)]:
            lo, hi = mean - hw, mean + hw
            if lo < 0:
                result[key] = (
                    np.array([max(0, int(lo + 180)), 40, 40],  dtype=np.uint8),
                    np.array([179,                   255, 255], dtype=np.uint8),
                    np.array([0,              40,  40],         dtype=np.uint8),
                    np.array([min(179, int(hi)), 255, 255],     dtype=np.uint8),
                    "wrap_lo",
                )
            elif hi > 179:
                result[key] = (
                    np.array([max(0, int(lo)), 40, 40],         dtype=np.uint8),
                    np.array([179,             255, 255],        dtype=np.uint8),
                    np.array([0,                        40, 40], dtype=np.uint8),
                    np.array([min(179, int(hi-180)), 255, 255],  dtype=np.uint8),
                    "wrap_hi",
                )
            else:
                result[key] = (
                    np.array([max(0, int(lo)), 40, 40],   dtype=np.uint8),
                    np.array([min(179, int(hi)), 255, 255], dtype=np.uint8),
                    None, None, "single",
                )
        return result

    def _apply_inrange_band(self, hsv: np.ndarray, key: str) -> np.ndarray:
        """Apply a precomputed inRange band by key ('core' or 'outer')."""
        lo_a, hi_a, lo_b, hi_b, mode = self._precomp_bands[key]
        if mode == "single":
            return cv2.inRange(hsv, lo_a, hi_a)
        return cv2.bitwise_or(cv2.inRange(hsv, lo_a, hi_a),
                              cv2.inRange(hsv, lo_b, hi_b))

    # ── Detection sub-masks ───────────────────────────────────────────────────

    def _mask_gaussian(self, hsv: np.ndarray, h_blur: np.ndarray) -> np.ndarray:
        """Method 1 — Gaussian back-projection with S/V gate."""
        # Use pre-allocated buffer to avoid repeated allocation
        if self._h_buf is None or self._h_buf.shape != hsv[:, :, 0].shape:
            self._h_buf = hsv[:, :, 0].copy()
        else:
            np.copyto(self._h_buf, hsv[:, :, 0])
        hsv[:, :, 0] = h_blur
        bp = cv2.calcBackProject([hsv], [0], self._conf_hist, [0, 180], 1)
        hsv[:, :, 0] = self._h_buf
        sv_ok = self._apply_inrange_band(hsv, "outer")   # reuse outer band as S/V gate
        return cv2.bitwise_and(cv2.threshold(bp, 0, 255, cv2.THRESH_BINARY)[1], sv_ok)

    def _mask_adaptive(self, hsv: np.ndarray, h_blur: np.ndarray) -> np.ndarray:
        """Method 2 — Adaptive hue threshold + precomputed σ-gate LUT."""
        # h_blur is uint8 0-179; use directly — skips per-frame normalize.
        # blockSize=11 (down from 21) halves the neighbourhood work.
        adapt  = cv2.adaptiveThreshold(
            h_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11, C=3,
        )
        hue_gate = self._hue_gate_lut[h_blur]
        return cv2.bitwise_and(adapt, hue_gate)

    def _mask_inrange(self, hsv: np.ndarray) -> np.ndarray:
        """Method 3 — Two-band inRange using fully precomputed bounds."""
        core  = self._apply_inrange_band(hsv, "core")
        outer = self._apply_inrange_band(hsv, "outer")
        # outer-only pixels get half weight; core pixels get full 255
        cv2.subtract(outer, core, dst=outer)          # removes core pixels, no temp alloc
        outer = cv2.LUT(outer, self._outer_lut)       # 255→128, 0→0
        return cv2.bitwise_or(core, outer)

    # ── Combined detection mask ───────────────────────────────────────────────

    def _detection_mask(self, hsv: np.ndarray,
                        locked: bool = False) -> tuple[np.ndarray, int]:
        """Return (mask, scale=1) always full-res.

        Locked     → fast inRange only (no back-projection or adaptive).
        Acquiring  → full pipeline matching calibrate_color exactly.
        """
        if self._conf_hist is None:
            return _pink_mask(hsv), 1

        if locked:
            mask = self._mask_inrange(hsv)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kern3)
            return mask, 1

        # ── Acquisition: full-res ─────────────────────────────────────────────
        algo = self._mask_algo

        if algo == "inrange":
            # Fastest path — no blur needed
            mask = self._mask_inrange(hsv)
        else:
            h_blur = cv2.GaussianBlur(hsv[:, :, 0], (5, 5), 0)
            if algo == "gaussian":
                mask = self._mask_gaussian(hsv, h_blur)
            elif algo == "adaptive":
                mask = self._mask_adaptive(hsv, h_blur)
            else:  # "all" — 2-of-3 majority vote with in-place accumulation
                m1 = self._mask_gaussian(hsv, h_blur)
                m2 = self._mask_adaptive(hsv, h_blur)
                m3 = self._mask_inrange(hsv)
                # In-place vote: reuse m1 buffer
                votes = (m1 > 0).view(np.uint8)
                votes += (m2 > 0).view(np.uint8)
                votes += (m3 > 0).view(np.uint8)
                _, mask = cv2.threshold(votes, 1, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   self._kern5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self._kern5)
        return mask, 1

    def track(self, frame: np.ndarray):
        """Run one tracking step.

        Returns (annotated_frame, cx, cy) where cx/cy is the tracked centre,
        or (annotated_frame, None, None) when no target is locked.

        Fast path (locked): skips the expensive 3-method detection pipeline and
        uses only a single inRange mask to gate CamShift (~3–4× faster per frame).
        Detection path (acquiring): runs the full 3-method pipeline on a
        half-resolution frame to find and confirm the blob.
        """
        h_frame, w_frame = frame.shape[:2]
        # Reuse persistent buffers; (re)allocate only when frame shape changes.
        if self._hsv_buf is None or self._hsv_buf.shape[:2] != (h_frame, w_frame):
            self._hsv_buf = np.empty((h_frame, w_frame, 3), dtype=np.uint8)
            self._out_buf = np.empty_like(frame)
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self._hsv_buf)
        hsv = self._hsv_buf
        np.copyto(self._out_buf, frame)
        out = self._out_buf

        if not self._use_camshift:
            # ── Detection-only path (no CamShift) ─────────────────────────────
            mask, _ = self._detection_mask(hsv)
            if getattr(self, "_mask_window", None):
                cv2.imshow(self._mask_window, mask)
            rect = _nearest_blob_rect(mask, frame.shape, self._box_filter)
            if rect is None:
                self._update_histogram_window()
                return out, None, None
            x, y, w, h = rect
            cx = x + w // 2
            cy = y + h // 2
            ex = (cx - w_frame / 2.0) / (w_frame / 2.0)
            ey = -(cy - h_frame / 2.0) / (h_frame / 2.0)
            centred    = abs(ex) < _CENTER_THRESHOLD and abs(ey) < _CENTER_THRESHOLD
            box_colour = (0, 233, 0) if centred else (180, 105, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), box_colour, 2)
            cv2.line(out, (0, cy), (w_frame, cy), (0, 233, 233), 1)
            cv2.line(out, (cx, 0), (cx, h_frame), (0, 233, 233), 1)
            cv2.circle(out, (cx, cy), 3, (0, 233, 233), -1)
            self._update_histogram_window()
            return out, cx, cy

        locked = (self._roi_hist is not None and
                  self._track_win is not None and
                  self._detect_count >= 3)

        # ── Detection / re-acquisition ────────────────────────────────────────
        mask = None
        blob = None
        if not locked:
            if self._track_win is not None and self._detect_count > 0:
                # Re-acquiring: restrict search to a padded region around the
                # last known window — avoids running the full pipeline on the
                # whole frame every cycle.
                twx, twy, tww, twh = self._track_win
                pad = max(tww, twh, 40)
                sx1 = max(0, twx - pad);      sy1 = max(0, twy - pad)
                sx2 = min(w_frame, twx + tww + pad)
                sy2 = min(h_frame, twy + twh + pad)
                mask_crop, _ = self._detection_mask(hsv[sy1:sy2, sx1:sx2])
                blob_crop     = _nearest_blob_rect(mask_crop, None, self._box_filter)
                if blob_crop is not None:
                    bx, by, bw, bh = blob_crop
                    blob = (bx + sx1, by + sy1, bw, bh)
                if getattr(self, "_mask_window", None):
                    mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
                    mask[sy1:sy2, sx1:sx2] = mask_crop
            else:
                # Cold acquisition: search full frame.
                mask, _ = self._detection_mask(hsv)
                blob     = _nearest_blob_rect(mask, frame.shape, self._box_filter)

            if blob is not None:
                bx, by, bw, bh = blob
                pad = max(8, int(max(bw, bh) * 0.3))
                ix = max(0, bx - pad)
                iy = max(0, by - pad)
                iw = min(w_frame - ix, bw + 2 * pad)
                ih = min(h_frame - iy, bh + 2 * pad)
                self._track_win    = (ix, iy, iw, ih)
                self._detect_count = min(self._detect_count + 1, 3)
                if self._detect_count == 1:
                    self._win_w_ema = float(iw)
                    self._win_h_ema = float(ih)
                if self._tracker_name and self._detect_count == 3:
                    try:
                        self._tracker_obj = _make_tracker(self._tracker_name)
                        self._tracker_obj.init(frame, (ix, iy, iw, ih))
                    except (RuntimeError, ValueError) as exc:
                        print(f"[Seeker] {exc}")
                        print("[Seeker] Falling back to CamShift.")
                        self._tracker_name = ""
                        self._tracker_obj  = None
            else:
                self._detect_count = 0
                self._track_win    = None

        if mask is not None and getattr(self, "_mask_window", None):
            cv2.imshow(self._mask_window, mask)

        if self._roi_hist is None or self._track_win is None or self._detect_count < 3:
            self._update_histogram_window()
            return out, None, None

        # ── MIL step (replaces CamShift when --tracker mil is active) ────────
        if self._tracker_name and self._tracker_obj is not None:
            ok, bbox = self._tracker_obj.update(frame)
            if ok:
                x, y, w, h = (int(v) for v in bbox)
                if w >= 4 and h >= 4:
                    # ── Color validation: confirm ROI still contains target ─────
                    x  = max(0, min(x, w_frame - 1))
                    y  = max(0, min(y, h_frame - 1))
                    w  = min(w, w_frame - x)
                    h  = min(h, h_frame - y)
                    roi_mask, _ = self._detection_mask(hsv[y:y + h, x:x + w])
                    if cv2.countNonZero(roi_mask) < _MIN_BLOB_AREA:
                        self._miss_count += 1
                        if self._miss_count >= _KF_MISS_MAX:
                            self._track_win      = None
                            self._detect_count   = 0
                            self._win_w_ema      = 0.0
                            self._win_h_ema      = 0.0
                            self._kf_initialized = False
                            self._miss_count     = 0
                            self._tracker_obj    = None
                        self._update_histogram_window()
                        return out, None, None
                    self._miss_count = 0
                    self._track_win = (x, y, w, h)
                    self._win_w_ema = self._EMA_ALPHA * w + (1 - self._EMA_ALPHA) * self._win_w_ema
                    self._win_h_ema = self._EMA_ALPHA * h + (1 - self._EMA_ALPHA) * self._win_h_ema
                    raw_cx = x + w / 2.0
                    raw_cy = y + h / 2.0
                    if self._use_kalman:
                        kf_dt = 0.0
                        now_t = time.monotonic()
                        kf_dt = min(now_t - self._kf_last_t, 0.5) if self._kf_initialized else 0.0
                        self._kf_last_t = now_t
                        if not self._kf_initialized:
                            self._kf_x0, self._kf_x1 = raw_cx, 0.0
                            self._kf_y0, self._kf_y1 = raw_cy, 0.0
                            self._kf_Px = [1.0, 0.0, 0.0, 1.0]
                            self._kf_Py = [1.0, 0.0, 0.0, 1.0]
                            self._kf_initialized = True
                            cx, cy = int(round(raw_cx)), int(round(raw_cy))
                        else:
                            self._kf_x0, self._kf_x1, *self._kf_Px = _kf1d(
                                self._kf_x0, self._kf_x1, *self._kf_Px, raw_cx, kf_dt)
                            self._kf_y0, self._kf_y1, *self._kf_Py = _kf1d(
                                self._kf_y0, self._kf_y1, *self._kf_Py, raw_cy, kf_dt)
                            cx = int(round(self._kf_x0))
                            cy = int(round(self._kf_y0))
                    else:
                        cx, cy = int(round(raw_cx)), int(round(raw_cy))
                    cx = max(0, min(cx, w_frame - 1))
                    cy = max(0, min(cy, h_frame - 1))
                    ex = (cx - w_frame / 2.0) / (w_frame / 2.0)
                    ey = -(cy - h_frame / 2.0) / (h_frame / 2.0)
                    centred    = abs(ex) < _CENTER_THRESHOLD and abs(ey) < _CENTER_THRESHOLD
                    box_colour = (0, 233, 0) if centred else (180, 105, 255)
                    cv2.rectangle(out, (x, y), (x + w, y + h), box_colour, 2)
                    cv2.line(out, (0, cy), (w_frame, cy), (0, 233, 233), 1)
                    cv2.line(out, (cx, 0), (cx, h_frame), (0, 233, 233), 1)
                    cv2.circle(out, (cx, cy), 3, (0, 233, 233), -1)
                    self._update_histogram_window()
                    return out, cx, cy
            # Tracker update failed or bbox too small — hard reset
            self._track_win      = None
            self._detect_count   = 0
            self._win_w_ema      = 0.0
            self._win_h_ema      = 0.0
            self._kf_initialized = False
            self._miss_count     = 0
            self._tracker_obj    = None
            self._update_histogram_window()
            return out, None, None

        # ── CamShift step ─────────────────────────────────────────────────────
        kf_dt = 0.0
        if self._use_kalman:
            now_t = time.monotonic()
            kf_dt = min(now_t - self._kf_last_t, 0.5) if self._kf_initialized else 0.0
            self._kf_last_t = now_t

        back_proj = cv2.calcBackProject([hsv], [0], self._roi_hist, [0, 180], 1)
        # Pre-translate the search window by Kalman-predicted velocity so CamShift
        # starts near where the target is expected to be this frame.
        twx, twy, tww, twh = self._track_win
        if self._use_kalman and self._kf_initialized and kf_dt > 0:
            dx = int(round(self._kf_x1 * kf_dt))
            dy = int(round(self._kf_y1 * kf_dt))
            twx = max(0, min(twx + dx, w_frame - tww))
            twy = max(0, min(twy + dy, h_frame - twh))
            self._track_win = (twx, twy, tww, twh)
        # Gate to padded search window instead of full-frame mask.
        pad_x = max(tww // 2, 20);  pad_y = max(twh // 2, 20)
        bx1 = max(0, twx - pad_x);  by1 = max(0, twy - pad_y)
        bx2 = min(w_frame, twx + tww + pad_x)
        by2 = min(h_frame, twy + twh + pad_y)
        if by1 > 0:        back_proj[:by1, :]       = 0
        if by2 < h_frame:  back_proj[by2:, :]       = 0
        if bx1 > 0:        back_proj[by1:by2, :bx1] = 0
        if bx2 < w_frame:  back_proj[by1:by2, bx2:] = 0
        cv2.GaussianBlur(back_proj, (3, 3), 0, dst=back_proj)
        cv2.dilate(back_proj, back_proj, self._kern5)

        if self._shift_algo == "meanshift":
            _, self._track_win = cv2.meanShift(
                back_proj, self._track_win, self._term_crit
            )
            ret = None
        else:
            ret, self._track_win = cv2.CamShift(
                back_proj, self._track_win, self._term_crit
            )

        # ── Clamp window to frame bounds ──────────────────────────────────────
        twx, twy, tww, twh = self._track_win
        twx = max(0, min(twx, w_frame - 1))
        twy = max(0, min(twy, h_frame - 1))
        tww = max(1, min(tww, w_frame - twx))
        twh = max(1, min(twh, h_frame - twy))
        self._track_win = (twx, twy, tww, twh)

        # ── Validate: drop lock if window collapsed or exploded ───────────────
        _, _, w, h = self._track_win
        camshift_bad = (w < 4 or h < 4 or w > w_frame * 0.9 or h > h_frame * 0.9)

        if not camshift_bad:
            # ── Signal density check — drop lock if back-proj is mostly empty ─
            wx, wy, ww, wh = self._track_win
            roi_bp  = back_proj[wy:wy + wh, wx:wx + ww]
            density = float(roi_bp.mean()) / 255.0
            if density < 0.05:
                camshift_bad = True

        if camshift_bad:
            if self._use_kalman and self._kf_initialized and self._miss_count < _KF_MISS_MAX:
                # ── Predict position for up to _KF_MISS_MAX frames ────────────
                self._miss_count += 1
                pred_dt = max(kf_dt, 1.0 / 30.0)
                self._kf_x0, self._kf_x1, *self._kf_Px = _kf1d_pred(
                    self._kf_x0, self._kf_x1, *self._kf_Px, pred_dt)
                self._kf_y0, self._kf_y1, *self._kf_Py = _kf1d_pred(
                    self._kf_y0, self._kf_y1, *self._kf_Py, pred_dt)
                pcx = max(0, min(int(round(self._kf_x0)), w_frame - 1))
                pcy = max(0, min(int(round(self._kf_y0)), h_frame - 1))
                # Move the search window to the predicted centre so CamShift
                # can re-acquire on the next frame.
                tw = int(self._win_w_ema) or 40
                th = int(self._win_h_ema) or 40
                self._track_win = (max(0, pcx - tw // 2), max(0, pcy - th // 2),
                                   min(tw, w_frame), min(th, h_frame))
                cv2.circle(out, (pcx, pcy), 5, (0, 165, 255), 2)   # orange = predicting
                cv2.line(out, (0, pcy), (w_frame, pcy), (0, 165, 255), 1)
                cv2.line(out, (pcx, 0), (pcx, h_frame), (0, 165, 255), 1)
                self._update_histogram_window()
                return out, pcx, pcy
            else:
                # Prediction budget exhausted — hard reset
                self._track_win      = None
                self._detect_count   = 0
                self._win_w_ema      = 0.0
                self._win_h_ema      = 0.0
                self._kf_initialized = False
                self._miss_count     = 0
                self._tracker_obj    = None
                self._update_histogram_window()
                return out, None, None

        # ── EMA smoothing of window size ──────────────────────────────────────
        self._win_w_ema = self._EMA_ALPHA * w + (1 - self._EMA_ALPHA) * self._win_w_ema
        self._win_h_ema = self._EMA_ALPHA * h + (1 - self._EMA_ALPHA) * self._win_h_ema

        # ── Snap: if blob disagrees with tracker centre, correct to blob ────────
        shift_cx = float(ret[0][0]) if ret is not None else (self._track_win[0] + self._track_win[2] / 2.0)
        shift_cy = float(ret[0][1]) if ret is not None else (self._track_win[1] + self._track_win[3] / 2.0)
        if blob is not None:
            cs_cx = int(shift_cx)
            cs_cy = int(shift_cy)
            bx, by, bw, bh = blob
            b_cx = bx + bw // 2
            b_cy = by + bh // 2
            dist  = ((cs_cx - b_cx) ** 2 + (cs_cy - b_cy) ** 2) ** 0.5
            if dist > max(bw, bh) * 0.5:
                pad = max(8, int(max(bw, bh) * 0.3))
                self._track_win    = (max(0, bx - pad), max(0, by - pad),
                                      min(w_frame - max(0, bx - pad), bw + 2 * pad),
                                      min(h_frame - max(0, by - pad), bh + 2 * pad))
                self._detect_count = max(self._detect_count - 1, 0)

        raw_cx = shift_cx
        raw_cy = shift_cy

        if self._use_kalman:
            # ── Track recovered — reset prediction counter ────────────────────
            self._miss_count = 0
            # ── Kalman filter on position ─────────────────────────────────────
            if not self._kf_initialized:
                self._kf_x0, self._kf_x1 = raw_cx, 0.0
                self._kf_y0, self._kf_y1 = raw_cy, 0.0
                self._kf_Px = [1.0, 0.0, 0.0, 1.0]
                self._kf_Py = [1.0, 0.0, 0.0, 1.0]
                self._kf_initialized = True
                cx, cy = int(round(raw_cx)), int(round(raw_cy))
            else:
                self._kf_x0, self._kf_x1, *self._kf_Px = _kf1d(
                    self._kf_x0, self._kf_x1, *self._kf_Px, raw_cx, kf_dt)
                self._kf_y0, self._kf_y1, *self._kf_Py = _kf1d(
                    self._kf_y0, self._kf_y1, *self._kf_Py, raw_cy, kf_dt)
                cx = int(round(self._kf_x0))
                cy = int(round(self._kf_y0))
        else:
            cx, cy = int(round(raw_cx)), int(round(raw_cy))
        cx = max(0, min(cx, w_frame - 1))
        cy = max(0, min(cy, h_frame - 1))

        # ── Draw rotated bounding box (green when centred, red otherwise) ──────
        ex = (cx - w_frame / 2.0) / (w_frame / 2.0)
        ey = -(cy - h_frame / 2.0) / (h_frame / 2.0)
        centred    = abs(ex) < _CENTER_THRESHOLD and abs(ey) < _CENTER_THRESHOLD
        box_colour = (0, 233, 0) if centred else (180, 105, 255)
        if ret is not None:
            pts = cv2.boxPoints(ret).astype(np.intp)
            cv2.polylines(out, [pts], True, box_colour, 2)
        else:
            twx, twy, tww, twh = self._track_win
            cv2.rectangle(out, (twx, twy), (twx + tww, twy + twh), box_colour, 2)

        # centroid crosshair
        cv2.line(out, (0, cy), (w_frame, cy), (0, 233, 233), 1)
        cv2.line(out, (cx, 0), (cx, h_frame), (0, 233, 233), 1)
        cv2.circle(out, (cx, cy), 3, (0, 233, 233), -1)

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
            cv2.line(canvas, (x, height), (x, height + 3), (200, 200, 200), 1)
            cv2.putText(canvas, str(hue), (max(0, x - 8), height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)
        cv2.putText(canvas, "Hue (0-179)", (width // 2 - 33, height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)
        return canvas

    def _update_histogram_window(self):
        """Push the calibration histogram to its dedicated window (if open)."""
        if not getattr(self, "_hist_window", None) or self._cal_hist is None:
            return
        cv2.imshow(self._hist_window, self._render_histogram(self._cal_hist))

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
        """Return (seq, ok, frame) from the background capture buffer, cropped if configured.

        seq increments each time the capture thread delivers a new frame.
        Callers can skip processing when seq is unchanged from the previous call.
        """
        with self._cap_lock:
            seq, ok, frame = self._cap_seq, self._cap_ok, self._cap_frame
        if frame is None:
            return 0, False, None
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
        return seq, ok, frame

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

                _seq, ok, frame = self.read_frame()
                if frame is None:
                    continue   # capture thread not ready yet
                if not ok:
                    print("[Seeker] End of stream.")
                    break

                annotated, cx, cy = self.track(frame)
                errorx, errory    = self.error_xy(cx, cy, frame.shape)

                if errorx is not None:
                    label = f"ex={errorx:+.3f}  ey={errory:+.3f}"
                    cv2.putText(annotated, label, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 233, 0), 2)

                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 233, 0), 2)

                cv2.imshow(self.window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[Seeker] Quit.")
                    break
                elif key == ord("r"):
                    self._track_win    = None
                    self._detect_count = 0
                    self._win_w_ema    = 0.0
                    self._win_h_ema    = 0.0
                    print("[Seeker] Tracker reset.")
        finally:
            self.close()
