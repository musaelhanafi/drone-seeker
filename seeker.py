import collections
import ctypes
import cv2
import numpy as np
import os
import sys
import threading
import time

from hud_display import HudDisplay


# ── Picamera2 drop-in replacement for cv2.VideoCapture ───────────────────────
# On Raspberry Pi 5 the rp1-cfe CSI driver exposes /dev/video0-7 as media
# pipeline sub-devices, NOT V4L2 capture devices.  cv2.VideoCapture therefore
# always fails.  Picamera2Capture wraps picamera2/libcamera to give Seeker the
# same isOpened / set / get / read / release interface it expects.
class Picamera2Capture:
    def __init__(self, width: int = 1280, height: int = 720, flip: bool = False):
        from picamera2 import Picamera2
        from libcamera import Transform
        self._w = width
        self._h = height
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            # Picamera2 format names are byte-order reversed vs the numpy array:
            # "RGB888" yields a B,G,R array (OpenCV BGR), "BGR888" yields R,G,B.
            # We feed capture_array() straight to OpenCV, so request "RGB888" to
            # get true BGR — otherwise R/B are swapped and everything looks red.
            main={"size": (self._w, self._h), "format": "RGB888"},
            transform=Transform(hflip=1, vflip=1) if flip else Transform(),
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
_MIN_BLOB_AREA = 7

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


_MIN_EXTENT   = 0.15   # minimum contour/bbox fill ratio
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
_GAUSS_SIGMA        = 2.0   # inrange/adaptive "outer" band: ±2.0σ
_GAUSS_CONF_SIGMA   = 2.0   # gaussian back-projection (Method 1) hue window: ±2.0σ

# ── Kalman filter tuning (position tracking) ──────────────────────────────────
_KF_Q_POS    = 2.0    # process noise — position  (px^2/s)
_KF_Q_VEL    = 80.0   # process noise — velocity  (px^2/s^3)
_KF_R        = 30.0   # measurement noise         (px^2)
# ── MIL scale maintenance ────────────────────────────────────────────────────
# cv2.TrackerMIL is a FIXED-SCALE appearance classifier: the box it returns never
# changes size from the one passed to init(). On an approach run the target grows
# by an order of magnitude, so a box seeded at long range ends up mostly
# background — the colour-validation gate then trips on frames where the target
# is plainly visible. Periodically re-detect the blob nearby and re-seed MIL at
# the correct size, which is what CamShift gets for free by re-fitting its window
# to the colour distribution every frame.
_MIL_RESCALE_EVERY = 10    # frames between scale checks while locked
_MIL_RESCALE_TOL   = 0.35  # re-seed when blob-derived area differs by >35%

# TrackerMIL.init() asserts !iposSamples.empty(): it samples positive patches
# from a ring around the box, and that ring clips to nothing when the box is a
# few pixels across or flush against the frame border. Keep every init() call
# above this size and inside this margin.
_MIL_MIN_WIN     = 12   # px, minimum box side handed to TrackerMIL.init()
_MIL_EDGE_MARGIN = 4    # px, keep the box this far off the frame border

_KF_MISS_MAX = 10     # predict this many frames after lock loss, then give up
                      # (~0.5-0.7 s of Kalman coast at 15-20 FPS; was 5 ≈ 0.3 s,
                      # too short to ride out brief CamShift dropouts / re-scales)


_TRACKER_CV_NAME = {
    "mil": "TrackerMIL",
    "kcf": "TrackerKCF",
}


def _make_tracker(name: str):
    """Create an OpenCV appearance tracker by name.

    Supported names:
      - "mil"  — TrackerMIL  (~10–20 ms per update, robust on colour targets)
      - "kcf"  — TrackerKCF  (~1–3 ms per update, ~5–10× faster than MIL,
                              weaker on scale change and non-rigid deformation)

    Handles three API variants across OpenCV versions:
      - free function  cv2.Tracker<Name>_create()  (OpenCV ≤ 4.4 / contrib)
      - legacy ns      cv2.legacy.Tracker<Name>_create()
      - class method   cv2.Tracker<Name>.create()  (OpenCV 4.5+)
    """
    cv_name = _TRACKER_CV_NAME.get(name.lower())
    if cv_name is None:
        raise ValueError(
            f"Unknown tracker '{name}'. Supported: {', '.join(_TRACKER_CV_NAME)}."
        )
    free_fn = f"{cv_name}_create"
    try:
        if hasattr(cv2, free_fn):
            return getattr(cv2, free_fn)()
        legacy = getattr(cv2, "legacy", None)
        if legacy and hasattr(legacy, free_fn):
            return getattr(legacy, free_fn)()
        if hasattr(cv2, cv_name):
            return getattr(cv2, cv_name).create()
        raise RuntimeError(
            f"{cv_name} not available in this OpenCV build (cv2 {cv2.__version__}). "
            f"Try: pip install opencv-contrib-python"
        )
    except cv2.error as exc:
        raise RuntimeError(f"{cv_name} failed to create: {exc}") from exc


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
    """Return a copy of *hist* with bins outside mean ± _GAUSS_CONF_SIGMA·σ zeroed.

    Drives the gaussian back-projection method (Method 1): back-projecting it onto
    an HSV frame produces non-zero values only for pixels whose hue falls within
    the ±_GAUSS_CONF_SIGMA·σ (2σ) confidence window.
    """
    bins = np.arange(180, dtype=np.float32)
    diff = np.abs(bins - mean)
    diff = np.minimum(diff, 180.0 - diff)          # circular wrap
    conf = hist.flatten().copy().astype(np.float32)
    conf[diff >= _GAUSS_CONF_SIGMA * std] = 0.0
    return conf.reshape(hist.shape)


def _apply_outres(frame: np.ndarray, outres) -> np.ndarray:
    """Scale *frame* to (w, h) from --outres.

    Applied BEFORE any crop, so --crop coordinates refer to the SCALED frame.
    Downscaling here is the cheapest way to cut detection/tracking CPU (and heat)
    on the Pi. INTER_AREA is the right filter shrinking; INTER_LINEAR growing."""
    if outres is None:
        return frame
    ow, oh = outres
    fh, fw = frame.shape[:2]
    if (fw, fh) == (ow, oh):
        return frame
    interp = cv2.INTER_AREA if (ow * oh) < (fw * fh) else cv2.INTER_LINEAR
    return cv2.resize(frame, (ow, oh), interpolation=interp)


class Seeker:
    def __init__(
        self,
        source: int | str = 0,
        window_name: str = "Seeker",
        capture_width: int | None = None,
        capture_height: int | None = None,
        crop: tuple[int | None, int | None, int | None, int | None] | None = None,
        outres: tuple[int, int] | None = None,
        histogram_file: str = _CAL_HISTOGRAM_FILE,
        show_histogram: bool = False,
        show_mask: bool = False,
        display: bool = True,
        mask_algo: str = "all",
        use_camshift: bool = True,
        shift_algo: str = "camshift",   # "camshift" | "meanshift"
        box_filter: bool = True,
        use_kalman: bool = True,
        tracker: str = "",
        flip: bool = False,
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
        self.outres          = outres # (w, h) to scale each frame to, or None
        self._flip           = flip
        self.display         = display
        self._show_histogram = show_histogram and display
        self._show_mask      = show_mask and display
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
        self._mil_frames      = 0      # frames tracked by MIL (scale-check cadence)
        self._detect_count    = 0      # consecutive successful detections
        self._res_logged      = False
        # Stage profiling (set by SeekerCtrl when --profile / --debug). track()
        # records the tracker.update sub-cost here each frame; the loop profiler
        # reads it via note(). Zero cost when self.profile is False.
        self.profile      = False
        self.t_detect_ms  = 0.0
        self.t_track_ms   = 0.0
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
            try:
                self.cap = Picamera2Capture(w, h, flip=self._flip)
            except (ImportError, Exception):
                self.cap = cv2.VideoCapture(self.source)
        elif isinstance(self.source, str) and ' ! ' in self.source:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
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
        if self.crop is not None:
            cx, cy, cw, ch = self.crop
            frame_w = (actual_w - cx) if cw is None else cw
            frame_h = (actual_h - cy) if ch is None else ch
        else:
            frame_w, frame_h = actual_w, actual_h
        # Pace playback ONLY for seekable video files so the clip plays at real
        # time. The capture thread otherwise decodes as fast as the CPU allows,
        # which fast-forwards a file source. Live sources MUST NOT be paced —
        # they are already paced by the driver and pacing would add latency /
        # drop fresh frames:
        #   • camera index   → self.source is an int (not a str)
        #   • UDP / GStreamer → self.source is a pipeline string (' ! ')
        # so both fail the isfile() check and keep _cap_interval == 0.
        self._cap_interval = 0.0
        is_pipeline = isinstance(self.source, str) and ' ! ' in self.source
        if isinstance(self.source, str) and not is_pipeline and os.path.isfile(self.source):
            src_fps = self.cap.get(cv2.CAP_PROP_FPS)
            # Some recordings carry a bogus FPS tag (0, NaN, or thousands — e.g. a
            # clip written before the encoder's rate stabilised). Trusting it makes
            # the clip play flat-out. Clamp to a plausible range; fall back to 30.
            if not (1.0 <= src_fps <= 120.0):
                print(f"[Seeker] Video FPS tag implausible ({src_fps:.1f}) — pacing to 30 fps")
                src_fps = 30.0
            self._cap_interval = 1.0 / src_fps
            print(f"[Seeker] Video file source — pacing playback to {src_fps:.2f} fps")
        # Start background capture thread so read_frame() never blocks on I/O.
        self._cap_lock  = threading.Lock()
        self._cap_stop  = False
        self._cap_ok    = False
        self._cap_frame = None
        self._cap_seq   = 0
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()
        disp_w = frame_w
        disp_h = frame_h
        if sys.platform == "win32":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass
        self._hist_window = None
        self._mask_window = None
        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, disp_w, disp_h)
            if self._show_histogram and self._cal_hist is not None:
                self._hist_window = f"{self.window_name} — Histogram"
                cv2.namedWindow(self._hist_window, cv2.WINDOW_NORMAL|cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow(self._hist_window, 360, 230)
            if self._show_mask:
                self._mask_window = f"{self.window_name} — Mask"
                cv2.namedWindow(self._mask_window, cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._mask_window, disp_w, disp_h)

    def _capture_loop(self):
        """Background thread: continuously read frames, always keep the latest.

        For seekable video-file sources (_cap_interval > 0) the reads are
        throttled to the file's native frame interval so the clip plays at real
        time; otherwise it would decode as fast as the CPU allows and the video
        fast-forwards. Live sources keep _cap_interval == 0 and run flat-out."""
        next_t = time.monotonic()
        while not self._cap_stop:
            ok, frame = self.cap.read()
            with self._cap_lock:
                self._cap_ok    = ok
                self._cap_frame = frame
                self._cap_seq  += 1
            if self._cap_interval:
                next_t += self._cap_interval
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    # Decode fell behind real time — reset the schedule so we
                    # don't burst-read to "catch up" and re-introduce fast-forward.
                    next_t = time.monotonic()

    def close(self):
        """Release the capture device and destroy the display window."""
        if getattr(self, "_cap_thread", None):
            self._cap_stop = True
            self._cap_thread.join(timeout=1.0)
            self._cap_thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.display:
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

    def _search_blob_near(self, hsv, win, w_frame, h_frame, locked=False):
        """Detect the target in a padded region around *win*.

        Returns a padded window (x, y, w, h) sized to the blob found there, or
        None. Mirrors the re-acquisition search used during lock-on, so the
        window it returns is directly comparable to an acquisition window.

        When *locked* is True, uses the fast inrange-only detection path
        (skips the acquisition 3-method vote). Callers that are already in a
        locked/tracking context — MIL scale-check and MIL re-acquire — can
        set this to cut the per-call cost 3–5×. The inrange mask is slightly
        looser than the vote, but for "is there still a blob near here?" and
        "what size is it?" that looseness is harmless and often preferable.
        """
        twx, twy, tww, twh = win
        pad = max(tww, twh, 40)
        sx1 = max(0, twx - pad)
        sy1 = max(0, twy - pad)
        sx2 = min(w_frame, twx + tww + pad)
        sy2 = min(h_frame, twy + twh + pad)
        if sx2 - sx1 < 4 or sy2 - sy1 < 4:
            return None
        mask_crop, _ = self._detection_mask(hsv[sy1:sy2, sx1:sx2], locked=locked)
        blob = _nearest_blob_rect(mask_crop, None, self._box_filter)
        if blob is None:
            return None
        bx, by, bw, bh = blob
        bx += sx1
        by += sy1
        pad2 = max(8, int(max(bw, bh) * 0.3))
        ix = max(0, bx - pad2)
        iy = max(0, by - pad2)
        iw = min(w_frame - ix, bw + 2 * pad2)
        ih = min(h_frame - iy, bh + 2 * pad2)
        if iw < 4 or ih < 4:
            return None
        return (ix, iy, iw, ih)

    def _roi_has_target(self, hsv_roi) -> bool:
        """Soft colour-presence test for the MIL window.

        Tests the same hue back-projection CamShift tracks on, not the strict
        3-method detection mask. The detector is a hard binary vote that comes up
        empty on most frames at long range, so using it to validate MIL kept
        tearing down a perfectly good lock while CamShift sailed through on the
        graded signal.
        """
        if hsv_roi.size == 0:
            return False
        if self._roi_hist is None:
            mask, _ = self._detection_mask(hsv_roi)
            return cv2.countNonZero(mask) >= _MIN_BLOB_AREA
        bp = cv2.calcBackProject([hsv_roi], [0], self._roi_hist, [0, 180], 1)
        return cv2.countNonZero(bp) >= _MIN_BLOB_AREA

    def _mil_safe_win(self, win, w_frame: int, h_frame: int):
        """Clamp *win* to something TrackerMIL.init() can actually sample.

        cv2.TrackerMIL.init() asserts `!iposSamples.empty()`: it draws positive
        patches from a ring around the box, and that set comes out empty when the
        box is a few pixels across or sits flush against the frame border — the
        sampling rectangle is then clipped to nothing. The assertion surfaces as
        cv2.error, which is NOT a RuntimeError/ValueError, so it escapes the
        usual tracker guards and takes the process down.

        Returns a usable (x, y, w, h), or None when the frame is too small.
        """
        m = _MIL_EDGE_MARGIN
        if (w_frame - 2 * m) < _MIL_MIN_WIN or (h_frame - 2 * m) < _MIL_MIN_WIN:
            return None
        x, y, w, h = (int(v) for v in win)
        w = min(max(w, _MIL_MIN_WIN), w_frame - 2 * m)
        h = min(max(h, _MIL_MIN_WIN), h_frame - 2 * m)
        x = max(m, min(x, w_frame - m - w))
        y = max(m, min(y, h_frame - m - h))
        return (x, y, w, h)

    def _reinit_mil(self, frame, win) -> bool:
        """Re-seed the MIL tracker at *win*. Returns True on success."""
        h_frame, w_frame = frame.shape[:2]
        safe = self._mil_safe_win(win, w_frame, h_frame)
        if safe is None:
            return False
        try:
            obj = _make_tracker(self._tracker_name)
            obj.init(frame, safe)
        except (RuntimeError, ValueError, cv2.error) as exc:
            print(f"[Seeker] MIL re-init failed on {safe}: {exc}")
            return False
        self._tracker_obj = obj
        self._track_win   = safe
        return True

    def _timed(self, attr, fn, *args):
        """Call fn(*args), adding its elapsed ms to self.<attr> when profiling.
        Returns fn's result. No timing overhead when self.profile is False."""
        if not self.profile:
            return fn(*args)
        _t = time.perf_counter()
        r = fn(*args)
        setattr(self, attr, getattr(self, attr) + (time.perf_counter() - _t) * 1000.0)
        return r

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
        if self.profile:                       # reset per-frame stage counters
            self.t_detect_ms = 0.0
            self.t_track_ms  = 0.0
        # Reuse persistent buffers; (re)allocate only when frame shape changes.
        if self._hsv_buf is None or self._hsv_buf.shape[:2] != (h_frame, w_frame):
            self._hsv_buf = np.empty((h_frame, w_frame, 3), dtype=np.uint8)
            self._out_buf = np.empty_like(frame)
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self._hsv_buf)
        hsv = self._hsv_buf
        np.copyto(self._out_buf, frame)
        out = self._out_buf

        if not self._use_camshift and not self._tracker_name:
            # ── Detection-only path (no tracker at all) ───────────────────────
            # NOTE: this must NOT swallow the MIL case. '--tracker mil' parses to
            # use_camshift=False (mil and camshift are mutually exclusive), so
            # testing _use_camshift alone returned here every frame and the MIL
            # step further down was unreachable — no tracker, no Kalman, no
            # coasting, just independent per-frame detection.
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
            centred    = abs(ex) < _CENTER_THRESHOLD and abs(ey-self.pitch_offset_norm) < _CENTER_THRESHOLD
            box_colour = (180, 105, 255) if centred else (0, 255, 255)
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
                    # Route through _reinit_mil so this path gets the same size /
                    # border clamping and the same cv2.error guard; a tiny or
                    # edge-hugging acquisition box would otherwise trip the
                    # !iposSamples.empty() assertion and kill the process.
                    if not self._reinit_mil(frame, (ix, iy, iw, ih)):
                        self._detect_count = 2   # not locked yet — try again next frame
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
            ok, bbox = self._timed("t_track_ms", self._tracker_obj.update, frame)
            if ok:
                x, y, w, h = (int(v) for v in bbox)
                if w >= 4 and h >= 4:
                    # ── Color validation: confirm ROI still contains target ─────
                    x  = max(0, min(x, w_frame - 1))
                    y  = max(0, min(y, h_frame - 1))
                    w  = min(w, w_frame - x)
                    h  = min(h, h_frame - y)
                    if not self._roi_has_target(hsv[y:y + h, x:x + w]):
                        # MIL's window no longer holds target colour. Because MIL
                        # is fixed-scale, the usual cause is a target that changed
                        # size and slid out of a stale box while still plainly
                        # visible — so try a local re-detect and re-seed MIL at the
                        # right size before falling back to blind coasting.
                        rewin = self._search_blob_near(
                            hsv, self._track_win or (x, y, w, h),
                            w_frame, h_frame, locked=True)
                        if rewin is not None and self._reinit_mil(frame, rewin):
                            x, y, w, h = rewin
                        else:
                            # Genuinely nothing nearby: coast on the last good window
                            # for up to _KF_MISS_MAX frames instead of blinking the
                            # box off; give up only past the grace window.
                            self._miss_count += 1
                            if self._miss_count < _KF_MISS_MAX and self._track_win is not None:
                                lx, ly, lw, lh = self._track_win
                                cx = max(0, min(lx + lw // 2, w_frame - 1))
                                cy = max(0, min(ly + lh // 2, h_frame - 1))
                                cv2.rectangle(out, (lx, ly), (lx + lw, ly + lh), (0, 165, 255), 2)  # amber = coasting
                                cv2.line(out, (0, cy), (w_frame, cy), (0, 165, 255), 1)
                                cv2.line(out, (cx, 0), (cx, h_frame), (0, 165, 255), 1)
                                cv2.circle(out, (cx, cy), 3, (0, 165, 255), -1)
                                self._update_histogram_window()
                                return out, cx, cy
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
                    # ── Scale maintenance ───────────────────────────────────────
                    # MIL never resizes its own box, so periodically compare it to
                    # the blob actually present and re-seed when they diverge.
                    self._mil_frames += 1
                    if self._mil_frames % _MIL_RESCALE_EVERY == 0:
                        rewin = self._search_blob_near(hsv, (x, y, w, h),
                                                       w_frame, h_frame, locked=True)
                        if rewin is not None:
                            new_area = rewin[2] * rewin[3]
                            cur_area = max(w * h, 1)
                            if abs(new_area - cur_area) > _MIL_RESCALE_TOL * cur_area:
                                if self._reinit_mil(frame, rewin):
                                    x, y, w, h = rewin
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
                    centred    = abs(ex) < _CENTER_THRESHOLD and abs(ey-self.pitch_offset_norm) < _CENTER_THRESHOLD
                    box_colour = (180, 105, 255) if centred else (0, 255, 255)
                    cv2.rectangle(out, (x, y), (x + w, y + h), box_colour, 2)
                    cv2.line(out, (0, cy), (w_frame, cy), (0, 233, 233), 1)
                    cv2.line(out, (cx, 0), (cx, h_frame), (0, 233, 233), 1)
                    cv2.circle(out, (cx, cy), 3, (0, 233, 233), -1)
                    self._update_histogram_window()
                    return out, cx, cy
            # Tracker update failed or bbox too small. Rather than hard-resetting on
            # a single MIL hiccup — which forces a 3-frame color re-acquire and blinks
            # the box — coast: hold the last known window for up to _KF_MISS_MAX frames,
            # retrying MIL each frame (it usually re-locks within a frame or two). Only
            # after _KF_MISS_MAX consecutive misses do we give up. Shares _miss_count
            # with the colour-validation gate, so any good frame clears it.
            self._miss_count += 1
            if self._miss_count < _KF_MISS_MAX and self._track_win is not None:
                x, y, w, h = self._track_win
                cx = max(0, min(x + w // 2, w_frame - 1))
                cy = max(0, min(y + h // 2, h_frame - 1))
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 2)  # amber = coasting
                cv2.line(out, (0, cy), (w_frame, cy), (0, 165, 255), 1)
                cv2.line(out, (cx, 0), (cx, h_frame), (0, 165, 255), 1)
                cv2.circle(out, (cx, cy), 3, (0, 165, 255), -1)
                self._update_histogram_window()
                return out, cx, cy
            # Exceeded the grace window (or no last window) — give up and hard reset.
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

        # ── Search-window crop, computed BEFORE back-projection ──────────────
        # Pre-translate the search window by Kalman-predicted velocity so CamShift
        # starts near where the target is expected to be this frame.
        twx, twy, tww, twh = self._track_win
        if self._use_kalman and self._kf_initialized and kf_dt > 0:
            dx = int(round(self._kf_x1 * kf_dt))
            dy = int(round(self._kf_y1 * kf_dt))
            twx = max(0, min(twx + dx, w_frame - tww))
            twy = max(0, min(twy + dy, h_frame - twh))
            self._track_win = (twx, twy, tww, twh)
        # Padded search window — same rule as before, but now used to crop the
        # HSV input to calcBackProject instead of zeroing pixels afterwards.
        # Only cropped-region pixels are actually computed by calcBackProject /
        # S-V mask / blur / dilate, so the per-frame track cost scales with
        # (search-window area), not (frame area). Typical win: 5–20× fewer
        # per-pixel ops when the target window is small relative to the frame.
        pad_x = max(tww // 2, 20);  pad_y = max(twh // 2, 20)
        bx1 = max(0, twx - pad_x);  by1 = max(0, twy - pad_y)
        bx2 = min(w_frame, twx + tww + pad_x)
        by2 = min(h_frame, twy + twh + pad_y)
        hsv_crop = hsv[by1:by2, bx1:bx2]

        # ── Back-projection on the CROP ──────────────────────────────────────
        back_proj = cv2.calcBackProject([hsv_crop], [0], self._roi_hist, [0, 180], 1)

        # ── S/V gate ─────────────────────────────────────────────────────────
        # The back-projection is hue-only, so it lights up *any* same-hue pixel
        # regardless of saturation/value — including shadowed/washed-out ground
        # that pulls the CamShift centroid off-target. AND-gate with the same
        # acquisition 'outer' band (hue±2σ with S,V ≥ 40 floors) that detection
        # uses, so only saturated/bright same-hue pixels survive.
        # Keep an UN-gated copy for the loss/density check below — the gate
        # removes dark/washed pixels to sharpen the centroid, but those must
        # NOT be read as "target gone" or the gate would *increase* track loss.
        presence = None
        if self._precomp_bands is not None:
            presence = back_proj.copy()
            sv_mask = self._apply_inrange_band(hsv_crop, "outer")
            cv2.bitwise_and(back_proj, sv_mask, dst=back_proj)
        cv2.GaussianBlur(back_proj, (3, 3), 0, dst=back_proj)
        cv2.dilate(back_proj, back_proj, self._kern5)
        # Process the un-gated presence map identically so the density threshold
        # stays calibrated to the original (pre-gate) signal.
        if presence is not None:
            cv2.GaussianBlur(presence, (3, 3), 0, dst=presence)
            cv2.dilate(presence, presence, self._kern5)

        # ── CamShift / MeanShift in CROP coordinates ─────────────────────────
        # Translate the search window from frame-absolute to crop-local, run
        # the tracker on the cropped probability image, then translate back.
        # CamShift only needs the probability array to cover its search
        # window — it doesn't care whether that array is full-frame or a crop.
        crop_win = (twx - bx1, twy - by1, tww, twh)
        if self._shift_algo == "meanshift":
            _, crop_win = cv2.meanShift(back_proj, crop_win, self._term_crit)
            ret = None
        else:
            ret, crop_win = cv2.CamShift(back_proj, crop_win, self._term_crit)
        # CamShift returns the fitted ellipse `ret` in crop-local coords; shift
        # its centre back to frame coords so downstream code (line snap /
        # rectangle draw) doesn't need to know about the crop.
        if ret is not None:
            ((cs_cx_local, cs_cy_local), size, angle) = ret
            ret = ((cs_cx_local + bx1, cs_cy_local + by1), size, angle)
        self._track_win = (crop_win[0] + bx1, crop_win[1] + by1,
                           crop_win[2],       crop_win[3])

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
            # Use the UN-gated presence map (falls back to back_proj when the
            # gate is inactive) so the S/V gate can't cause false losses.
            # dens_src is now crop-sized; translate the (frame-absolute) window
            # into crop-local coords, then clip to the crop bounds so a window
            # that partially escaped the padded search region doesn't read out
            # of bounds and return a spurious zero-density (which would trip a
            # false loss).
            wx, wy, ww, wh = self._track_win
            dens_src = presence if presence is not None else back_proj
            cx1 = max(0, wx - bx1);       cy1 = max(0, wy - by1)
            cx2 = min(bx2 - bx1, wx + ww - bx1)
            cy2 = min(by2 - by1, wy + wh - by1)
            if cx2 > cx1 and cy2 > cy1:
                roi_bp  = dens_src[cy1:cy2, cx1:cx2]
                density = float(roi_bp.mean()) / 255.0
                if density < 0.05:
                    camshift_bad = True
            else:
                # window is entirely outside the crop we computed — nothing to
                # measure. Fail closed so we drop lock rather than hold on to
                # a stale one.
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
        centred    = abs(ex) < _CENTER_THRESHOLD and abs(ey-self.pitch_offset_norm) < _CENTER_THRESHOLD
        box_colour = (180, 105, 255) if centred else (0, 255, 255)
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
        return float(errorx), float(errory - self.pitch_offset_norm)

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
        if ok:
            frame = _apply_outres(frame, self.outres)   # scale BEFORE crop
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
                  + (f"  (scaled to {self.outres[0]}x{self.outres[1]})" if self.outres else "")
                  + (f"  (cropped from raw, offset {self.crop[:2]})" if self.crop else ""))
            self._res_logged = True
        return seq, ok, frame

    def run(self):
        """Open source, run CamShift tracking loop; press 'q' to quit,
        'r' to reset the tracker."""
        self.open()
        prev_time = time.time()
        frame_times = collections.deque(maxlen=30)
        started = False
        try:
            while True:
                curr_time = time.time()
                frame_times.append(curr_time - prev_time)
                prev_time = curr_time
                fps = 1.0 / (sum(frame_times) / len(frame_times))

                _seq, ok, frame = self.read_frame()
                if frame is None:
                    # None before any frame = capture thread warming up; None
                    # *after* frames have flowed = end of a video file → stop
                    # (otherwise the loop spins forever on a finished file).
                    if started:
                        print("[Seeker] End of stream.")
                        break
                    continue
                if not ok:
                    print("[Seeker] End of stream.")
                    break
                started = True

                annotated, cx, cy = self.track(frame)
                errorx, errory    = self.error_xy(cx, cy, frame.shape)

                if errorx is not None:
                    label = f"ex={errorx:+.3f}  ey={errory:+.3f}"
                    cv2.putText(annotated, label, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 233, 0), 2)

                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 233, 0), 2)

                if self.display:
                    cv2.imshow(self.window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = 0xFF
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
