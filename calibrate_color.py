#!/usr/bin/env python3
"""
calibrate_color.py — Auto-detect pink, show bounding box, save hue histogram.

Detection strategy:
  - If a saved histogram file exists on startup (or after pressing S), a
    Gaussian is fitted to it and detection uses |hue - mean| < 2.5*std.
  - Otherwise falls back to hardcoded pink HSV threshold ranges.

Controls:
  S      — save histogram of current detection
  R      — reset / clear saved histogram
  D      — draw / select region manually (overrides auto-detection)
  C      — clear manual region selection
  Q/Esc  — quit
"""

import argparse
import sys
import threading

import cv2
import numpy as np

# Import shared detection helpers from seeker so behavior is identical
from seeker import (
    _fit_gaussian, _load_histogram, _pink_mask, _nearest_blob_rect,
    _confidence_hist, _GAUSS_SIGMA,
)

WINDOW      = "Calibrate Color"
HIST_WINDOW = "Histogram"
MASK_WINDOW = "Mask"
OUT_FILE    = "color_histogram.txt"

# Module-level precomputed constants
_KERN5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_OUTER_LUT = np.zeros(256, dtype=np.uint8)
_OUTER_LUT[255] = 128



def _build_inrange_bounds(gauss_mean: float, gauss_std: float) -> dict:
    """Precompute cv2.inRange bounds for core (±1σ) and outer (±Nσ) bands."""
    result = {}
    for key, (mean, hw) in [("core",  (gauss_mean, gauss_std)),
                             ("outer", (gauss_mean, _GAUSS_SIGMA * gauss_std))]:
        lo, hi = mean - hw, mean + hw
        if lo < 0:
            result[key] = (
                np.array([max(0, int(lo + 180)), 40, 40],    dtype=np.uint8),
                np.array([179,                   255, 255],   dtype=np.uint8),
                np.array([0,                     40,  40],    dtype=np.uint8),
                np.array([min(179, int(hi)),      255, 255],  dtype=np.uint8),
                "wrap_lo",
            )
        elif hi > 179:
            result[key] = (
                np.array([max(0, int(lo)),          40,  40], dtype=np.uint8),
                np.array([179,                      255, 255], dtype=np.uint8),
                np.array([0,                        40,  40], dtype=np.uint8),
                np.array([min(179, int(hi - 180)),  255, 255], dtype=np.uint8),
                "wrap_hi",
            )
        else:
            result[key] = (
                np.array([max(0, int(lo)),   40,  40],  dtype=np.uint8),
                np.array([min(179, int(hi)), 255, 255], dtype=np.uint8),
                None, None, "single",
            )
    return result


def _build_hue_gate_lut(gauss_mean: float, gauss_std: float) -> np.ndarray:
    """256-element LUT: hue → 255 if within ±Nσ of mean, else 0."""
    lut  = np.zeros(256, dtype=np.uint8)
    bins = np.arange(180, dtype=np.float32)
    d    = np.abs(bins - gauss_mean)
    d    = np.minimum(d, 180.0 - d)
    lut[:180] = (d < _GAUSS_SIGMA * gauss_std).astype(np.uint8) * 255
    return lut


def _apply_inrange_band(hsv: np.ndarray, bands: dict, key: str) -> np.ndarray:
    lo_a, hi_a, lo_b, hi_b, mode = bands[key]
    if mode == "single":
        return cv2.inRange(hsv, lo_a, hi_a)
    return cv2.bitwise_or(cv2.inRange(hsv, lo_a, hi_a),
                          cv2.inRange(hsv, lo_b, hi_b))


def _mask_gaussian(hsv: np.ndarray, h_blur: np.ndarray,
                   conf_hist: np.ndarray, bands: dict) -> np.ndarray:
    """Method 1 — Gaussian back-projection with S/V gate."""
    h_save = hsv[:, :, 0].copy()   # save H channel only (1 plane, not full frame)
    hsv[:, :, 0] = h_blur
    bp = cv2.calcBackProject([hsv], [0], conf_hist, [0, 180], 1)
    hsv[:, :, 0] = h_save
    sv_ok = _apply_inrange_band(hsv, bands, "outer")
    _, bp_bin = cv2.threshold(bp, 0, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(bp_bin, sv_ok)


def _mask_adaptive(hsv: np.ndarray, h_blur: np.ndarray,
                   hue_gate_lut: np.ndarray) -> np.ndarray:
    """Method 2 — Adaptive hue threshold + precomputed σ-gate LUT."""
    # h_blur is uint8 0-179 — skip per-frame normalize; blockSize 11 (down from 21).
    adapt    = cv2.adaptiveThreshold(
        h_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=3,
    )
    hue_gate = cv2.LUT(h_blur, hue_gate_lut)
    return cv2.bitwise_and(adapt, hue_gate)


def _mask_inrange(hsv: np.ndarray, bands: dict) -> np.ndarray:
    """Method 3 — Soft-weighted two-band inRange (core=255, outer-only=128)."""
    core  = _apply_inrange_band(hsv, bands, "core")
    outer = _apply_inrange_band(hsv, bands, "outer")
    cv2.subtract(outer, core, dst=outer)
    outer = cv2.LUT(outer, _OUTER_LUT)   # 255→128, 0→0
    return cv2.bitwise_or(core, outer)


def _detection_mask(hsv: np.ndarray, conf_hist, bands, hue_gate_lut,
                    algo: str = "all") -> np.ndarray:
    """Build detection mask using the selected algorithm.

    algo: "gaussian" | "adaptive" | "inrange" | "all" (2-of-3 majority vote)
    Falls back to hardcoded pink ranges when conf_hist is None.
    bands and hue_gate_lut are precomputed from gauss_mean/std; pass None when
    conf_hist is None.
    """
    if conf_hist is not None:
        h_blur = cv2.GaussianBlur(hsv[:, :, 0], (5, 5), 0)
        if algo == "gaussian":
            mask = _mask_gaussian(hsv, h_blur, conf_hist, bands)
        elif algo == "adaptive":
            mask = _mask_adaptive(hsv, h_blur, hue_gate_lut)
        elif algo == "inrange":
            mask = _mask_inrange(hsv, bands)
        else:  # "all" — 2-of-3 majority vote
            m1    = _mask_gaussian(hsv, h_blur, conf_hist, bands)
            m2    = _mask_adaptive(hsv, h_blur, hue_gate_lut)
            m3    = _mask_inrange(hsv, bands)
            votes = (m1 > 0).view(np.uint8)
            votes = votes + (m2 > 0).view(np.uint8)
            votes = votes + (m3 > 0).view(np.uint8)
            _, mask = cv2.threshold(votes, 1, 255, cv2.THRESH_BINARY)
    else:
        mask = _pink_mask(hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   _KERN5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, _KERN5)
    return mask


# ── Histogram helpers ─────────────────────────────────────────────────────────

def _compute_histogram(frame: np.ndarray, rect, mask: np.ndarray) -> np.ndarray:
    """Compute and normalise the hue histogram of the blob ROI."""
    x, y, w, h  = rect
    roi_bgr     = frame[y:y + h, x:x + w]
    roi_mask    = mask[y:y + h, x:x + w]
    hsv_roi     = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist        = cv2.calcHist([hsv_roi], [0], roi_mask, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist



def _draw_histogram(hist, width=360, height=200) -> np.ndarray:
    """Render histogram as a BGR image with hue-coloured bars and axis labels."""
    canvas  = np.zeros((height + 30, width, 3), dtype=np.uint8)
    max_v   = float(hist.max()) or 1.0
    bar_w   = width / 180.0
    for i, v in enumerate(hist.flatten()):
        bar_h  = int(v / max_v * height)
        colour = cv2.cvtColor(
            np.uint8([[[int(i), 220, 200]]]), cv2.COLOR_HSV2BGR
        )[0][0].tolist()
        x0 = int(i * bar_w)
        x1 = max(x0 + 1, int((i + 1) * bar_w))
        cv2.rectangle(canvas, (x0, height - bar_h), (x1, height), colour, -1)
    for hue in range(0, 181, 30):
        x = int(hue * bar_w)
        cv2.line(canvas, (x, height), (x, height + 5), (200, 200, 200), 1)
        cv2.putText(canvas, str(hue), (max(0, x - 8), height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, "Hue (0-179)", (width // 2 - 35, height + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pink colour calibration tool")
    ap.add_argument("--source",  default="0",
                    help="Camera index or video file (default: 0)")
    ap.add_argument("--res",     type=int, nargs=2, default=None,
                    metavar=("W", "H"),
                    help="Camera capture resolution (e.g. --res 1280 720)")
    ap.add_argument("--output",  default=OUT_FILE,
                    help=f"Output histogram file (default: {OUT_FILE})")
    ap.add_argument("--mask", action="store_true", default=False,
                    help="Show detection mask in a separate window (default: disabled)")
    ap.add_argument("--mask-algo", default="all",
                    choices=["gaussian", "adaptive", "inrange", "all"],
                    help="Detection algorithm: gaussian, adaptive, inrange, or all (2-of-3 vote, default)")
    ap.add_argument("--crop", type=str, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"),
                    help="Crop each frame to this ROI (e.g. --crop 320 180 640 360). "
                         "Use - for W or H to mean 'rest of dimension after offset'.")
    args = ap.parse_args()

    crop = (tuple(None if v == '-' else int(v) for v in args.crop)
            if args.crop else None)

    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source {source!r}", file=sys.stderr)
        sys.exit(1)

    if args.res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.res[1])

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Cal] Camera: {actual_w}x{actual_h}"
          + (f"  crop={crop}" if crop else ""))

    cv2.namedWindow(WINDOW,      cv2.WINDOW_NORMAL)
    cv2.namedWindow(HIST_WINDOW, cv2.WINDOW_NORMAL)
    if args.mask:
        cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(HIST_WINDOW, 360, 230)

    # Load existing histogram so detection immediately uses it
    saved_hist = _load_histogram(args.output)
    if saved_hist is not None:
        gauss_mean, gauss_std = _fit_gaussian(saved_hist)
        conf_hist    = _confidence_hist(saved_hist, gauss_mean, gauss_std)
        bands        = _build_inrange_bounds(gauss_mean, gauss_std)
        hue_gate_lut = _build_hue_gate_lut(gauss_mean, gauss_std)
        kept = int((conf_hist.flatten() > 0).sum())
        print(f"[Cal] Confidence hist: mean={gauss_mean:.1f}  std={gauss_std:.1f}  bins={kept}/180")
    else:
        gauss_mean = gauss_std = conf_hist = bands = hue_gate_lut = None
        print("[Cal] No histogram file found — using hardcoded pink ranges.")

    hist          = None   # histogram of current detection
    saved         = saved_hist is not None
    manual_rect   = None   # user-drawn ROI (overrides auto-detected blob when set)
    detect_count  = 0      # consecutive frames blob was detected (lock after 3)
    miss_count    = 0      # consecutive frames blob was absent (drop after MISS_THRESH)
    last_center   = None   # (cx, cy) of last confirmed blob — biases blob selection
    MISS_THRESH   = 5

    # ── Background capture thread ─────────────────────────────────────────────
    _cap_lock  = threading.Lock()
    _cap_buf   = [False, None]   # [ok, frame]
    _cap_stop  = [False]

    def _capture_loop():
        while not _cap_stop[0]:
            ok, frm = cap.read()
            with _cap_lock:
                _cap_buf[0], _cap_buf[1] = ok, frm

    _cap_thread = threading.Thread(target=_capture_loop, daemon=True)
    _cap_thread.start()

    # Pre-allocated per-frame buffers (lazily sized on first frame)
    _hsv_buf = None
    _disp_buf = None

    print("[Cal] Point camera at target.")
    print("[Cal]  S — save   R — reset   D — draw region   C — clear region   Q/Esc — quit")

    while True:
        with _cap_lock:
            ok, frame = _cap_buf[0], _cap_buf[1]
        if not ok or frame is None:
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            continue

        if crop is not None:
            cx, cy, cw, ch = crop
            fh, fw = frame.shape[:2]
            if cw is None:
                cw = fw - cx
            if ch is None:
                ch = fh - cy
            frame = frame[cy:cy + ch, cx:cx + cw]

        fh, fw = frame.shape[:2]
        if _hsv_buf is None or _hsv_buf.shape[:2] != (fh, fw):
            _hsv_buf  = np.empty((fh, fw, 3), dtype=np.uint8)
            _disp_buf = np.empty_like(frame)
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=_hsv_buf)
        hsv = _hsv_buf
        np.copyto(_disp_buf, frame)
        display = _disp_buf

        mask = _detection_mask(hsv, conf_hist, bands, hue_gate_lut, args.mask_algo)
        rect = _nearest_blob_rect(mask, frame.shape, prefer_pt=last_center)
        if args.mask:
            cv2.imshow(MASK_WINDOW, mask)

        # Update confirmation / miss counters (manual ROI bypasses both).
        if manual_rect is None:
            if rect is not None:
                detect_count = min(detect_count + 1, 3)
                miss_count   = 0
                x, y, w, h  = rect
                last_center  = (x + w // 2, y + h // 2)
            else:
                miss_count += 1
                if miss_count >= MISS_THRESH:
                    detect_count = 0
                    last_center  = None
                    miss_count   = 0
        confirmed = (manual_rect is not None) or (detect_count >= 3)

        active_rect = manual_rect if manual_rect is not None else (rect if confirmed else None)
        if active_rect is not None:
            x, y, w, h = active_rect
            hist = _compute_histogram(frame, active_rect, mask)
            # bounding box: cyan for manual, green if saved, yellow otherwise
            if manual_rect is not None:
                colour = (255, 255, 0)
            else:
                colour = (0, 255, 0) if saved else (0, 255, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)
            label = f"{w}x{h}px" + (" [manual]" if manual_rect is not None else "")
            cv2.putText(display, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
            cv2.imshow(HIST_WINDOW, _draw_histogram(hist))
        else:
            hist = None

        # detection mode label
        if conf_hist is not None:
            mode_label = (f"algo={args.mask_algo}  mean={gauss_mean:.0f}"
                          f"  std={gauss_std:.1f}  3σ={_GAUSS_SIGMA*gauss_std:.1f}")
        else:
            mode_label = "Fallback: pink HSV ranges"
        if manual_rect is not None:
            mode_label += "  [manual ROI]"

        # status line
        if saved:
            status = f"Saved: {args.output}"
            s_col  = (0, 255, 0)
        elif active_rect is not None:
            status = "Press S to save"
            s_col  = (0, 255, 255)
        elif rect is not None:
            status = f"Confirming... {detect_count}/3"
            s_col  = (0, 165, 255)
        else:
            status = "No target detected"
            s_col  = (0, 0, 255)

        cv2.putText(display, mode_label,
                    (10, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        cv2.putText(display, f"S=save  R=reset  D=draw  C=clear  Q=quit   {status}",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, s_col, 1)

        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            roi = cv2.selectROI(WINDOW, display, fromCenter=False, showCrosshair=True)
            if roi[2] > 0 and roi[3] > 0:
                manual_rect = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
                print(f"[Cal] Manual ROI set: {manual_rect}")
            else:
                print("[Cal] ROI selection cancelled.")
        elif key == ord('c'):
            manual_rect = None
            print("[Cal] Manual ROI cleared — back to auto-detection.")
        elif key == ord('s'):
            if hist is not None:
                np.savetxt(args.output, hist.flatten(), fmt="%.4f")
                print(f"[Cal] Histogram saved → {args.output}  ({len(hist)} bins)")
                saved        = True
                gauss_mean, gauss_std = _fit_gaussian(hist)
                conf_hist    = _confidence_hist(hist, gauss_mean, gauss_std)
                bands        = _build_inrange_bounds(gauss_mean, gauss_std)
                hue_gate_lut = _build_hue_gate_lut(gauss_mean, gauss_std)
                kept = int((conf_hist.flatten() > 0).sum())
                print(f"[Cal] Detection updated: mean={gauss_mean:.1f}  std={gauss_std:.1f}  bins={kept}/180")
            else:
                print("[Cal] No target detected — nothing to save.")
        elif key == ord('r'):
            hist         = None
            saved        = False
            manual_rect  = None
            detect_count = 0
            miss_count   = 0
            last_center  = None
            gauss_mean   = gauss_std = conf_hist = bands = hue_gate_lut = None
            print("[Cal] Reset — reverted to pink HSV ranges.")

    _cap_stop[0] = True
    _cap_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
