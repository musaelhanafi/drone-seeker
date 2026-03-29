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
  Q/Esc  — quit
"""

import argparse
import sys

import cv2
import numpy as np

# Import shared detection helpers from seeker so behavior is identical
from seeker import _fit_gaussian, _load_histogram, _pink_mask, _nearest_blob_rect, _confidence_hist

WINDOW      = "Calibrate Color"
HIST_WINDOW = "Histogram"
MASK_WINDOW = "Mask"
OUT_FILE    = "color_histogram.txt"



def _conf_mask(hsv: np.ndarray, conf_hist: np.ndarray) -> np.ndarray:
    """Back-project the confidence histogram onto an HSV frame.
    Accepts pixels whose hue bin has non-zero weight (i.e. within 3*std of mean)
    and also pass saturation/value floors.
    Dilate is applied before open to fill gaps left by sparse histogram bins."""
    bp  = cv2.calcBackProject([hsv], [0], conf_hist, [0, 180], 1)
    in_sat = hsv[:, :, 1] > 40
    in_val = hsv[:, :, 2] > 40
    mask = ((bp > 0) & in_sat & in_val).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
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
        conf_hist  = _confidence_hist(saved_hist, gauss_mean, gauss_std)
        kept = int((conf_hist.flatten() > 0).sum())
        print(f"[Cal] Confidence hist: mean={gauss_mean:.1f}  std={gauss_std:.1f}  bins={kept}/180")
    else:
        gauss_mean = gauss_std = conf_hist = None
        print("[Cal] No histogram file found — using hardcoded pink ranges.")

    hist  = None   # histogram of current detection
    saved = saved_hist is not None

    print("[Cal] Point camera at target.")
    print("[Cal]  S — save   R — reset   Q/Esc — quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Cal] End of stream.")
            break

        if crop is not None:
            cx, cy, cw, ch = crop
            fh, fw = frame.shape[:2]
            if cw is None:
                cw = fw - cx
            if ch is None:
                ch = fh - cy
            frame = frame[cy:cy + ch, cx:cx + cw]

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if conf_hist is not None:
            mask = _conf_mask(hsv, conf_hist)
        else:
            mask = _pink_mask(hsv)
        rect    = _nearest_blob_rect(mask, frame.shape)
        display = frame.copy()
        if args.mask:
            cv2.imshow(MASK_WINDOW, mask)

        if rect is not None:
            x, y, w, h = rect
            hist = _compute_histogram(frame, rect, mask)
            # bounding box: green if histogram already saved, yellow otherwise
            colour = (0, 255, 0) if saved else (0, 255, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(display, f"{w}x{h}px", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
            cv2.imshow(HIST_WINDOW, _draw_histogram(hist))
        else:
            hist = None

        # detection mode label
        mode_label = (f"Confidence 3*delta  mean={gauss_mean:.0f}  std={gauss_std:.1f}"
                      if conf_hist is not None else "Fallback: pink HSV ranges")

        # status line
        if saved:
            status = f"Saved: {args.output}"
            s_col  = (0, 255, 0)
        elif rect is not None:
            status = "Press S to save"
            s_col  = (0, 255, 255)
        else:
            status = "No target detected"
            s_col  = (0, 0, 255)

        cv2.putText(display, mode_label,
                    (10, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        cv2.putText(display, f"S=save  R=reset  Q=quit   {status}",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, s_col, 1)

        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            if hist is not None:
                np.savetxt(args.output, hist.flatten(), fmt="%.4f")
                print(f"[Cal] Histogram saved → {args.output}  ({len(hist)} bins)")
                saved = True
                gauss_mean, gauss_std = _fit_gaussian(hist)
                conf_hist = _confidence_hist(hist, gauss_mean, gauss_std)
                kept = int((conf_hist.flatten() > 0).sum())
                print(f"[Cal] Detection updated: mean={gauss_mean:.1f}  std={gauss_std:.1f}  bins={kept}/180")
            else:
                print("[Cal] No target detected — nothing to save.")
        elif key == ord('r'):
            hist       = None
            saved      = False
            gauss_mean = gauss_std = conf_hist = None
            print("[Cal] Reset — reverted to pink HSV ranges.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
