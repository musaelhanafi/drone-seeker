#!/usr/bin/env python3
"""
app_calibrate.py — Pause-and-select colour calibration from a video source.

Plays a video (or camera) source. Pause on a good frame, drag a box over the
target colour, and the hue histogram of that region is saved to the calibration
file the Seeker loads (color_histogram.txt). Resume to scrub for a better frame.

Workflow:
    P  — pause playback (freezes the current frame, then opens region select)
    S  — (while paused) re-open region selection on the frozen frame
    R  — resume playback
    Z  — clear / discard the current selection (revert frozen frame)
    Q / Esc — quit

Region selection uses OpenCV's drag-box selector: drag a rectangle, then press
ENTER/SPACE to confirm or C to cancel. On confirm, the hue histogram is computed
and written to --output immediately.

Usage:
    python3 script/app_calibrate.py --source video.mp4
    python3 script/app_calibrate.py --source 0
    python3 script/app_calibrate.py --source clip.mp4 --output my_hist.txt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Import the project's calibration helpers (one level above script/) so the
# saved histogram is byte-identical to what calibrate_color.py / Seeker expect.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seeker import _fit_gaussian, _load_histogram, _GAUSS_SIGMA  # noqa: E402
from calibrate_color import _draw_histogram                      # noqa: E402

WINDOW      = "Calibrate (P=pause/select  R=resume  Z=clear  Q=quit)"
HIST_WINDOW = "Histogram"

# Saturation / Value floor — drop near-black and washed-out pixels so shadows
# and highlights inside the box don't pollute the hue histogram. Matches the
# S/V floor (40) used by calibrate_color.py's inRange bands.
_SV_FLOOR = 40


def _parse_source(value: str) -> int | str:
    """Camera index (int) or file path / pipeline (str)."""
    try:
        return int(value)
    except ValueError:
        return value


def _compute_histogram(frame_bgr: np.ndarray, rect) -> np.ndarray:
    """Normalised 180-bin hue histogram of the selected ROI.

    Gates out low-S / low-V pixels so the histogram captures the target hue,
    not shadow/highlight noise. Output matches Seeker's expected format
    (savetxt of the flattened 180-bin float histogram)."""
    x, y, w, h = rect
    roi = frame_bgr[y:y + h, x:x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gate = cv2.inRange(
        hsv,
        np.array([0,   _SV_FLOOR, _SV_FLOOR], dtype=np.uint8),
        np.array([179, 255,       255],       dtype=np.uint8),
    )
    hist = cv2.calcHist([hsv], [0], gate, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def _save_histogram(hist: np.ndarray, output: str) -> None:
    np.savetxt(output, hist.flatten(), fmt="%.4f")
    mean, std = _fit_gaussian(hist)
    kept = int((hist.flatten() > 0).sum())
    print(f"[Cal] Histogram saved → {output}  ({len(hist)} bins)")
    print(f"[Cal] Fit: mean={mean:.1f}  std={std:.1f}  "
          f"±{_GAUSS_SIGMA:.1f}σ={_GAUSS_SIGMA*std:.1f}  active_bins={kept}/180")


def _select_and_calibrate(frame_bgr: np.ndarray, output: str):
    """Drag-select a ROI on the frozen frame, compute + save its histogram.

    Returns (hist, rect) on success, or (None, None) if cancelled / empty."""
    roi = cv2.selectROI(WINDOW, frame_bgr, fromCenter=False, showCrosshair=True)
    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        print("[Cal] Selection cancelled.")
        return None, None
    rect = (x, y, w, h)
    hist = _compute_histogram(frame_bgr, rect)
    if float(hist.sum()) <= 0:
        print("[Cal] Selected region has no usable colour (all pixels gated). "
              "Try a brighter / more saturated area.")
        return None, None
    _save_histogram(hist, output)
    cv2.imshow(HIST_WINDOW, _draw_histogram(hist))
    return hist, rect


def _resolve_crop(crop, fw: int, fh: int):
    """Resolve (offset_x, offset_y, w, h) against a frame, clamped to bounds.

    X/Y are the crop's top-left offset position; W/H the size (None = 'rest of
    dimension after the offset'). The offset is clamped into the frame and the
    size trimmed so the window never runs off the edge — an out-of-range offset
    yields the largest valid window at that corner instead of an empty crop."""
    cx, cy, cw, ch = crop
    cx = min(max(0, cx), max(0, fw - 1))
    cy = min(max(0, cy), max(0, fh - 1))
    cw = fw - cx if cw is None else min(cw, fw - cx)
    ch = fh - cy if ch is None else min(ch, fh - cy)
    return cx, cy, cw, ch


def _apply_crop(frame: np.ndarray, crop):
    if crop is None:
        return frame
    fh, fw = frame.shape[:2]
    cx, cy, cw, ch = _resolve_crop(crop, fw, fh)
    return frame[cy:cy + ch, cx:cx + cw]


def _draw_hud(display: np.ndarray, paused: bool, rect, output: str):
    h = display.shape[0]
    if paused:
        state, col = "PAUSED", (0, 215, 255)
    else:
        state, col = "PLAYING", (0, 255, 0)
    cv2.putText(display, state, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    if rect is not None:
        x, y, w, h_r = rect
        cv2.rectangle(display, (x, y), (x + w, y + h_r), (255, 255, 0), 2)
        cv2.putText(display, f"calib ROI {w}x{h_r}", (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    hint = ("P=pause+select   S=reselect   R=resume   Z=clear   Q=quit"
            if paused else
            "P=pause to select region   R=resume   Q=quit")
    cv2.putText(display, hint, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display, f"out: {output}", (10, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def main():
    ap = argparse.ArgumentParser(
        description="Pause-and-select colour calibration from a video source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--source", default="0",
                    help="Video file path or camera index (default: 0)")
    ap.add_argument("--res", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="Request this capture resolution (e.g. --res 1280 720)")
    ap.add_argument("--output", default="color_histogram.txt",
                    help="Output histogram file (default: color_histogram.txt)")
    ap.add_argument("--crop", type=str, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"),
                    help="Crop each frame to this ROI (e.g. --crop 320 180 640 360). "
                         "Use - for W or H to mean 'rest of dimension after offset'.")
    ap.add_argument("--flip", action="store_true", default=False,
                    help="Flip frames 180° (both axes) for upside-down cameras")
    args = ap.parse_args()

    crop = (tuple(None if v == '-' else int(v) for v in args.crop)
            if args.crop else None)
    source = _parse_source(args.source)
    is_file = isinstance(source, str)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source {source!r}", file=sys.stderr)
        sys.exit(1)
    if args.res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.res[1])

    # Pace file playback to its native FPS; cameras self-pace, so fall back to a
    # snappy 1 ms wait for them. waitKey doubles as both the frame delay and the
    # key reader, so playback speed is "real time" without a separate sleep.
    fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0
    play_delay = max(1, int(round(1000.0 / fps))) if fps and fps > 0 else 1
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_msg = ""
    if crop is not None and actual_w > 0 and actual_h > 0:
        rx, ry, rw, rh = _resolve_crop(crop, actual_w, actual_h)
        crop_msg = f"  crop=offset({rx},{ry}) size {rw}x{rh}"
    print(f"[Cal] Opened {source!r}  {actual_w}x{actual_h}"
          + (f"  {fps:.1f}fps" if fps else "")
          + crop_msg)

    if _load_histogram(args.output) is not None:
        print(f"[Cal] Existing calibration found at {args.output} "
              "(will be overwritten on save).")
    print("[Cal] Press P to pause and drag-select a region; R to resume.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(HIST_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(HIST_WINDOW, 360, 230)

    paused   = False
    frozen   = None    # frame held while paused
    rect     = None    # last confirmed calibration ROI (drawn as overlay)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file:
                    # Loop the clip so the user can keep scrubbing for a frame.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[Cal] End of stream.")
                break
            if args.flip:
                frame = cv2.flip(frame, -1)
            frame = _apply_crop(frame, crop)
            current = frame
            rect = None   # selection only meaningful on a frozen frame
        else:
            current = frozen

        display = current.copy()
        _draw_hud(display, paused, rect, args.output)
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(play_delay if not paused else 20) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = True
            frozen = current.copy()
            print("[Cal] Paused — drag a box over the target colour.")
            _, rect = _select_and_calibrate(frozen, args.output)
        elif key == ord('s') and paused:
            _, rect = _select_and_calibrate(frozen, args.output)
        elif key == ord('r'):
            if paused:
                print("[Cal] Resumed.")
            paused = False
            rect = None
        elif key == ord('z'):
            rect = None
            print("[Cal] Selection cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
