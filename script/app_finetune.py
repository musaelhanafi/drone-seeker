#!/usr/bin/env python3
"""
Fine-tune the colour calibration from the target actually being tracked.

app_calibrate.py builds color_histogram.txt from a single hand-picked ROI in one
frame. That snapshot is only ever right for the distance, lighting and exposure
of that one frame — as the target closes in, its apparent hue shifts, and a
histogram fitted too tightly around the original sample stops firing. The
symptom is a detector that only wakes up once the target is already large.

This script closes the loop instead: it runs the real detect → track pipeline
over a clip (or a live stream), samples the hue of pixels inside the window the
tracker is actually holding, and refits the histogram from those samples. The
tracker supplies the target's location on frames where the *detector* alone
would have found nothing, so the refined histogram learns the target as it truly
appears across the whole run, not at one instant.

Only the core of the tracked window is sampled (--core), and only pixels that
clear saturation/value floors and sit within --sigma-window of the seed hue, so
background inside the box cannot drag the calibration off the target.

The result is verified before it is written: the clip is replayed with the old
and the new histogram and both are scored on lock retention. A refit that scores
worse is reported and NOT saved unless you pass --force.

Usage:
    python3 script/app_finetune.py --source rec.mp4
    python3 script/app_finetune.py --source rec.mp4 --blend 1.0 --core 0.5
    python3 script/app_finetune.py --source rec.mp4 --dry-run
    python3 script/app_finetune.py --udpsrc 5600 --frames 600
"""

from __future__ import annotations
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seeker import (Seeker, _apply_outres, _fit_gaussian,   # noqa: E402
                    _load_histogram)

_SHIFT_ALGOS = {"camshift", "meanshift"}
_APPEARANCE  = {"mil"}


def _parse_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def _build_udpsrc_pipeline(port: int, codec: str) -> str:
    """GStreamer pipeline receiving an RTP H.264/MJPEG stream on a UDP port.

    Matches script/app_calibrate.py so a stream can be fine-tuned with the same
    --udpsrc/--udpsrc-codec invocation used to calibrate it."""
    if codec == "mjpeg":
        return (
            f"udpsrc port={port} "
            "! application/x-rtp,encoding-name=JPEG "
            "! rtpjpegdepay ! jpegdec ! videoconvert "
            "! appsink drop=1 max-buffers=1"
        )
    return (
        f"udpsrc port={port} "
        "! application/x-rtp,payload=96 "
        "! rtph264depay ! avdec_h264 ! videoconvert "
        "! appsink drop=1 max-buffers=1"
    )


def _parse_tracker_opt(value: str) -> tuple[bool, str, bool, str]:
    """Parse '--tracker' into (use_camshift, shift_algo, use_kalman, tracker)."""
    tokens = {t.strip().lower() for t in value.split(",") if t.strip()}
    valid = {"kalman"} | _SHIFT_ALGOS | _APPEARANCE
    unknown = tokens - valid
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown tracker token(s): {', '.join(sorted(unknown))}. "
            f"Valid: kalman, {', '.join(sorted(_SHIFT_ALGOS))}, mil")
    shift = tokens & _SHIFT_ALGOS
    if len(shift) > 1:
        raise argparse.ArgumentTypeError(
            "Cannot combine 'camshift' and 'meanshift' — mutually exclusive.")
    tracker = "mil" if "mil" in tokens else ""
    if tracker and shift:
        raise argparse.ArgumentTypeError(
            "Cannot combine 'mil' with a shift tracker — mutually exclusive.")
    return bool(shift) and not tracker, next(iter(shift), "camshift"), \
        "kalman" in tokens, tracker


# ── frame source ─────────────────────────────────────────────────────────────

def _open_source(source, res, flip):
    """Return (cap, is_picam). Files/pipelines via OpenCV, ints via Picamera2."""
    if isinstance(source, int):
        rw, rh = res if res else (1280, 720)
        try:
            from seeker import Picamera2Capture
            print(f"[Tune] Using Picamera2 backend  {rw}x{rh}")
            return Picamera2Capture(rw, rh, flip=flip), True
        except Exception as e:                                # noqa: BLE001
            print(f"[Tune] Picamera2 unavailable ({e}); trying OpenCV V4L2")
            cap = cv2.VideoCapture(source)
    elif isinstance(source, str) and " ! " in source:
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(source)
    if res and not isinstance(source, str):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    return cap, False


def _prepare(frame, outres, crop, flip, is_picam):
    """Apply the same transforms Seeker.read_frame does: outres, then crop."""
    if flip and not is_picam:            # Picamera2 flips in hardware
        frame = cv2.flip(frame, -1)
    frame = _apply_outres(frame, outres)
    if crop is not None:
        x, y, w, h = crop
        fh, fw = frame.shape[:2]
        w = fw - x if w is None else w
        h = fh - y if h is None else h
        frame = frame[y:y + h, x:x + w]
    return frame


def _read_all(source, res, outres, crop, flip, limit):
    """Read frames into memory (files) or grab `limit` frames from a stream."""
    cap, is_picam = _open_source(source, res, flip)
    ok_open = cap.isOpened() if hasattr(cap, "isOpened") else True
    if not ok_open:
        sys.exit(f"[Tune] ERROR: cannot open source {source!r}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(_prepare(frame, outres, crop, flip, is_picam))
        if limit and len(frames) >= limit:
            break
    try:
        cap.release()
    except Exception:                                          # noqa: BLE001
        pass
    return frames


# ── histogram maths ──────────────────────────────────────────────────────────

def _normalise(counts: np.ndarray) -> np.ndarray:
    """Scale a 180-bin histogram to 0-255, matching calibrate_color.py output."""
    counts = counts.astype(np.float32)
    peak = counts.max()
    if peak <= 0:
        return counts
    return counts * (255.0 / peak)


def _circ_delta(bins: np.ndarray, mean: float) -> np.ndarray:
    """Signed hue distance to *mean*, wrapped to [-90, 90]."""
    return ((bins - mean + 90.0) % 180.0) - 90.0


def _collect(frames, seeker, args, seed_mean, seed_std):
    """Track through *frames*, accumulating hue counts from the tracked ROI."""
    counts = np.zeros(180, dtype=np.float64)
    hue_window = args.sigma_window * seed_std if args.sigma_window > 0 else None
    locked = sampled = 0

    for frame in frames:
        _, cx, cy = seeker.track(frame)
        if cx is None or seeker._track_win is None:
            continue
        locked += 1
        # Skip coasting frames: the window is a prediction, not an observation.
        if getattr(seeker, "_miss_count", 0) > 0:
            continue

        x, y, w, h = seeker._track_win
        fh, fw = frame.shape[:2]
        # Shrink to the window core so the padded border (mostly background)
        # never contributes hue samples.
        cw, ch = max(2, int(w * args.core)), max(2, int(h * args.core))
        cx0 = max(0, min(x + (w - cw) // 2, fw - 1))
        cy0 = max(0, min(y + (h - ch) // 2, fh - 1))
        cw, ch = min(cw, fw - cx0), min(ch, fh - cy0)
        if cw < 2 or ch < 2:
            continue

        hsv = cv2.cvtColor(frame[cy0:cy0 + ch, cx0:cx0 + cw], cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        keep = (sat >= args.sat_min) & (val >= args.val_min)
        if hue_window is not None:
            keep &= np.abs(_circ_delta(hue.astype(np.float32), seed_mean)) <= hue_window
        sel = hue[keep]
        if sel.size == 0:
            continue
        counts += np.bincount(sel.ravel(), minlength=180)
        sampled += 1

    return counts, locked, sampled


# ── scoring ──────────────────────────────────────────────────────────────────

def _score(frames, hist_path, args) -> tuple[float, int, int]:
    """Replay *frames* with the histogram at *hist_path*.

    Returns (tracked%, lock losses, first-lock frame). The first-lock frame is
    the one that matters for an approach run: a calibration that acquires the
    target earlier buys the seeker more time to guide on it.
    """
    use_camshift, shift_algo, use_kalman, tracker = _parse_tracker_opt(args.tracker)
    sk = Seeker(source=0, histogram_file=str(hist_path), display=False,
                mask_algo=args.mask_algo, use_camshift=use_camshift,
                shift_algo=shift_algo, box_filter=not args.no_box_filter,
                use_kalman=use_kalman, tracker=tracker)
    tracked = resets = 0
    had = False
    first = -1
    for i, frame in enumerate(frames):
        _, cx, _ = sk.track(frame)
        if cx is not None:
            tracked += 1
            if first < 0:
                first = i
        elif had:
            resets += 1
        had = cx is not None
    return 100.0 * tracked / max(len(frames), 1), resets, first


def main():
    ap = argparse.ArgumentParser(
        description="Refit color_histogram.txt from the tracked target ROI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="Camera index, video file, or GStreamer pipeline")
    ap.add_argument("--udpsrc", type=int, default=None, metavar="PORT",
                    help="Receive an RTP stream on this UDP port (overrides --source)")
    ap.add_argument("--udpsrc-codec", default="h264", choices=["h264", "mjpeg"],
                    help="Codec for --udpsrc")
    ap.add_argument("--res", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="Capture resolution to request")
    ap.add_argument("--outres", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="Scale each frame to W H before cropping")
    ap.add_argument("--crop", type=str, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"),
                    help="Crop region; '-' for W/H means 'to the edge'")
    ap.add_argument("--flip", action="store_true",
                    help="Rotate the image 180 degrees")
    ap.add_argument("--frames", type=int, default=0, metavar="N",
                    help="Stop after N frames (0 = whole file). Required for live streams")
    ap.add_argument("--tracker", default="camshift,kalman", type=str,
                    help="Tracker used to follow the target while sampling")
    ap.add_argument("--mask-algo", default="all",
                    choices=["gaussian", "adaptive", "inrange", "all"],
                    help="Detection algorithm(s)")
    ap.add_argument("--no-box-filter", action="store_true",
                    help="Disable the blob extent/solidity filter")
    ap.add_argument("--histogram", default="color_histogram.txt",
                    help="Seed histogram to refine")
    ap.add_argument("--output", default=None,
                    help="Where to write the refined histogram (default: in place)")
    ap.add_argument("--blend", type=float, default=0.5, metavar="W",
                    help="Weight of the newly sampled histogram (1.0 = replace seed)")
    ap.add_argument("--core", type=float, default=0.6, metavar="F",
                    help="Fraction of the tracked window sampled, centred")
    ap.add_argument("--sat-min", type=int, default=40,
                    help="Minimum saturation for a pixel to be sampled")
    ap.add_argument("--val-min", type=int, default=40,
                    help="Minimum value for a pixel to be sampled")
    ap.add_argument("--sigma-window", type=float, default=6.0, metavar="K",
                    help="Only sample hues within K*sigma of the seed (0 = no limit)")
    ap.add_argument("--min-samples", type=int, default=20, metavar="N",
                    help="Refuse to refit on fewer than N sampled frames")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the refit and its score but write nothing")
    ap.add_argument("--force", action="store_true",
                    help="Write even if the refit scores worse than the seed")
    args = ap.parse_args()

    if not 0.0 <= args.blend <= 1.0:
        sys.exit("[Tune] ERROR: --blend must be between 0.0 and 1.0")
    if not 0.1 <= args.core <= 1.0:
        sys.exit("[Tune] ERROR: --core must be between 0.1 and 1.0")

    hist_path = Path(args.histogram)
    seed = _load_histogram(str(hist_path))
    if seed is None:
        sys.exit(f"[Tune] ERROR: need a seed histogram — run app_calibrate.py first "
                 f"(looked for {hist_path})")
    seed_mean, seed_std = _fit_gaussian(seed)
    print(f"[Tune] Seed: mean={seed_mean:.1f}  std={seed_std:.1f}  "
          f"bins={(seed.flatten() > 0).sum()}/180")

    source = (_build_udpsrc_pipeline(args.udpsrc, args.udpsrc_codec)
              if args.udpsrc is not None else _parse_source(args.source))
    is_live = isinstance(source, int) or (isinstance(source, str) and " ! " in source)
    if is_live and not args.frames:
        sys.exit("[Tune] ERROR: --frames N is required for a live camera/stream "
                 "(there is no end of file to stop at)")

    crop = (tuple(None if v == '-' else int(v) for v in args.crop)
            if args.crop else None)
    outres = tuple(args.outres) if args.outres else None

    print(f"[Tune] Reading frames from {source!r} …")
    frames = _read_all(source, tuple(args.res) if args.res else None,
                       outres, crop, args.flip, args.frames)
    if not frames:
        sys.exit("[Tune] ERROR: no frames read from source")
    fh, fw = frames[0].shape[:2]
    print(f"[Tune] {len(frames)} frames at {fw}x{fh}")

    # ── Pass 1: track with the seed and sample the tracked ROI ───────────────
    use_camshift, shift_algo, use_kalman, tracker = _parse_tracker_opt(args.tracker)
    sk = Seeker(source=0, histogram_file=str(hist_path), display=False,
                mask_algo=args.mask_algo, use_camshift=use_camshift,
                shift_algo=shift_algo, box_filter=not args.no_box_filter,
                use_kalman=use_kalman, tracker=tracker)
    counts, locked, sampled = _collect(frames, sk, args, seed_mean, seed_std)
    print(f"[Tune] Tracked on {locked}/{len(frames)} frames; "
          f"sampled hue from {sampled}")

    if sampled < args.min_samples:
        sys.exit(f"[Tune] ERROR: only {sampled} usable frames (need "
                 f"{args.min_samples}). The seed histogram may be too far off to "
                 f"bootstrap from — recalibrate with app_calibrate.py instead.")
    if counts.sum() <= 0:
        sys.exit("[Tune] ERROR: no pixels passed the saturation/value floors")

    # ── Refit ────────────────────────────────────────────────────────────────
    fresh = _normalise(counts)
    refined = _normalise(args.blend * fresh + (1.0 - args.blend) * seed.flatten())
    new_mean, new_std = _fit_gaussian(refined.reshape(180, 1))
    print(f"[Tune] Refit: mean={new_mean:.1f} (seed {seed_mean:.1f}, "
          f"delta {_circ_delta(np.array([new_mean]), seed_mean)[0]:+.1f})  "
          f"std={new_std:.1f} (seed {seed_std:.1f})  "
          f"bins={(refined > 0).sum()}/180 (seed {(seed.flatten() > 0).sum()})")

    # ── Verify before writing ────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        np.savetxt(tmp.name, refined, fmt="%.4f")
        tmp_path = Path(tmp.name)
    try:
        old_pct, old_res, old_first = _score(frames, hist_path, args)
        new_pct, new_res, new_first = _score(frames, tmp_path, args)
    finally:
        tmp_path.unlink(missing_ok=True)

    def _first(v):
        return "never" if v < 0 else f"frame {v}"

    print(f"[Tune] Score on this clip  (tracker: {args.tracker})")
    print(f"[Tune]   seed    : tracked {old_pct:5.1f}%   lock losses {old_res:3d}   "
          f"first lock {_first(old_first)}")
    print(f"[Tune]   refined : tracked {new_pct:5.1f}%   lock losses {new_res:3d}   "
          f"first lock {_first(new_first)}")
    if old_first >= 0 and new_first >= 0 and new_first != old_first:
        d = old_first - new_first
        print(f"[Tune]   acquisition is {abs(d)} frames "
              f"{'EARLIER' if d > 0 else 'later'} with the refit")

    better = (new_pct > old_pct + 0.5) or (new_pct >= old_pct - 0.5 and new_res < old_res)
    if args.dry_run:
        print("[Tune] --dry-run: nothing written.")
        return
    if not better and not args.force:
        print("[Tune] Refit is not an improvement on this clip — NOT written. "
              "Re-run with --force to write it anyway, or try --blend 1.0 / "
              "--core 0.5 / a clip where the target is better exposed.")
        return

    out_path = Path(args.output) if args.output else hist_path
    if out_path.exists():
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.copy2(out_path, backup)
        print(f"[Tune] Backed up {out_path} → {backup}")
    np.savetxt(out_path, refined, fmt="%.4f")
    print(f"[Tune] Wrote refined histogram → {out_path}"
          + ("  (forced)" if not better else ""))


if __name__ == "__main__":
    main()
