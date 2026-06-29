#!/usr/bin/env python3
"""
test_detect_color.py — pure colour detect + track, same pipeline as main.py, no MAVLink.

Drives the Seeker vision engine directly (the same object SeekerCtrl wraps in
main.py) and runs its built-in detect → track → display loop. It mirrors main.py's
source / display / tracker / detection options so the colour detection and tracking
behaviour is identical; only the flight-control / MAVLink layer is omitted.

Video-file sources play at their native FPS (the Seeker capture thread paces file
reads); camera / UDP sources run flat-out, paced by the driver.

Dropped from main.py (control / MAVLink layer, not part of pure detection):
    --connection --baud --auto --px4 --no-prediction --no-hud-pitch --no-hud-yaw
    --record --debug --max-fps

Usage:
    python3 script/test_detect_color.py
    python3 script/test_detect_color.py --source video.mp4
    python3 script/test_detect_color.py --udpsrc 5600 --udpsrc-codec h264
    python3 script/test_detect_color.py --tracker meanshift,kalman --mask
    python3 script/test_detect_color.py --crop 320 180 640 360 --mask-algo inrange

Controls (from Seeker.run):
    q   quit
    r   reset the tracker
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Import the Seeker vision engine from the project root (one level above script/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seeker import Seeker  # noqa: E402

_SHIFT_ALGOS         = {"camshift", "meanshift"}
_APPEARANCE_TRACKERS = {"mil"}


def _parse_tracker_opt(value: str) -> tuple[bool, str, bool, str]:
    """Parse '--tracker' value into (use_camshift, shift_algo, use_kalman, tracker_name).

    Tokens (comma-separated):
      camshift / meanshift  — shift-based tracker variant (mutually exclusive)
      kalman                — enable Kalman filter
      mil                   — MIL appearance tracker (mutually exclusive with shift)
    Examples:
      camshift,kalman   → CamShift + Kalman  (default)
      camshift          → CamShift, no Kalman
      meanshift,kalman  → MeanShift + Kalman
      mil               → MIL tracker, no Kalman
      mil,kalman        → MIL tracker + Kalman
    """
    tokens = {t.strip().lower() for t in value.split(",") if t.strip()}
    _VALID = {"kalman"} | _SHIFT_ALGOS | _APPEARANCE_TRACKERS
    unknown = tokens - _VALID
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown tracker token(s): {', '.join(sorted(unknown))}. "
            f"Valid: kalman, {', '.join(sorted(_SHIFT_ALGOS))}, mil"
        )
    shift = tokens & _SHIFT_ALGOS
    if len(shift) > 1:
        raise argparse.ArgumentTypeError(
            "Cannot combine 'camshift' and 'meanshift' — they are mutually exclusive."
        )
    tracker_name = "mil" if "mil" in tokens else ""
    shift_algo   = next(iter(shift), "camshift")
    use_camshift = bool(shift) and not tracker_name
    use_kalman   = "kalman" in tokens
    if tracker_name and shift:
        raise argparse.ArgumentTypeError(
            f"Cannot combine 'mil' with '{shift_algo}' — they are mutually exclusive."
        )
    return use_camshift, shift_algo, use_kalman, tracker_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure colour detect + track — same vision pipeline as main.py, no MAVLink",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0), video file path (e.g. video.mp4), "
             "or GStreamer pipeline string. Overridden by --udpsrc.",
    )
    parser.add_argument(
        "--udpsrc",
        type=int,
        default=None,
        metavar="PORT",
        help="Receive camera stream from UDP port via GStreamer (e.g. --udpsrc 5600). "
             "Overrides --source. Use --udpsrc-codec to select codec.",
    )
    parser.add_argument(
        "--udpsrc-codec",
        default="h264",
        choices=["h264", "mjpeg"],
        metavar="CODEC",
        help="Codec for --udpsrc stream: h264 (default) or mjpeg.",
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Request this capture resolution from the camera (e.g. --res 1280 720)",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        default=False,
        help="Show calibration histogram in a separate window (default: disabled)",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        default=False,
        help="Show detection mask in a separate window (default: disabled)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="Headless: no OpenCV preview window (also disables histogram/mask windows).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="Cap CPU threads (OpenCV/BLAS) to N to flatten peak current "
             "(e.g. --threads 2). Default: uncapped.",
    )
    parser.add_argument(
        "--tracker",
        default="camshift,kalman",
        metavar="TOKENS",
        help=(
            "Comma-separated tracking components (default: camshift,kalman). "
            "Tokens: camshift  meanshift  kalman  mil. "
            "Examples: 'camshift,kalman'  'meanshift,kalman'  'mil'  'mil,kalman'"
        ),
    )
    parser.add_argument(
        "--no-box-filter",
        action="store_true",
        default=False,
        help="Disable box-like shape filter; accept any blob regardless of shape",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        default=False,
        help="Flip captured frames 180 degrees (both axes). Use when camera is mounted upside-down.",
    )
    parser.add_argument(
        "--mask-algo",
        default="all",
        choices=["gaussian", "adaptive", "inrange", "all"],
        help="Detection mask algorithm: gaussian, adaptive, inrange, or all (2-of-3 vote, default)",
    )
    parser.add_argument(
        "--crop",
        type=str,
        nargs=4,
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="Crop each frame to this ROI (e.g. --crop 320 180 640 360). "
             "Use - for W or H to mean 'rest of dimension after offset'.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable per-frame stage timing capture in the tracker (internal).",
    )
    return parser.parse_args()


def _parse_source(value: str) -> int | str:
    """Return an int for camera index, or str for file path."""
    try:
        return int(value)
    except ValueError:
        return value


def _build_udpsrc_pipeline(port: int, codec: str) -> str:
    """GStreamer pipeline that receives an RTP video stream on `port` and hands
    decoded BGR frames to OpenCV via appsink. Ported from main.py so the test
    harness consumes the same UDP source the seeker uses."""
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


def main():
    args = parse_args()

    if args.threads:
        import cv2
        cv2.setNumThreads(args.threads)
        print(f"[test] thread cap = {args.threads} (OpenCV/BLAS)")

    if args.udpsrc is not None:
        source = _build_udpsrc_pipeline(args.udpsrc, args.udpsrc_codec)
    else:
        source = _parse_source(args.source)

    try:
        use_camshift, shift_algo, use_kalman, tracker_name = _parse_tracker_opt(args.tracker)
    except argparse.ArgumentTypeError as e:
        print(f"error: --tracker: {e}")
        raise SystemExit(1)

    seeker = Seeker(
        source=source,
        capture_width=args.res[0] if args.res else None,
        capture_height=args.res[1] if args.res else None,
        crop=tuple(None if v == '-' else int(v) for v in args.crop) if args.crop else None,
        show_histogram=args.histogram,
        show_mask=args.mask,
        display=not args.no_display,
        mask_algo=args.mask_algo,
        use_camshift=use_camshift,
        shift_algo=shift_algo,
        box_filter=not args.no_box_filter,
        use_kalman=use_kalman,
        tracker=tracker_name,
        flip=args.flip,
    )
    seeker.profile = args.profile

    try:
        seeker.run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
