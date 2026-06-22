import argparse
import os

# Force pymavlink to use MAVLink 2.0 — TRACKING_MESSAGE (msgid 11045) is only
# defined in the v2.0 ardupilotmega dialect. With the default v1.0 dialect,
# `tracking_message_send` doesn't exist and our send_tracking() call would do
# nothing (PX4 ends up in TRACKING with no incoming errors).
os.environ["MAVLINK20"] = "1"


def _apply_thread_cap():
    """Cap CPU threads (BLAS/OpenMP) before numpy/cv2 import to flatten peak
    current — the Pi-5 browns out on OpenCV bursts. Honours --threads /
    SEEKER_THREADS (read from argv here because argparse runs after import)."""
    import sys
    t = os.environ.get("SEEKER_THREADS")
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            t = sys.argv[i + 1]
        elif a.startswith("--threads="):
            t = a.split("=", 1)[1]
    if t:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(v, t)


_apply_thread_cap()

from seekerctrl import SeekerCtrl

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
    parser = argparse.ArgumentParser(description="Drone Seeker — pink tracking + MAVLink")
    parser.add_argument(
        "--connection",
        default="udpin:0.0.0.0:14560",
        help="MAVLink connection string (default: udpin:0.0.0.0:14560)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=57600,
        help="Baud rate for serial connections (default: 57600)",
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
        help="Request this capture resolution from the camera (e.g. --capture-res 1280 720)",
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
        help="Headless: no OpenCV preview window (saves CPU/power on the drone). "
             "Also disables the histogram/mask windows.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="Cap CPU threads (OpenCV/BLAS) to N to flatten peak current and "
             "avoid Pi-5 brownout (e.g. --threads 2). Default: uncapped.",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=25.0,
        metavar="FPS",
        help="Cap processing rate to FPS (race-to-idle): sleeps between frames "
             "to cut average CPU/power. Default: 25. Use 0 to disable the cap.",
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
        "--no-hud-pitch",
        action="store_true",
        default=False,
        help="Disable pitch ladder in HUD",
    )
    parser.add_argument(
        "--no-hud-yaw",
        action="store_true",
        default=False,
        help="Disable yaw compass in HUD",
    )
    parser.add_argument(
        "--no-prediction",
        action="store_true",
        default=False,
        help="Disable latency + PN lead input prediction (default: enabled)",
    )
    parser.add_argument(
        "--mask-algo",
        default="all",
        choices=["gaussian", "adaptive", "inrange", "all"],
        help="Detection mask algorithm: gaussian, adaptive, inrange, or all (2-of-3 vote, default)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Log tracking telemetry to tracking.csv while in TRACKING mode. "
             "Also enables stage profiling and appends the reports to "
             "profile_<tracker>_<timestamp>.log.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Print per-stage loop timing (capture/track/ctrl+mav/hud/display) "
             "to stdout every ~2 s. (--debug also enables this and saves it to "
             "a logfile.)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Record annotated video to tracking_<timestamp>.avi while in TRACKING mode",
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
        "--auto",
        action="store_true",
        default=False,
        help="Force ch6 active (simulate PWM 1500) — seeker armed without physical RC switch",
    )
    parser.add_argument(
        "--px4",
        action="store_true",
        default=False,
        help="Target PX4 instead of ArduPilot: maps STABILIZE→STABILIZED (main 7), "
             "AUTO→AUTO_MISSION (main 4, sub 4). Heartbeat custom_mode decoded with "
             "PX4 encoding. Without this flag, normal ArduCopter mode codes are used.",
    )
    return parser.parse_args()


def _parse_source(value: str) -> int | str:
    """Return an int for camera index, or str for file path."""
    try:
        return int(value)
    except ValueError:
        return value


def _build_udpsrc_pipeline(port: int, codec: str) -> str:
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
        print(f"[main] thread cap = {args.threads} (OpenCV/BLAS)")

    if args.udpsrc is not None:
        source = _build_udpsrc_pipeline(args.udpsrc, args.udpsrc_codec)
    else:
        source = _parse_source(args.source)

    try:
        use_camshift, shift_algo, use_kalman, tracker_name = _parse_tracker_opt(args.tracker)
    except argparse.ArgumentTypeError as e:
        print(f"error: --tracker: {e}")
        raise SystemExit(1)

    ctrl = SeekerCtrl(
        connection_string=args.connection,
        baud=args.baud,
        source=source,
        capture_width=args.res[0] if args.res else None,
        capture_height=args.res[1] if args.res else None,
        crop=tuple(None if v == '-' else int(v) for v in args.crop) if args.crop else None,
        show_histogram=args.histogram,
        show_mask=args.mask,
        display=not args.no_display,
        max_fps=args.max_fps,
        debug_log=args.debug,
        profile=args.profile,
        record=args.record,
        input_prediction=not args.no_prediction,
        mask_algo=args.mask_algo,
        use_camshift=use_camshift,
        shift_algo=shift_algo,
        box_filter=not args.no_box_filter,
        use_kalman=use_kalman,
        tracker=tracker_name,
        hud_pitch=not args.no_hud_pitch,
        hud_yaw=not args.no_hud_yaw,
        auto=args.auto,
        flip=args.flip,
        px4=args.px4,
    )
    ctrl.connect()

    try:
        ctrl.run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
