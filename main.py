import argparse

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
        help="Camera index (e.g. 0) or video file path (e.g. video.mp4)",
    )
    parser.add_argument(
        "--joystick",
        action="store_true",
        default=False,
        help="Enable joystick RC override (default: disabled)",
    )
    parser.add_argument(
        "--joy-index",
        type=int,
        default=0,
        help="Joystick device index (default: 0)",
    )
    parser.add_argument(
        "--joy-rate",
        type=int,
        default=50,
        help="Joystick send rate in Hz (default: 50)",
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
        help="Log tracking telemetry to tracking.csv while in TRACKING mode",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Record annotated video to tracking_<timestamp>.avi while in TRACKING mode",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=False,
        help="Auto mode: enter TRACKING when within 1 km of target; ch6 keeps FC in AUTO otherwise",
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
    return parser.parse_args()


def _parse_source(value: str) -> int | str:
    """Return an int for camera index, or str for file path."""
    try:
        return int(value)
    except ValueError:
        return value


def main():
    args   = parse_args()
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
        joystick_enabled=args.joystick,
        joystick_index=args.joy_index,
        joystick_rate=args.joy_rate,
        capture_width=args.res[0] if args.res else None,
        capture_height=args.res[1] if args.res else None,
        crop=tuple(None if v == '-' else int(v) for v in args.crop) if args.crop else None,
        show_histogram=args.histogram,
        show_mask=args.mask,
        debug_log=args.debug,
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
        auto_mode=args.auto,
    )
    ctrl.connect()

    try:
        ctrl.run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
