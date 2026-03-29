import argparse

from seekerctrl import SeekerCtrl


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
    )
    ctrl.connect()

    try:
        ctrl.run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
