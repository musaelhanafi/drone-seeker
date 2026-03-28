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
    )
    ctrl.connect()

    try:
        ctrl.run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
