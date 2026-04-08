"""
test_joystick.py — Live joystick → MAVLink RC_CHANNELS_OVERRIDE sender.

Opens a joystick and a MAVLink connection, then sends RC channel override
whenever joystick input changes (and at a minimum keep-alive rate).

CH6 is NOT sent as a raw RC override — instead it toggles the flight mode:
  CH6 high (2000) → AUTOTUNE   (ArduPlane custom_mode 8)
  CH6 low  (1000) → STABILIZE  (ArduPlane custom_mode 2)
The mode command is sent once on each rising/falling edge.

Usage:
    python3 test_joystick.py --connect udp:127.0.0.1:14550
    python3 test_joystick.py --connect /dev/ttyUSB0 --baud 57600 --joy 0 --rate 20
"""

import argparse
import signal
import time
import threading

from pymavlink import mavutil

from joystick_handler import JoystickHandler

UINT16_MAX = 65535
_RELEASE   = {k: 0 for k in ("ch1", "ch2", "ch3", "ch4", "ch5")}

_MODE_STABILIZE = 2
_MODE_AUTOTUNE  = 8
_MODE_NAMES     = {_MODE_STABILIZE: "STABILIZE", _MODE_AUTOTUNE: "AUTOTUNE"}


def send_rc_override(master, ch: dict):
    """Send ch1-ch5 as RC override; ch6 is handled as a mode command instead."""
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        ch.get("ch1", UINT16_MAX),
        ch.get("ch2", UINT16_MAX),
        ch.get("ch3", UINT16_MAX),
        ch.get("ch4", UINT16_MAX),
        ch.get("ch5", UINT16_MAX),
        UINT16_MAX,   # ch6 — not overridden, handled via mode command
        UINT16_MAX,   # ch7 unused
        UINT16_MAX,   # ch8 unused
    )


def set_mode(master, custom_mode: int):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        custom_mode,
        0, 0, 0, 0, 0,
    )
    print(f"\n[MODE] → {_MODE_NAMES.get(custom_mode, custom_mode)}")


def print_channels(ch: dict, mode_label: str):
    keys  = ("ch1", "ch2", "ch3", "ch4", "ch5")
    parts = "  ".join(f"{k.upper()}:{ch[k]:4d}" for k in keys if k in ch)
    print(f"\r{parts}  CH6-MODE:{mode_label:<10}", end="", flush=True)


def joystick_loop(master, joy: JoystickHandler, rate: float, stop: threading.Event):
    interval   = 1.0 / rate
    prev_ch    = {}
    last_send  = 0.0
    prev_ch6   = None   # track edge: None = unknown, 1000 or 2000
    mode_label = "?"

    while not stop.is_set():
        now = time.monotonic()

        joy.pump()
        ch = joy.read_channels()

        # --- CH6 edge detection → mode command ---
        cur_ch6 = ch["ch6"]
        if cur_ch6 != prev_ch6:
            if cur_ch6 == 2000:
                set_mode(master, _MODE_AUTOTUNE)
                mode_label = "AUTOTUNE"
            else:
                set_mode(master, _MODE_STABILIZE)
                mode_label = "STABILIZE"
            prev_ch6 = cur_ch6

        # --- CH1-CH5 RC override (change-triggered + keep-alive) ---
        rc_ch = {k: ch[k] for k in ("ch1", "ch2", "ch3", "ch4", "ch5")}
        prev_rc = {k: prev_ch.get(k) for k in rc_ch}
        changed = rc_ch != prev_rc

        if changed or (now - last_send) >= interval:
            send_rc_override(master, rc_ch)
            print_channels(rc_ch, mode_label)
            prev_ch   = ch
            last_send = now

        sleep_for = interval - (time.monotonic() - now)
        if sleep_for > 0:
            time.sleep(sleep_for)

    # Release override so FC resumes normal RC input
    send_rc_override(master, _RELEASE)
    print("\n[JOY] RC override released")


def main():
    parser = argparse.ArgumentParser(description="Joystick → MAVLink RC override sender")
    parser.add_argument("--connect", default="udp:127.0.0.1:14550",
                        help="MAVLink connection string (default: udp:127.0.0.1:14550)")
    parser.add_argument("--baud",    type=int, default=57600,
                        help="Baud rate for serial connections (default: 57600)")
    parser.add_argument("--joy",     type=int, default=0,
                        help="Joystick index (default: 0)")
    parser.add_argument("--rate",    type=float, default=20.0,
                        help="Minimum send rate in Hz even without change (default: 20)")
    args = parser.parse_args()

    # --- MAVLink connection ---
    print(f"[MAV] Connecting to {args.connect} ...")
    master = mavutil.mavlink_connection(args.connect, baud=args.baud)
    master.wait_heartbeat()
    print(f"[MAV] Heartbeat from system {master.target_system} "
          f"component {master.target_component}")

    # --- Joystick ---
    joy = JoystickHandler(joy_index=args.joy, thr_invert=False)
    joy.open()

    # --- Graceful shutdown on Ctrl-C / SIGTERM ---
    stop = threading.Event()

    def _shutdown(sig, frame):
        print("\n[SIG] Shutting down ...")
        stop.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[JOY] Sending RC override at up to {args.rate:.0f} Hz  (Ctrl-C to quit)")
    print("[JOY] CH6 high=AUTOTUNE  CH6 low=STABILIZE\n")

    try:
        joystick_loop(master, joy, args.rate, stop)
    finally:
        joy.close()
        print("[JOY] Joystick closed")


if __name__ == "__main__":
    main()
