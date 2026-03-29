import collections
import threading
import time

import cv2
from pymavlink import mavutil

from joystick_handler import JoystickHandler
from seeker import Seeker

# Tracking errors are sent as DEBUG_VECT (MAVLink ID 250, standard message).
# x = errorx, y = errory, z unused.
# ArduPlane decodes them in handle_message (case MAVLINK_MSG_ID_DEBUG_VECT).

# RC channel 6 PWM threshold to consider the switch "active"
_CH6_ACTIVE_PWM = 1400

# ArduPlane custom mode numbers
_TRACKING_MODE = 27
_LOITER_MODE   = 5   # fallback mode when ch6 goes low
_RTL_MODE      = 11  # RTL when no target and ch6 not armed

# ArduPlane custom_mode → display name
_PLANE_MODES: dict[int, str] = {
    0:  "MANUAL",        1:  "CIRCLE",      2:  "STABILIZE",
    3:  "TRAINING",      4:  "ACRO",        5:  "LOITER",
    6:  "FBW_B",         7:  "CRUISE",      8:  "AUTOTUNE",
    10: "AUTO",          11: "RTL",         12: "LOITER",
    13: "TAKEOFF",       15: "GUIDED",      17: "QSTABILIZE",
    18: "QHOVER",        19: "QLOITER",     20: "QLAND",
    21: "QRTL",          27: "TRACKING",
}


UINT16_MAX = 65535


class SeekerCtrl:
    def __init__(
        self,
        connection_string: str,
        baud: int = 57600,
        source: int | str = 0,
        joystick_enabled: bool = False,
        joystick_index: int = 0,
        joystick_rate: int = 50,

        capture_width: int | None = None,
        capture_height: int | None = None,
        crop: tuple[int | None, int | None, int | None, int | None] | None = None,
        show_histogram: bool = False,
        show_mask: bool = False,
    ):
        self.connection_string = connection_string
        self.baud   = baud
        self.master = None

        self.rc_channels    = {}
        self._in_tracking   = False   # True while flight mode is TRACKING
        self._flight_mode   = "?"     # last known flight mode name from HEARTBEAT
        self._prev_ch6_on   = False   # previous ch6 armed state for edge detection

        self.seeker = Seeker(source=source,
                             capture_width=capture_width,
                             capture_height=capture_height,
                             crop=crop,
                             show_histogram=show_histogram,
                             show_mask=show_mask)

        # ── Joystick ──────────────────────────────────────────────────────────
        self._joystick_enabled    = joystick_enabled
        self._joystick_index      = joystick_index
        self._joystick_rate       = joystick_rate
        self._joy_handler: JoystickHandler | None = None
        self._joy_thread:  threading.Thread | None = None
        self._joy_stop     = threading.Event()

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self):
        print(f"Connecting to {self.connection_string} ...")
        self.master = mavutil.mavlink_connection(
            self.connection_string, baud=self.baud
        )
        self.master.wait_heartbeat()
        print(
            f"Heartbeat received (system {self.master.target_system}, "
            f"component {self.master.target_component})"
        )
        if self._joystick_enabled:
            self._start_joystick_thread()

    # ── Joystick thread ───────────────────────────────────────────────────────

    def _start_joystick_thread(self):
        self._joy_handler = JoystickHandler(joy_index=self._joystick_index,
                                             thr_invert=False)
        self._joy_handler.open()
        self._joy_stop.clear()
        self._joy_thread = threading.Thread(
            target=self._joystick_loop, daemon=True, name="joystick"
        )
        self._joy_thread.start()
        print("[JOY] Joystick thread started")

    def _joystick_loop(self):
        interval   = 1.0 / self._joystick_rate
        prev_ch    = {}
        last_send  = 0.0

        while not self._joy_stop.is_set():
            now = time.monotonic()
            ch  = self._joy_handler.read_channels()
            changed = ch != prev_ch

            if changed or now - last_send >= interval:
                last_send = now
                prev_ch   = ch
                self._send_rc_override(ch)

            sleep_for = interval - (time.monotonic() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

        # Release override on exit
        self._send_rc_override({k: 0 for k in ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6")})
        print("[JOY] RC override released")

    def _send_rc_override(self, ch: dict):
        if self.master is None:
            return
        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            ch.get("ch1", UINT16_MAX),
            ch.get("ch2", UINT16_MAX),
            ch.get("ch3", UINT16_MAX),
            ch.get("ch4", UINT16_MAX),
            ch.get("ch5", UINT16_MAX),
            ch.get("ch6", UINT16_MAX),
            UINT16_MAX,
            UINT16_MAX,
        )

    def _stop_joystick_thread(self):
        if self._joy_thread is not None:
            self._joy_stop.set()
            self._joy_thread.join(timeout=2.0)
            self._joy_thread = None
        if self._joy_handler is not None:
            self._joy_handler.close()
            self._joy_handler = None

    # ── RC (non-blocking poll) ────────────────────────────────────────────────

    def _poll_rc(self):
        """Non-blocking: drain any pending RC_CHANNELS messages and update
        self.rc_channels.  Returns the latest dict (may be unchanged)."""
        while True:
            msg = self.master.recv_match(type="RC_CHANNELS", blocking=False)
            if msg is None:
                break
            channels = {
                f"ch{i}": getattr(msg, f"chan{i}_raw")
                for i in range(1, msg.chancount + 1)
                if hasattr(msg, f"chan{i}_raw")
            }
            if channels != self.rc_channels:
                self.rc_channels = channels
                print("RC:", channels)
        return self.rc_channels

    def _poll_heartbeat(self):
        """Update _flight_mode from the last HEARTBEAT stored by pymavlink."""
        msg = self.master.messages.get("HEARTBEAT")
        if msg:
            self._flight_mode = _PLANE_MODES.get(msg.custom_mode,
                                                  f"MODE({msg.custom_mode})")

    def _ch6_active(self) -> bool:
        pwm = self.rc_channels.get("ch6", 0)
        return pwm >= _CH6_ACTIVE_PWM

    # ── Flight mode ───────────────────────────────────────────────────────────

    def _set_mode(self, custom_mode: int, label: str):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            custom_mode,
            0, 0, 0, 0, 0,
        )
        print(f"[Ctrl] Mode → {label}")

    def set_mode_tracking(self):
        self._set_mode(_TRACKING_MODE, "TRACKING")

    def set_mode_loiter(self):
        self._set_mode(_LOITER_MODE, "LOITER")

    def set_mode_rtl(self):
        self._set_mode(_RTL_MODE, "RTL")

    # ── TRACKING MAVLink message ──────────────────────────────────────────────

    def send_tracking(self, errorx: float, errory: float):
        """Send tracking errors via DEBUG_VECT (ID 250).

        errorx → x field, errory → y field, both normalised [-1, 1].
        """
        self.master.mav.debug_vect_send(
            b"tracking\x00\x00",
            int(time.monotonic() * 1e6),
            errorx, errory, 0.0,
        )

    # ── Standalone RC monitor (blocking, for debugging) ───────────────────────

    def monitor_rc(self):
        """Block and print RC channel values whenever they change."""
        if self.master is None:
            raise RuntimeError("Not connected. Call connect() first.")
        print("Monitoring RC channels (Ctrl+C to stop) ...")
        while True:
            msg = self.master.recv_match(type="RC_CHANNELS", blocking=True)
            if msg is None:
                continue
            channels = {
                f"ch{i}": getattr(msg, f"chan{i}_raw")
                for i in range(1, msg.chancount + 1)
                if hasattr(msg, f"chan{i}_raw")
            }
            if channels != self.rc_channels:
                self.rc_channels = channels
                print("RC:", channels)

    # ── Main integrated loop ──────────────────────────────────────────────────

    def run(self):
        """Open camera, display tracked feed, monitor RC ch6, and feed
        TRACKING messages to the flight controller when activated."""
        if self.master is None:
            raise RuntimeError("Not connected. Call connect() first.")

        self.seeker.open()
        frame_times: collections.deque = collections.deque(maxlen=30)
        prev_time = time.monotonic()
        try:
            while True:
                # ── 1. Grab frame & run pink CamShift tracker ─────────────────
                ok, frame = self.seeker.read_frame()
                if not ok:
                    print("[Ctrl] End of stream.")
                    break

                now = time.monotonic()
                frame_times.append(now - prev_time)
                prev_time = now
                fps = 1.0 / (sum(frame_times) / len(frame_times))

                annotated, cx, cy = self.seeker.track(frame)
                errorx, errory    = self.seeker.error_xy(cx, cy, frame.shape)
                target_locked     = errorx is not None

                # ── 2. Poll RC & HEARTBEAT (non-blocking) ─────────────────────
                if self._joy_handler is not None:
                    self._joy_handler.pump()
                self._poll_rc()
                self._poll_heartbeat()
                ch6_on = self._ch6_active()

                # ── 3. Mode management ────────────────────────────────────────
                ch6_fell = self._prev_ch6_on and not ch6_on  # armed → disarmed edge

                # Rule 5: on falling edge of ch6 → RTL
                if ch6_fell:
                    self._in_tracking = False
                    self.set_mode_rtl()

                elif ch6_on:
                    # Rule 3: detected + armed → enter tracking (once)
                    if target_locked and not self._in_tracking:
                        self.set_mode_tracking()
                        self._in_tracking = True
                    # Rules 1,2,4: no change to tracking flag based on detection

                self._prev_ch6_on = ch6_on

                # ── 4. Feed TRACKING error while active ───────────────────────
                if self._in_tracking and target_locked:
                    if self._flight_mode == "TRACKING":
                        print(f"[SEND] errorx={errorx:+.3f}  errory={errory:+.3f}")
                    self.send_tracking(errorx, errory)

                # ── 5. Annotate HUD ───────────────────────────────────────────
                status  = "ON" if self._in_tracking else (
                    "OFF" if not ch6_on else "NO TARGET"
                )

                h_frame = annotated.shape[0]
                err_str = f"ex={errorx:+.3f}  ey={errory:+.3f}" if target_locked else ""
                cv2.putText(annotated,
                            f"MODE: {self._flight_mode}  LOCK: {status}",
                            (10, h_frame - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                cv2.putText(annotated,
                            f"FPS: {fps:.1f}  {err_str}",
                            (10, h_frame - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                cv2.imshow(self.seeker.window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[Ctrl] Quit.")
                    break
                elif key == ord("r"):
                    self.seeker._track_win    = None
                    self.seeker._detect_count = 0
                    print("[Ctrl] Tracker reset.")

        finally:
            self.seeker.close()
            if self._joystick_enabled:
                self._stop_joystick_thread()
