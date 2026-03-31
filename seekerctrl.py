import collections
import csv
import math
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

# ── Tracking control tuning ───────────────────────────────────────────────────
_TRK_MAX_DEG = 30.0    # must match ArduPlane TRK_MAX_DEG
_LATENCY_S   = 0.08    # pipeline latency to compensate (s)

# ArduPlane custom mode numbers
_TRACKING_MODE = 27
_LOITER_MODE   = 5   # fallback mode when ch6 goes low
_AUTO_MODE     = 10  # AUTO when lock lost / ch6 disarmed

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
        debug_log: bool = False,
    ):
        self.connection_string = connection_string
        self.baud   = baud
        self.master = None

        self.rc_channels    = {}
        self._in_tracking   = False   # True while flight mode is TRACKING
        self._flight_mode   = "?"     # last known flight mode name from HEARTBEAT
        self._prev_ch6_on   = False   # previous ch6 armed state for edge detection

        # ── MAVLink telemetry state ───────────────────────────────────────────
        self._srv1_raw      = 0        # SERVO_OUTPUT_RAW servo1 (µs)
        self._srv2_raw      = 0        # SERVO_OUTPUT_RAW servo2 (µs)
        self._roll_deg      = 0.0      # ATTITUDE roll (deg)
        self._pitch_deg     = 0.0      # ATTITUDE pitch (deg)
        self._roll_rate_dps = 0.0      # ATTITUDE rollspeed (deg/s)
        self._pitch_rate_dps= 0.0      # ATTITUDE pitchspeed (deg/s)
        # PID_TUNING — ArduPlane actual PID term outputs (axis 1=roll, 2=pitch)
        self._pid_roll_P    = 0.0      # roll  P term (cd)
        self._pid_roll_I    = 0.0      # roll  I term (cd)
        self._pid_roll_D    = 0.0      # roll  D term (cd)
        self._pid_roll_des  = 0.0      # roll  desired (deg)
        self._pid_pitch_P   = 0.0      # pitch P term (cd)
        self._pid_pitch_I   = 0.0      # pitch I term (cd)
        self._pid_pitch_D   = 0.0      # pitch D term (cd)
        self._pid_pitch_des = 0.0      # pitch desired (deg)
        self._rel_alt_m  = 0.0        # GLOBAL_POSITION_INT relative_alt (m, AGL)

        # ── Lock-loss hold state ──────────────────────────────────────────────
        self._last_errorx = 0.0   # last valid errorx (re-sent while lock is lost)
        self._last_errory = 0.0   # last valid errory (re-sent while lock is lost)

        # ── Latency prediction state ──────────────────────────────────────────
        self._prev_errorx_v  = 0.0   # raw errorx from previous frame
        self._prev_errory_v  = 0.0   # raw errory from previous frame
        self._prev_err_t     = 0.0   # monotonic time of previous error sample

        # ── CSV logger ────────────────────────────────────────────────────────
        self._debug_log  = debug_log
        self._csv_file   = None
        self._csv_writer = None

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
        self._request_data_streams()
        if self._joystick_enabled:
            self._start_joystick_thread()

    # ── Stream requests ───────────────────────────────────────────────────────

    def _request_data_streams(self):
        """Ask ArduPlane to stream the telemetry messages we log."""
        streams = [
            (30,  25),   # ATTITUDE            @ 25 Hz
            (36,  25),   # SERVO_OUTPUT_RAW    @ 25 Hz
            (33,   5),   # GLOBAL_POSITION_INT @  5 Hz
        ]
        if self._debug_log:
            streams.append((98, 25))   # PID_TUNING @ 25 Hz — debug only
        for msg_id, rate_hz in streams:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id,
                int(1e6 / rate_hz),   # interval in µs
                0, 0, 0, 0, 0,
            )

    # ── MAVLink telemetry poll ────────────────────────────────────────────────

    def _poll_mavlink_state(self):
        """Drain the pymavlink message cache for telemetry fields we log."""
        msg = self.master.messages.get("SERVO_OUTPUT_RAW")
        if msg:
            self._srv1_raw = msg.servo1_raw
            self._srv2_raw = msg.servo2_raw

        msg = self.master.messages.get("ATTITUDE")
        if msg:
            self._roll_deg       = math.degrees(msg.roll)
            self._pitch_deg      = math.degrees(msg.pitch)
            self._roll_rate_dps  = math.degrees(msg.rollspeed)
            self._pitch_rate_dps = math.degrees(msg.pitchspeed)

        # PID_TUNING is only streamed in debug mode; skip the queue drain otherwise.
        if self._debug_log:
            while True:
                pid_msg = self.master.recv_match(type="PID_TUNING", blocking=False)
                if pid_msg is None:
                    break
                if pid_msg.axis == 1:   # roll
                    self._pid_roll_P   = pid_msg.P
                    self._pid_roll_I   = pid_msg.I
                    self._pid_roll_D   = pid_msg.D
                    self._pid_roll_des = pid_msg.desired
                elif pid_msg.axis == 2: # pitch
                    self._pid_pitch_P   = pid_msg.P
                    self._pid_pitch_I   = pid_msg.I
                    self._pid_pitch_D   = pid_msg.D
                    self._pid_pitch_des = pid_msg.desired

        msg = self.master.messages.get("GLOBAL_POSITION_INT")
        if msg:
            self._rel_alt_m = msg.relative_alt * 1e-3   # mm → m (AGL)

    # ── CSV logger ────────────────────────────────────────────────────────────

    def _open_csv(self):
        self._csv_file   = open("tracking.csv", "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp_s", "errorx", "errory",
            "aileron", "elevator",
            "roll_deg", "pitch_deg", "roll_rate_dps", "pitch_rate_dps",
            "pid_roll_desired", "pid_roll_P", "pid_roll_I", "pid_roll_D",
            "pid_pitch_desired", "pid_pitch_P", "pid_pitch_I", "pid_pitch_D",
            "alt_agl_m",
        ])
        print("[LOG] tracking.csv opened")

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
            print("[LOG] tracking.csv closed")

    def _log_row(self, timestamp: float, errorx, errory):
        if self._csv_writer is None:
            return
        # Elevon demix: srv1/srv2 centred at 1500 µs
        aileron  = (self._srv1_raw - self._srv2_raw) / 700
        elevator = (self._srv1_raw + self._srv2_raw - 2700) / 700
        self._csv_writer.writerow([
            f"{timestamp:.3f}",
            f"{errorx:+.4f}" if errorx is not None else "",
            f"{errory:+.4f}" if errory is not None else "",
            f"{aileron:.4f}",
            f"{elevator:.4f}",
            f"{self._roll_deg:.3f}",
            f"{self._pitch_deg:.3f}",
            f"{self._roll_rate_dps:.3f}",
            f"{self._pitch_rate_dps:.3f}",
            f"{self._pid_roll_des:.4f}",
            f"{self._pid_roll_P:.4f}",
            f"{self._pid_roll_I:.4f}",
            f"{self._pid_roll_D:.4f}",
            f"{self._pid_pitch_des:.4f}",
            f"{self._pid_pitch_P:.4f}",
            f"{self._pid_pitch_I:.4f}",
            f"{self._pid_pitch_D:.4f}",
            f"{self._rel_alt_m:.2f}",
        ])

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

    def set_mode_auto(self):
        self._set_mode(_AUTO_MODE, "AUTO")

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
        prev_time       = time.monotonic()
        prev_in_tracking = False
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

                # ── 2. Poll RC, HEARTBEAT & telemetry (non-blocking) ──────────
                if self._joy_handler is not None:
                    self._joy_handler.pump()
                self._poll_rc()
                self._poll_heartbeat()
                self._poll_mavlink_state()
                ch6_on = self._ch6_active()

                # ── 3. Mode management ────────────────────────────────────────
                ch6_fell = self._prev_ch6_on and not ch6_on  # armed → disarmed edge

                # Rule 5: on falling edge of ch6 → AUTO
                if ch6_fell:
                    self._in_tracking = False
                    self.set_mode_auto()

                elif ch6_on:
                    # Rule 3: detected + armed → enter tracking (once)
                    if target_locked and not self._in_tracking:
                        self.set_mode_tracking()
                        self._in_tracking = True
                    # Rules 1,2,4: no change to tracking flag based on detection

                self._prev_ch6_on = ch6_on

                # ── 4. CSV lifecycle ──────────────────────────────────────────
                if self._debug_log:
                    if self._in_tracking and not prev_in_tracking:
                        self._open_csv()
                    elif not self._in_tracking and prev_in_tracking:
                        self._close_csv()
                prev_in_tracking = self._in_tracking

                # ── 5. Feed TRACKING error while active ───────────────────────
                if self._in_tracking and self._flight_mode == "TRACKING":
                    if target_locked:
                        # Latency prediction: shift errorx/errory forward by the
                        # estimated pipeline delay to reduce lag-induced overshoot.
                        raw_ex, raw_ey = errorx, errory
                        dt_err = now - self._prev_err_t if self._prev_err_t > 0.0 else 0.0
                        if 0.0 < dt_err < 0.5:
                            dx_dt  = (raw_ex - self._prev_errorx_v) / dt_err
                            dy_dt  = (raw_ey - self._prev_errory_v) / dt_err
                            errorx = max(-1.0, min(1.0, raw_ex + dx_dt * _LATENCY_S))
                            errory = max(-1.0, min(1.0, raw_ey + dy_dt * _LATENCY_S))
                        self._prev_errorx_v = raw_ex
                        self._prev_errory_v = raw_ey
                        self._prev_err_t    = now

                        self._last_errorx = errorx
                        self._last_errory = errory
                        self.send_tracking(errorx, errory)
                        self._log_row(now, errorx, errory)

                    else:
                        # Lock lost: zero errorx so the aircraft levels its wings
                        # and flies straight.  Keep last errory so pitch/dive toward
                        # the target altitude is maintained.  ArduPlane never times
                        # out, so tracking resumes immediately when lock returns.
                        self.send_tracking(0.0, self._last_errory)
                        self._log_row(now, 0.0, self._last_errory)

                # ── 6. Annotate HUD ───────────────────────────────────────────
                status  = "ON" if self._in_tracking else (
                    "OFF" if not ch6_on else "NO TARGET"
                )

                h_frame = annotated.shape[0]
                err_str = f"ex={errorx:+.3f}  ey={errory:+.3f}" if target_locked else ""
                cv2.putText(annotated,
                            f"MODE: {self._flight_mode}  LOCK: {status}",
                            (10, h_frame - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
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
            self._close_csv()
            self.seeker.close()
            if self._joystick_enabled:
                self._stop_joystick_thread()
