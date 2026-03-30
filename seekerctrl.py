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

# ── Fixed target position ─────────────────────────────────────────────────────
# WGS-84 coordinates of the hot-pink ground target.
# Edit these before each flight to match the actual target placement.
TARGET_LAT     =  -6.897344909   # decimal degrees  (+N / -S)
TARGET_LON     = 107.566439439   # decimal degrees  (+E / -W)
TARGET_ALT_MSL = 744.1      # metres above mean sea level


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
        self._srv1_raw   = 0          # SERVO_OUTPUT_RAW servo1 (µs)
        self._srv2_raw   = 0          # SERVO_OUTPUT_RAW servo2 (µs)
        self._pitch_deg  = 0.0        # ATTITUDE pitch (deg)
        self._rel_alt_m  = 0.0        # GLOBAL_POSITION_INT relative_alt (m, AGL)
        self._alt_msl_m  = 0.0        # GLOBAL_POSITION_INT alt (m, MSL)
        self._lat        = 0          # GLOBAL_POSITION_INT lat (1e7 deg)
        self._lon        = 0          # GLOBAL_POSITION_INT lon (1e7 deg)

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
        # MAVLink message IDs we want
        streams = [
            (30,  25),   # ATTITUDE            @ 25 Hz
            (36,  25),   # SERVO_OUTPUT_RAW    @ 25 Hz
            (33,   5),   # GLOBAL_POSITION_INT @  5 Hz
        ]
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
            self._pitch_deg = math.degrees(msg.pitch)

        msg = self.master.messages.get("GLOBAL_POSITION_INT")
        if msg:
            self._rel_alt_m = msg.relative_alt * 1e-3   # mm → m (AGL)
            self._alt_msl_m = msg.alt          * 1e-3   # mm → m (MSL)
            self._lat = msg.lat
            self._lon = msg.lon
            #print(f"[GPS] lat={self._lat*1e-7:.7f}  lon={self._lon*1e-7:.7f}"
            #      f"  alt_msl={self._alt_msl_m:.1f}m  alt_agl={self._rel_alt_m:.1f}m")

    # ── Distance / altitude relative to target ────────────────────────────────

    def _dist_to_target_m(self) -> float | None:
        """Horizontal haversine distance (m) from current position to TARGET."""
        if self._lat == 0:
            return None
        lat1 = math.radians(self._lat  * 1e-7)
        lon1 = math.radians(self._lon  * 1e-7)
        lat2 = math.radians(TARGET_LAT)
        lon2 = math.radians(TARGET_LON)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371000.0 * 2 * math.asin(math.sqrt(a))

    def _alt_rel_to_target_m(self) -> float:
        """Altitude of the aircraft above the target (m).  Positive = above target."""
        return self._alt_msl_m - TARGET_ALT_MSL

    def _geo_pitch_deg(self) -> float | None:
        """Geometric pitch angle (deg) required to point the nose at the target.

        Uses horizontal distance and altitude above target:
            pitch = atan2(-alt_above_target, dist_to_target)

        Negative = nose down (aircraft is above target and must dive).
        Returns None when position data is not yet available.
        """
        dist = self._dist_to_target_m()
        if dist is None:
            return None
        return math.degrees(math.atan2(-self._alt_rel_to_target_m(), max(dist, 1.0)))

    # ── CSV logger ────────────────────────────────────────────────────────────

    def _open_csv(self):
        self._csv_file   = open("tracking.csv", "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp_s", "errorx", "errory",
            "aileron", "elevator",
            "pitch_deg", "alt_above_target_m", "dist_to_target_m", "geo_pitch_deg",
        ])
        print("[LOG] tracking.csv opened")

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
            print("[LOG] tracking.csv closed")

    def _log_row(self, timestamp: float, errorx, errory, geo_pitch):
        if self._csv_writer is None:
            return
        # Elevon demix: srv1/srv2 centred at 1500 µs
        aileron  = (self._srv1_raw - self._srv2_raw)/700
        elevator = (self._srv1_raw + self._srv2_raw-2700)/700
        dist = self._dist_to_target_m()
        self._csv_writer.writerow([
            f"{timestamp:.3f}",
            f"{errorx:+.4f}"    if errorx    is not None else "",
            f"{errory:+.4f}"    if errory    is not None else "",
            f"{aileron:.1f}",
            f"{elevator:.1f}",
            f"{self._pitch_deg:.2f}",
            f"{self._alt_rel_to_target_m():.2f}",
            f"{dist:.1f}"       if dist      is not None else "",
            f"{geo_pitch:.2f}"  if geo_pitch is not None else "",
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
                geo_pitch = self._geo_pitch_deg()

                if self._in_tracking and target_locked:
                    if self._flight_mode == "TRACKING":
                        self.send_tracking(errorx, errory)
                        self._log_row(now, errorx, errory, geo_pitch)

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
