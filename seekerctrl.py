import collections
import csv
import math
import threading
import time

import cv2
from pymavlink import mavutil

from hud_display import HudDisplay
from joystick_handler import JoystickHandler
from seeker import Seeker

# Tracking errors are sent as DEBUG_VECT (MAVLink ID 250, standard message).
# x = errorx, y = errory, z unused.
# ArduPlane decodes them in handle_message (case MAVLINK_MSG_ID_DEBUG_VECT).

# RC channel 6 PWM threshold to consider the switch "active"
_CH6_ACTIVE_PWM = 1400

# ── Tracking control tuning ───────────────────────────────────────────────────
_LATENCY_S     = 0.08   # pipeline latency to compensate (s)
_PN_LEAD_S     = 0.30   # proportional navigation lead time (s) — increase for faster aircraft
_TRK_TERM_ALT        = 0.0         # must match ArduPlane TRK_TERM_ALT (m above target MSL); 0 = disabled
_TRK_PITCH_OFFSET    = 3.0         # must match ArduPlane TRK_PITCH_OFFSET (deg) — cruise nose-up bias
_TRK_TERM_PTCH       = 0.0         # must match ArduPlane TRK_TERM_PTCH (deg)   — extra nose-down in terminal
_TRK_MAX_DEG         = 30.0        # must match ArduPlane TRK_MAX_DEG — full-scale error angle (deg)
_TRK_TARGET_ALT_MSL  = 744.0            # must match ArduPlane TRK_TGT_ALT  (m MSL)
_TRK_TARGET_LAT      = -6.897367724    # must match ArduPlane TRK_TGT_LAT  (decimal deg)
_TRK_TARGET_LON      = 107.566559898   # must match ArduPlane TRK_TGT_LON  (decimal deg)

# ArduPlane custom mode numbers
_TRACKING_MODE   = 27
_LOITER_MODE     = 5   # fallback mode when ch6 goes low
_AUTO_MODE       = 10  # AUTO when lock lost / ch6 disarmed
_STABILIZE_MODE  = 2   # STABILIZE — auto mode ch6-low fallback

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
        record: bool = False,
        input_prediction: bool = True,
        mask_algo: str = "all",
        use_camshift: bool = True,
        box_filter: bool = True,
        hud_pitch: bool = True,
        hud_yaw: bool = True,
        auto_mode: bool = False,
    ):
        self._input_prediction = input_prediction
        self._auto_mode        = auto_mode
        self.connection_string = connection_string
        self.baud   = baud
        self.master = None

        self.rc_channels    = {}
        self._in_tracking    = False   # True while flight mode is TRACKING
        self._flight_mode    = "?"     # last known flight mode name from HEARTBEAT
        self._prev_ch6_on    = False   # previous ch6 armed state for edge detection
        self._commanded_mode = -1      # last custom_mode we sent a command for

        # ── MAVLink telemetry state ───────────────────────────────────────────
        self._srv1_raw      = 0        # SERVO_OUTPUT_RAW servo1 (µs)
        self._srv2_raw      = 0        # SERVO_OUTPUT_RAW servo2 (µs)
        self._roll_deg       = 0.0      # ATTITUDE roll (deg)
        self._pitch_deg      = 0.0      # ATTITUDE pitch (deg)
        self._nav_pitch_deg  = 0.0      # NAV_CONTROLLER_OUTPUT nav_pitch (deg)
        self._yaw_deg        = 0.0      # ATTITUDE yaw (deg)
        self._roll_rate_dps  = 0.0      # ATTITUDE rollspeed (deg/s)
        self._pitch_rate_dps = 0.0      # ATTITUDE pitchspeed (deg/s)
        self._lat            = 0.0      # GLOBAL_POSITION_INT lat (deg)
        self._lon            = 0.0      # GLOBAL_POSITION_INT lon (deg)
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
        self._alt_msl_m  = 0.0        # GLOBAL_POSITION_INT alt (m, MSL)
        self._airspeed_ms = 0.0       # VFR_HUD airspeed (m/s)
        self._throttle_pct = 0        # VFR_HUD throttle (0-100 %)
        self._home_lat   = None       # HOME_POSITION lat (deg)
        self._home_lon   = None       # HOME_POSITION lon (deg)

        # ── Target position (fetched from ArduPlane params on connect) ─────────
        self._target_alt_msl = _TRK_TARGET_ALT_MSL   # TRK_TGT_ALT (m MSL)
        self._target_lat     = _TRK_TARGET_LAT        # TRK_TGT_LAT (decimal deg)
        self._target_lon     = _TRK_TARGET_LON        # TRK_TGT_LON (decimal deg)
        self._term_alt       = _TRK_TERM_ALT          # TRK_TERM_ALT (m above target MSL)

        # ── Mission state ─────────────────────────────────────────────────────
        self._waypoint_count = 0    # total mission items (from MISSION_COUNT)
        self._current_wp     = 0    # current waypoint seq (from MISSION_CURRENT)

        # ── Lock-loss hold state ──────────────────────────────────────────────
        self._last_errorx        = 0.0   # last valid errorx (re-sent while lock is lost)
        self._last_errory        = 0.0   # last valid errory (re-sent while lock is lost)
        self._lost_count         = 0     # consecutive target_locked=False frames (after seeker prediction)
        self._tracking_entry_count = 0   # how many times TRACKING mode has been entered

        # ── Latency prediction state ──────────────────────────────────────────
        self._prev_errorx_v  = 0.0   # raw errorx from previous frame
        self._prev_errory_v  = 0.0   # raw errory from previous frame
        self._prev_err_t     = 0.0   # monotonic time of previous error sample

        # ── CSV logger ────────────────────────────────────────────────────────
        self._debug_log  = debug_log
        self._csv_file   = None
        self._csv_writer = None

        # ── Video recorder ────────────────────────────────────────────────────
        self._record     = record
        self._vwriter    = None   # cv2.VideoWriter, open only during TRACKING

        self._hud = HudDisplay(show_pitch=hud_pitch, show_yaw=hud_yaw)

        self.seeker = Seeker(source=source,
                             capture_width=capture_width,
                             capture_height=capture_height,
                             crop=crop,
                             show_histogram=show_histogram,
                             show_mask=show_mask,
                             mask_algo=mask_algo,
                             use_camshift=use_camshift,
                             box_filter=box_filter)

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
        self._fetch_tracking_params()
        self._fetch_mission_count()
        if self._joystick_enabled:
            self._start_joystick_thread()

    # ── Parameter fetch ───────────────────────────────────────────────────────

    _TRACKED_PARAMS = {
        "TRK_TGT_ALT":  "_target_alt_msl",
        "TRK_TGT_LAT":  "_target_lat",
        "TRK_TGT_LON":  "_target_lon",
        "TRK_TERM_ALT": "_term_alt",
    }

    def _fetch_tracking_params(self):
        """Request TRK_TGT_ALT/LAT/LON from ArduPlane and wait for replies."""
        for name in self._TRACKED_PARAMS:
            self.master.mav.param_request_read_send(
                self.master.target_system,
                self.master.target_component,
                name.encode(),
                -1,   # -1 = lookup by name
            )
        # Collect replies with a short timeout; fall back to defaults if absent.
        remaining = set(self._TRACKED_PARAMS)
        deadline  = time.monotonic() + 2.0
        while remaining and time.monotonic() < deadline:
            msg = self.master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.2)
            if msg is None:
                continue
            pname = msg.param_id.rstrip("\x00")
            if pname in remaining:
                setattr(self, self._TRACKED_PARAMS[pname], msg.param_value)
                remaining.discard(pname)
                print(f"[Param] {pname} = {msg.param_value}")
        for pname in remaining:
            print(f"[Param] {pname} not received — using default")

    def _fetch_mission_count(self):
        """Request mission item count, reset current WP to 0, store count."""
        self.master.mav.mission_request_list_send(
            self.master.target_system,
            self.master.target_component,
        )
        msg = self.master.recv_match(type="MISSION_COUNT", blocking=True, timeout=3.0)
        if msg is not None:
            self._waypoint_count = msg.count
            print(f"[Mission] {msg.count} waypoints")
        else:
            print("[Mission] MISSION_COUNT not received — waypoint check disabled")
        # Set the active waypoint to 0 on the FC.
        self.master.mav.mission_set_current_send(
            self.master.target_system,
            self.master.target_component,
            0,
        )
        self._current_wp = 0
        print("[Mission] Current WP set to 0")

    # ── Stream requests ───────────────────────────────────────────────────────

    def _request_data_streams(self):
        """Ask ArduPlane to stream the telemetry messages we log."""
        streams = [
            (30,  25),   # ATTITUDE              @ 25 Hz
            (36,  25),   # SERVO_OUTPUT_RAW      @ 25 Hz
            (33,   5),   # GLOBAL_POSITION_INT   @  5 Hz
            (74,  10),   # VFR_HUD               @ 10 Hz
            (62,  25),   # NAV_CONTROLLER_OUTPUT @ 25 Hz
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
            self._yaw_deg        = math.degrees(msg.yaw) % 360
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
            self._alt_msl_m = msg.alt * 1e-3             # mm → m (MSL)
            self._lat       = msg.lat * 1e-7             # degE7 → deg
            self._lon       = msg.lon * 1e-7             # degE7 → deg

        msg = self.master.messages.get("VFR_HUD")
        if msg:
            self._airspeed_ms  = msg.airspeed
            self._throttle_pct = msg.throttle

        msg = self.master.messages.get("HOME_POSITION")
        if msg:
            self._home_lat = msg.latitude  * 1e-7   # degE7 → deg
            self._home_lon = msg.longitude * 1e-7

        msg = self.master.messages.get("MISSION_CURRENT")
        if msg:
            self._current_wp = msg.seq

        msg = self.master.messages.get("NAV_CONTROLLER_OUTPUT")
        if msg:
            self._nav_pitch_deg = msg.nav_pitch

    def _dist_to_home_m(self) -> float:
        """Haversine distance (m) from current GPS position to home."""
        if self._home_lat is None or not self._lat:
            return 0.0
        R = 6371000.0
        lat1 = math.radians(self._lat)
        lat2 = math.radians(self._home_lat)
        dlat = lat2 - lat1
        dlon = math.radians(self._home_lon - self._lon)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return R * 2.0 * math.asin(math.sqrt(a))

    def _dist_to_target_m(self) -> float:
        """Haversine distance (m) from current GPS position to target."""
        if not self._lat:
            return float("inf")
        R = 6371000.0
        lat1 = math.radians(self._lat)
        lat2 = math.radians(self._target_lat)
        dlat = lat2 - lat1
        dlon = math.radians(self._target_lon - self._lon)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return R * 2.0 * math.asin(math.sqrt(a))

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
            "alt_rel_m", "airspeed_ms", "throttle_pct",
            "nav_pitch_deg",
            "target_locked", "terminal",
            "dist_m",
        ])
        print("[LOG] tracking.csv opened")

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
            print("[LOG] tracking.csv closed")

    def _log_row(self, timestamp: float, errorx, errory,
                 target_locked: bool = True, terminal: bool = False):
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
            f"{self._alt_msl_m - self._target_alt_msl:.2f}",
            f"{self._airspeed_ms:.2f}",
            f"{self._throttle_pct}",
            f"{self._nav_pitch_deg:.3f}",
            "1" if target_locked else "0",
            "1" if terminal else "0",
            f"{math.sqrt(self._dist_to_target_m()**2 + (self._alt_msl_m - self._target_alt_msl)**2):.1f}",
        ])

    # ── Video recorder ────────────────────────────────────────────────────────

    def _open_video(self, frame_shape):
        import datetime
        h, w = frame_shape[:2]
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"tracking_{ts}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._vwriter = cv2.VideoWriter(path, fourcc, 25.0, (w, h))
        print(f"[REC] recording → {path}")

    def _close_video(self):
        if self._vwriter is not None:
            self._vwriter.release()
            self._vwriter = None
            print("[REC] recording stopped")

    def _write_frame(self, frame):
        if self._vwriter is not None:
            self._vwriter.write(frame)

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
        if self._commanded_mode == custom_mode:
            return
        self._commanded_mode = custom_mode
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            custom_mode,
            0, 0, 0, 0, 0,
        )

    def set_mode_tracking(self):
        self._set_mode(_TRACKING_MODE, "TRACKING")

    def set_mode_loiter(self):
        self._set_mode(_LOITER_MODE, "LOITER")

    def set_mode_auto(self):
        self._set_mode(_AUTO_MODE, "AUTO")

    def set_mode_stabilize(self):
        self._set_mode(_STABILIZE_MODE, "STABILIZE")

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
                if frame is None:
                    continue   # capture thread not ready yet
                if not ok:
                    print("[Ctrl] End of stream.")
                    break

                now = time.monotonic()
                frame_times.append(now - prev_time)
                prev_time = now
                fps = 1.0 / (sum(frame_times) / len(frame_times))

                annotated, cx, cy = self.seeker.track(frame)
                target_locked     = cx is not None and cy is not None
                if target_locked:
                    errorx, errory = self.seeker.error_xy(cx, cy, frame.shape)
                else:
                    errorx, errory = 0.0, 0.0

                # ── 2. Poll RC, HEARTBEAT & telemetry (non-blocking) ──────────
                if self._joy_handler is not None:
                    self._joy_handler.pump()
                self._poll_rc()
                self._poll_heartbeat()
                self._poll_mavlink_state()
                ch6_on = self._ch6_active()

                # ── 3. Mode management ────────────────────────────────────────
                ch6_fell = self._prev_ch6_on and not ch6_on  # armed → disarmed edge

                if self._auto_mode:
                    dist_to_target_m = self._dist_to_target_m()
                    close_enough     = dist_to_target_m < 700.0
                    on_last_wp       = (self._waypoint_count > 0 and
                                        self._current_wp == self._waypoint_count - 1)

                    # ch6 high → AUTO (or TRACKING when lock conditions are met)
                    # ch6 low  → STABILIZE
                    if ch6_fell:
                        self._in_tracking = False
                        self.set_mode_stabilize()

                    elif ch6_on:
                        if close_enough and on_last_wp and target_locked and not self._in_tracking:
                            # Within 700 m of target AND on final waypoint AND camera locked → enter TRACKING
                            self.set_mode_tracking()
                            self._in_tracking = True
                            self._tracking_entry_count += 1
                        elif not (close_enough and on_last_wp) and not self._in_tracking:
                            # Not yet in range or not on last waypoint → follow AUTO mission
                            # Guard: only send AUTO when not already in tracking, to avoid
                            # pulling out of TRACKING if distance briefly crosses 700 m.
                            self.set_mode_auto()

                    else:
                        # ch6 is low and no edge — hold STABILIZE
                        self.set_mode_stabilize()

                else:
                    dist_to_target_m = None

                    # Rule 5: on falling edge of ch6 → AUTO
                    if ch6_fell:
                        self._in_tracking = False
                        self.set_mode_auto()

                    elif ch6_on:
                        # Rule 3: detected + armed → enter tracking (once)
                        if target_locked and not self._in_tracking:
                            self.set_mode_tracking()
                            self._in_tracking = True
                            self._tracking_entry_count += 1
                        # Rules 1,2,4: no change to tracking flag based on detection

                self._prev_ch6_on = ch6_on

                # ── 4. CSV / video lifecycle ──────────────────────────────────
                if self._in_tracking and not prev_in_tracking:
                    if self._debug_log:
                        self._open_csv()
                    if self._record:
                        self._open_video(frame.shape)
                elif not self._in_tracking and prev_in_tracking:
                    # Tracking just stopped — send a final zero error so
                    # ArduPlane does not act on any stale error still in flight.
                    self.send_tracking(0.0, 0.0)
                    if self._debug_log:
                        self._close_csv()
                    if self._record:
                        self._close_video()
                prev_in_tracking = self._in_tracking

                # ── 5. Feed TRACKING error while active ───────────────────────
                alt_dist_m  = self._alt_msl_m - self._target_alt_msl
                in_terminal = (self._term_alt > 0.0 and alt_dist_m <= self._term_alt)

                if self._in_tracking and self._flight_mode == "TRACKING":
                    if target_locked:
                        # Latency + PN lead prediction: shift errorx/errory forward
                        # to reduce lag-induced overshoot. Disabled by _INPUT_PREDICTION=False.
                        raw_ex, raw_ey = errorx, errory
                        if self._input_prediction:
                            dt_err = now - self._prev_err_t if self._prev_err_t > 0.0 else 0.0
                            if 0.0 < dt_err < 0.5:
                                dx_dt  = (raw_ex - self._prev_errorx_v) / dt_err
                                dy_dt  = (raw_ey - self._prev_errory_v) / dt_err
                                lead   = _LATENCY_S + _PN_LEAD_S
                                errorx = max(-1.0, min(1.0, raw_ex + dx_dt * lead))
                                errory = max(-1.0, min(1.0, raw_ey + dy_dt * lead))
                        self._prev_errorx_v = raw_ex
                        self._prev_errory_v = raw_ey
                        self._prev_err_t    = now

                        self._last_errorx = errorx
                        self._last_errory = errory
                        self._lost_count  = 0
                        self.send_tracking(errorx, errory)
                        # Log effective errory: subtract pitch offsets so the CSV
                        # reflects the actual PID setpoint ArduPlane is driving to.
                        offset_norm = _TRK_PITCH_OFFSET / _TRK_MAX_DEG
                        if in_terminal:
                            offset_norm += _TRK_TERM_PTCH / _TRK_MAX_DEG
                        log_ey = errory - offset_norm
                        self._log_row(now, errorx, log_ey,
                                      target_locked=True, terminal=in_terminal)

                    else:
                        self._lost_count += 1
                        committed = self._tracking_entry_count >= 3
                        limit     = 30 if committed else 10
                        if self._lost_count >= limit:
                            # Loss limit reached — return to AUTO.
                            self._lost_count  = 0
                            self._in_tracking = False
                            self.set_mode_auto()
                        elif committed:
                            # 3+ tracking entries: keep sending last known errors
                            # so ArduPlane stays pointed at the last known direction.
                            self.send_tracking(self._last_errorx, self._last_errory)
                        else:
                            # < 3 tracking entries: level off while waiting.
                            self.send_tracking(0.0, 0.0)
                        self._log_row(now, self._last_errorx if committed else 0.0,
                                      self._last_errory if committed else 0.0,
                                      target_locked=False, terminal=in_terminal)

                # ── 6. Annotate HUD ───────────────────────────────────────────
                h_frame  = annotated.shape[0]
                dist_tgt_m = dist_to_target_m if dist_to_target_m is not None else self._dist_to_target_m()
                spd_kmh  = self._airspeed_ms * 3.6
                err_str  = f"  ex={errorx:+.3f} ey={errory:+.3f}" if target_locked else ""
                mode_wp  = "%s:%d" % (self._flight_mode, self._current_wp)
                desc = "%s, dist %.3f km, alt %+.0f m, v %.2f km/jam, throttle %.0f%%." % (
                    mode_wp, dist_tgt_m / 1000.0,
                    alt_dist_m, spd_kmh, self._throttle_pct,
                )
                cv2.putText(annotated,
                            f"FPS: {fps:.1f}  LOCK: {'ON' if ch6_on else 'OFF'}{err_str}",
                            (5, h_frame - 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(annotated,
                            desc,
                            (5, h_frame - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2)

                self._hud.draw_hud(True, annotated,
                                   self._lat, self._lon,
                                   self._yaw_deg, self._pitch_deg, self._roll_deg)

                if self._record and self._in_tracking:
                    self._write_frame(annotated)
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
            self._close_video()
            self.seeker.close()
            if self._joystick_enabled:
                self._stop_joystick_thread()
