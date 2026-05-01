import collections
import csv
import math
import subprocess
import threading
import time

import cv2
from pymavlink import mavutil

from hud_display import HudDisplay
from seeker import Seeker

# Tracking errors are sent as TRACKING_MESSAGE (MAVLink ID 11045, ardupilotmega dialect).
# Fields: errorx, errory, both normalised to [-1, 1].
# ArduPlane decodes them in handle_message (case MAVLINK_MSG_ID_TRACKING_MESSAGE).

# RC channel 6 PWM threshold to consider the switch "active"
_CH6_ACTIVE_PWM = 1400
_CH6_FORCE_ACTIVE_PWM = 1700

# ── Tracking control tuning ───────────────────────────────────────────────────
_LATENCY_S     = 0.08   # pipeline latency to compensate (s)
_PN_LEAD_S     = 0.30   # proportional navigation lead time (s) — increase for faster aircraft
_TRK_PITCH_OFFSET    = 3.0         # must match ArduPlane TRK_PITCH_OFFSET (deg) — cruise nose-up bias
_TRK_MAX_DEG         = 30.0        # must match ArduPlane TRK_MAX_DEG — full-scale error angle (deg)
_TRK_CLOSE_M         = 1000.0           # enter TRACKING only within this slant distance (m)
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
        shift_algo: str = "camshift",
        box_filter: bool = True,
        use_kalman: bool = True,
        tracker: str = "",
        hud_pitch: bool = True,
        hud_yaw: bool = True,
        auto: bool = False,
    ):
        self._input_prediction = input_prediction
        self.connection_string = connection_string
        self.baud   = baud
        self.master = None

        self.rc_channels    = {}
        self._in_tracking    = False   # True while flight mode is TRACKING
        self._flight_mode    = "?"     # last known flight mode name from HEARTBEAT
        self._prev_ch6_on       = False   # previous ch6 armed state for edge detection
        self._prev_ch6_force_on = False   # previous ch6 force-active state for rising-edge detection
        self._commanded_mode = -1      # last custom_mode we sent a command for
        self._wp_takeoff_sent = False  # WP-0 reset sent only once at startup
        self._armed           = False  # True when FC reports MAV_MODE_FLAG_SAFETY_ARMED

        # ── MAVLink telemetry state ───────────────────────────────────────────
        self._srv1_raw      = 0        # SERVO_OUTPUT_RAW servo1 (µs)
        self._srv2_raw      = 0        # SERVO_OUTPUT_RAW servo2 (µs)
        self._srv1_trim     = 1500.0   # SERVO1_TRIM (µs)
        self._srv1_min      = 1000.0   # SERVO1_MIN  (µs)
        self._srv1_max      = 2000.0   # SERVO1_MAX  (µs)
        self._srv2_trim     = 1500.0   # SERVO2_TRIM (µs)
        self._srv2_max      = 2000.0   # SERVO2_MAX  (µs)
        self._srv4_raw      = 0        # SERVO_OUTPUT_RAW servo4 — L-Rudvator (µs)
        self._srv4_trim     = 1500.0   # SERVO4_TRIM (µs)
        self._srv4_max      = 2000.0   # SERVO4_MAX  (µs)
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
        self._airspeed_ms    = 0.0     # VFR_HUD airspeed (m/s)
        self._groundspeed_ms = 0.0    # VFR_HUD groundspeed (m/s)
        self._throttle_pct = 0        # VFR_HUD throttle (0-100 %)
        self._home_lat   = None       # HOME_POSITION lat (deg)
        self._home_lon   = None       # HOME_POSITION lon (deg)

        # ── Target position (fetched from ArduPlane params on connect) ─────────
        self._target_alt_msl = _TRK_TARGET_ALT_MSL   # TRK_TGT_ALT (m MSL)
        self._target_lat     = _TRK_TARGET_LAT        # TRK_TGT_LAT (decimal deg)
        self._target_lon     = _TRK_TARGET_LON        # TRK_TGT_LON (decimal deg)
        self._pitch_offset   = _TRK_PITCH_OFFSET      # TRK_PITCH_OFFSET (deg)

        # ── Mission state ─────────────────────────────────────────────────────
        self._waypoint_count = 0    # total mission items (from MISSION_COUNT)
        self._current_wp     = 0    # current waypoint seq (from MISSION_CURRENT)
        self._prev_wp        = -1   # previous WP for transition detection
        self._prev_flight_mode  = ""    # previous flight mode for transition detection
        self._pending_video_open = False  # waiting for FPS warmup before opening video

        # ── Lock-loss hold state ──────────────────────────────────────────────
        self._last_errorx        = 0.0   # last valid errorx (re-sent while lock is lost)
        self._last_errory        = 0.0   # last valid errory (re-sent while lock is lost)
        self._lost_count         = 0     # consecutive target_locked=False frames (after seeker prediction)

        # ── Latency prediction state ──────────────────────────────────────────
        self._prev_errorx_v  = 0.0   # raw errorx from previous frame
        self._prev_errory_v  = 0.0   # raw errory from previous frame
        self._prev_err_t     = 0.0   # monotonic time of previous error sample

        # ── CSV logger ────────────────────────────────────────────────────────
        self._debug_log  = debug_log
        self._csv_file   = None
        self._csv_writer = None

        # ── Video recorder (FFmpeg pipe) ──────────────────────────────────────
        self._record      = record
        self._ffmpeg      = None   # subprocess.Popen for TRACKING phase
        self._ffmpeg_tkof = None   # subprocess.Popen for takeoff (WP 1)

        self._auto = auto
        self._hud = HudDisplay(show_pitch=hud_pitch, show_yaw=hud_yaw)

        self.seeker = Seeker(source=source,
                             capture_width=capture_width,
                             capture_height=capture_height,
                             crop=crop,
                             show_histogram=show_histogram,
                             show_mask=show_mask,
                             mask_algo=mask_algo,
                             use_camshift=use_camshift,
                             shift_algo=shift_algo,
                             box_filter=box_filter,
                             use_kalman=use_kalman,
                             tracker=tracker,
                             pitch_offset_norm=2*self._pitch_offset / _TRK_MAX_DEG)

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
        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            *([0] * 18),
        )
        print("[RC] All overrides released")
        self._set_mode(0, "MANUAL")
        self._disarm()
        self._request_data_streams()
        self._fetch_tracking_params()
        self._fetch_mission_count()

    # ── Parameter fetch ───────────────────────────────────────────────────────

    _TRACKED_PARAMS = {
        "TRK_TGT_ALT":      "_target_alt_msl",
        "TRK_TGT_LAT":      "_target_lat",
        "TRK_TGT_LON":      "_target_lon",
        "TRK_PITCH_OFFSET": "_pitch_offset",
        "SERVO1_TRIM":      "_srv1_trim",
        "SERVO1_MIN":       "_srv1_min",
        "SERVO1_MAX":       "_srv1_max",
        "SERVO2_TRIM":      "_srv2_trim",
        "SERVO2_MAX":       "_srv2_max",
        "SERVO4_TRIM":      "_srv4_trim",
        "SERVO4_MAX":       "_srv4_max",
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
        self.seeker.pitch_offset_norm = self._pitch_offset / _TRK_MAX_DEG

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
            self._srv4_raw = msg.servo4_raw

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
            self._airspeed_ms    = msg.airspeed
            self._groundspeed_ms = msg.groundspeed
            self._throttle_pct   = msg.throttle

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
        ts = getattr(self, "_tracking_ts", None) or \
             __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"tracking_{ts}.csv"
        self._csv_file   = open(fname, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp_s", "errorx", "errory",
            "aileron", "elevator", "rudder",
            "roll_deg", "pitch_deg", "roll_rate_dps", "pitch_rate_dps",
            "pid_roll_desired", "pid_roll_P", "pid_roll_I", "pid_roll_D",
            "pid_pitch_desired", "pid_pitch_P", "pid_pitch_I", "pid_pitch_D",
            "alt_rel_m", "groundspeed_ms", "throttle_pct",
            "nav_pitch_deg",
            "target_locked",
            "dist_m",
        ])
        print(f"[LOG] {fname} opened")

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
            print("[LOG] tracking.csv closed")

    def _log_row(self, timestamp: float, errorx, errory,
                 target_locked: bool = True):
        if self._csv_writer is None:
            return
        # Aileron — asymmetric ANGLE normalisation (mirrors SIM_XPlane ANGLE type):
        # use half_up = max-trim when above trim, half_dn = trim-min when below.
        srv1_half = (self._srv1_max - self._srv1_trim) if self._srv1_raw >= self._srv1_trim \
                    else (self._srv1_trim - self._srv1_min)
        aileron   = (self._srv1_raw - self._srv1_trim) / srv1_half
        # Vtail demix — mirrors SIM_XPlane VTAIL_ELEVATOR / VTAIL_RUDDER:
        # SV2 = R-rudvator (ch2), SV4 = L-rudvator (ch4)
        sum_trim  = self._srv2_trim + self._srv4_trim
        denom     = (self._srv2_max - self._srv2_trim) + (self._srv4_max - self._srv4_trim)
        elevator  = -(self._srv2_raw + self._srv4_raw - sum_trim) / denom
        rudder    =  (self._srv4_raw - self._srv2_raw) / denom
        self._csv_writer.writerow([
            f"{timestamp:.3f}",
            f"{errorx:+.4f}" if errorx is not None else "",
            f"{errory:+.4f}" if errory is not None else "",
            f"{aileron:.4f}",
            f"{elevator:.4f}",
            f"{rudder:.4f}",
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
            f"{self._groundspeed_ms:.2f}",
            f"{self._throttle_pct}",
            f"{self._nav_pitch_deg:.3f}",
            "1" if target_locked else "0",
            f"{self._dist_to_target_m():.1f}",
        ])

    # ── Video recorder ────────────────────────────────────────────────────────

    def _open_video(self, frame_shape, fps: float, label: str = "tracking"):
        import datetime
        h, w = frame_shape[:2]
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{label}_{ts}.mp4"
        cmd  = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if label == "takeoff":
            self._ffmpeg_tkof = proc
        else:
            self._ffmpeg = proc
            self._tracking_ts = ts
        print(f"[REC] {label} recording → {path}  fps={fps:.1f}")

    def _close_video(self, label: str = "tracking"):
        proc = self._ffmpeg_tkof if label == "takeoff" else self._ffmpeg
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()
        if label == "takeoff":
            self._ffmpeg_tkof = None
        else:
            self._ffmpeg = None
        print(f"[REC] {label} recording stopped")

    def _write_frame(self, frame):
        raw = frame.tobytes()
        for proc in (self._ffmpeg_tkof, self._ffmpeg):
            if proc is not None:
                try:
                    proc.stdin.write(raw)
                except BrokenPipeError:
                    pass

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
        msg = self.master.messages.get("HEARTBEAT")
        if msg:
            self._flight_mode = _PLANE_MODES.get(msg.custom_mode,
                                                  f"MODE({msg.custom_mode})")
            self._armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def _ch6_active(self) -> bool:
        if self._auto:
            return True
        pwm = self.rc_channels.get("ch6", 0)
        return pwm >= _CH6_ACTIVE_PWM and pwm < _CH6_FORCE_ACTIVE_PWM
    def _ch6_force_active(self) -> bool:
        pwm = self.rc_channels.get("ch6", 0)
        return pwm >= _CH6_FORCE_ACTIVE_PWM


    # ── Flight mode ───────────────────────────────────────────────────────────

    def _set_mode(self, custom_mode: int, label: str):
        if self._commanded_mode == custom_mode:
            return
        self._commanded_mode = custom_mode
        print(f"[MODE] → {label} ({custom_mode})")
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

    def _force_wp_takeoff(self):
        """Reset mission to WP 0 so ArduPlane executes the takeoff waypoint."""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MISSION_CURRENT,
            0,
            0,   # WP index 0 (takeoff)
            0, 0, 0, 0, 0, 0,
        )

    def set_mode_auto(self):
        if not self._wp_takeoff_sent:
            self._force_wp_takeoff()
            self._wp_takeoff_sent = True
        self._set_mode(_AUTO_MODE, "AUTO")

    def set_mode_manual(self):
        self._set_mode(0, "MANUAL")

    def set_mode_stabilize(self):
        self._set_mode(_STABILIZE_MODE, "STABILIZE")

    def _disarm(self):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )
        print("[ARM] Disarm sent")

    # ── TRACKING MAVLink message ──────────────────────────────────────────────

    def send_tracking(self, errorx: float, errory: float):
        """Send tracking errors via TRACKING_MESSAGE (ID 11045).

        errorx/errory are normalised [-1, 1].
        """
        self.master.mav.tracking_message_send(
            int(time.monotonic() * 1e6),
            errorx, errory,
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

        # Wait until AUTO mode — display camera + HUD while waiting.
        print("[Ctrl] Waiting for AUTO mode ...")
        _prev_seq_wait = -1
        while True:
            self._poll_rc()
            self._poll_heartbeat()
            self._poll_mavlink_state()
            if self._flight_mode == "AUTO":
                print("[Ctrl] AUTO mode confirmed — starting seeker loop.")
                if self._record:
                    self._pending_video_open = True
                    print("[REC] AUTO — queuing takeoff recording (waiting for FPS warmup)")
                break
            seq, ok, frame = self.seeker.read_frame()
            if frame is None or seq == _prev_seq_wait:
                time.sleep(0.002)
                continue
            _prev_seq_wait = seq
            annotated, _, _ = self.seeker.track(frame)
            self._hud.draw_hud(True, annotated,
                               self._lat, self._lon,
                               self._yaw_deg, self._pitch_deg, self._roll_deg,
                               self._pitch_offset / _TRK_MAX_DEG)
            h_w       = annotated.shape[0]
            dist_km   = self._dist_to_target_m() / 1000.0
            alt_rel_m = self._alt_msl_m - self._target_alt_msl
            spd_kmh   = self._groundspeed_ms * 3.6
            armed_str = "ARMED" if self._armed else "DISARMED"
            cv2.putText(annotated,
                        f"WAITING AUTO  mode={self._flight_mode}  SEEKER: {'ON' if self._ch6_active() or self._ch6_force_active() else 'OFF'} ",
                        (5, h_w - 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(annotated,
                        "%s:%d, dist %.3f km, alt %+.0f m, v %.2f km/jam, throttle %.0f%%." % (
                            self._flight_mode, self._current_wp,
                            dist_km, alt_rel_m, spd_kmh, self._throttle_pct),
                        (5, h_w - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2)
            cv2.imshow(self.seeker.window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
            time.sleep(0.02)

        frame_times: collections.deque = collections.deque(maxlen=30)
        prev_time        = time.monotonic()
        prev_in_tracking = False
        prev_frame_seq   = -1
        try:
            while True:
                # ── 1. Grab frame & run pink CamShift tracker ─────────────────
                frame_seq, ok, frame = self.seeker.read_frame()
                if frame is None:
                    time.sleep(0.002)
                    continue   # capture thread not ready yet
                if not ok:
                    print("[Ctrl] End of stream.")
                    break
                if frame_seq == prev_frame_seq:
                    # No new frame yet — pump MAVLink and yield to avoid
                    # spinning the main loop at CPU speed and saturating the
                    # MAVLink link with redundant send_tracking() calls.
                    self._poll_rc()
                    time.sleep(0.002)
                    continue
                prev_frame_seq = frame_seq

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
                self._poll_rc()
                self._poll_heartbeat()
                self._poll_mavlink_state()

                # ── 3. Mode management ────────────────────────────────────────
                ch6_on   = self._ch6_active() or self._ch6_force_active()
                ch6_fell = self._prev_ch6_on and not ch6_on  # armed → disarmed edge

                dist_to_target_m = self._dist_to_target_m()
                close_enough     = dist_to_target_m <= _TRK_CLOSE_M
                in_auto          = (self._flight_mode == "AUTO")
                on_last_wp       = (self._waypoint_count > 0 and
                                    self._current_wp == self._waypoint_count - 1)

                if not self.rc_channels:
                    pass  # hold MANUAL until first RC packet arrives
                elif self._ch6_active():
                    # Enter TRACKING only when in AUTO mode, on the last waypoint,
                    # and within close-enough range.
                    if (in_auto and on_last_wp and close_enough and
                            target_locked and not self._in_tracking):
                        self.set_mode_tracking()
                        self._in_tracking = True
                    elif ch6_fell and self._in_tracking:
                        self._in_tracking = False
                        self.set_mode_stabilize()
                    elif not self._in_tracking:
                        if not ch6_on:
                            self.set_mode_stabilize()
                        elif self._armed:
                            self.set_mode_auto()

                elif self._ch6_force_active():
                    ch6_force_rose = not self._prev_ch6_force_on
                    if ch6_force_rose:
                        # Rising edge → reset to MANUAL + WP 0 (takeoff)
                        self._in_tracking    = False
                        self._wp_takeoff_sent = False
                        self.set_mode_manual()
                        self._force_wp_takeoff()
                        print("[MODE] ch6 force-active ↑ → MANUAL + WP 0 reset")
                    elif ch6_fell:
                        self._in_tracking = False
                        self.set_mode_auto()
                    elif ch6_on and target_locked and not self._in_tracking:
                        self.set_mode_tracking()
                        self._in_tracking = True

                self._prev_ch6_on       = ch6_on
                self._prev_ch6_force_on = self._ch6_force_active()

                # ── 4. CSV / video lifecycle ──────────────────────────────────
                # Takeoff file: open on STABILIZE entry (pre-launch), close when WP leaves 1.
                if self._record:
                    if (self._flight_mode == "STABILIZE" and
                            self._prev_flight_mode != "STABILIZE" and
                            self._ffmpeg_tkof is None):
                        self._pending_video_open = True
                        print("[REC] STABILIZE — warming up FPS before takeoff recording")
                    elif self._current_wp != 1 and self._prev_wp == 1:
                        self._pending_video_open = False
                        self._close_video("takeoff")

                # Open takeoff video once frame_times buffer is full.
                if (self._pending_video_open and self._ffmpeg_tkof is None and
                        len(frame_times) >= frame_times.maxlen):
                    self._open_video(frame.shape, fps, "takeoff")
                    self._pending_video_open = False

                if self._in_tracking and not prev_in_tracking:
                    if self._debug_log and self._csv_writer is None:
                        self._open_csv()
                    if self._record and self._ffmpeg is None:
                        self._open_video(frame.shape, fps, "tracking")
                elif not self._in_tracking and prev_in_tracking:
                    # Tracking just stopped — send a final zero error so
                    # ArduPlane does not act on any stale error still in flight.
                    self.send_tracking(0.0, 0.0)

                if (self._flight_mode in ("STABILIZE", "MANUAL") and
                        self._prev_flight_mode not in ("STABILIZE", "MANUAL")):
                    if self._debug_log:
                        self._close_csv()
                    if self._record:
                        self._close_video("tracking")
                prev_in_tracking       = self._in_tracking
                self._prev_wp          = self._current_wp
                self._prev_flight_mode = self._flight_mode

                # ── 5. Feed TRACKING error while active ───────────────────────
                alt_dist_m  = self._alt_msl_m - self._target_alt_msl
                self.seeker.pitch_offset_norm = self._pitch_offset / _TRK_MAX_DEG

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
                        self._log_row(now, errorx, errory, target_locked=True)

                    else:
                        self._lost_count += 1
                        if self._lost_count >= 10:
                            self._lost_count  = 0
                            self._in_tracking = False
                            if self._ch6_active() or self._ch6_force_active():
                                self.set_mode_auto()
                        else:
                            self.send_tracking(self._last_errorx, self._last_errory)
                            self._log_row(now, self._last_errorx, self._last_errory,
                                          target_locked=False)

                # ── 6. Annotate HUD ───────────────────────────────────────────
                h_frame  = annotated.shape[0]
                dist_tgt_m = dist_to_target_m if dist_to_target_m is not None else self._dist_to_target_m()
                spd_kmh  = self._groundspeed_ms * 3.6
                err_str  = f"  ex={errorx:+.3f} ey={errory:+.3f}" if target_locked else ""
                mode_wp  = "%s:%d" % (self._flight_mode, self._current_wp)
                desc = "%s, dist %.3f km, alt %+.0f m, v %.2f km/jam, throttle %.0f%%." % (
                    mode_wp, dist_tgt_m / 1000.0,
                    alt_dist_m, spd_kmh, self._throttle_pct,
                )
                cv2.putText(annotated,
                            f"FPS: {fps:.1f}  SEEKER: {'ON' if self._ch6_active() or self._ch6_force_active() else 'OFF'}{err_str}",
                            (5, h_frame - 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(annotated,
                            desc,
                            (5, h_frame - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2)

                self._hud.draw_hud(True, annotated,
                                   self._lat, self._lon,
                                   self._yaw_deg, self._pitch_deg, self._roll_deg,
                               self._pitch_offset / _TRK_MAX_DEG)

                if self._record:
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
            self._close_video("takeoff")
            self._close_video("tracking")
            self.seeker.close()
