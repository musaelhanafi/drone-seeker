import struct

import cv2
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mav_module

from seeker import Seeker

# ── Custom TRACKING MAVLink message (ArduPlane-specific, ID 230) ──────────────
# Defined in ArduPlane/GCS_MAVLink_Plane.cpp.
# errorx / errory are normalised to [-1, 1]; the flight controller converts
# them to radians using TRACKING_MAX_DELTA_RAD (3 deg).

_TRACKING_MSG_ID    = 230
_TRACKING_CRC_EXTRA = 250

# RC channel 6 PWM threshold to consider the switch "active"
_CH6_ACTIVE_PWM = 1400

# ArduPlane custom mode numbers
_TRACKING_MODE = 27
_LOITER_MODE   = 5   # fallback mode when ch6 goes low


class MAVLink_tracking_message(mav_module.MAVLink_message):
    id                 = _TRACKING_MSG_ID
    msgname            = "TRACKING"
    fieldnames         = ["errorx", "errory"]
    ordered_fieldnames = ["errorx", "errory"]
    fieldtypes         = ["float", "float"]
    fielddisplays_by_name: dict = {}
    fieldenums_by_name: dict    = {}
    fieldunits_by_name: dict    = {}
    native_format      = bytearray(b"<ff")
    orders             = [0, 1]
    lengths            = [1, 1]
    array_lengths      = [0, 0]
    crc_extra          = _TRACKING_CRC_EXTRA
    unpacker           = struct.Struct("<ff")
    instance_field     = None
    instance_offset    = -1

    def __init__(self, errorx: float, errory: float):
        super().__init__(_TRACKING_MSG_ID, "TRACKING")
        self._fieldnames = self.fieldnames
        self.errorx = errorx
        self.errory = errory

    def pack(self, mav, crc_extra=_TRACKING_CRC_EXTRA, payload=None):
        if payload is None:
            payload = struct.pack("<ff", self.errorx, self.errory)
        return super().pack(mav, crc_extra=crc_extra, payload=payload)


class SeekerCtrl:
    def __init__(
        self,
        connection_string: str,
        baud: int = 57600,
        source: int | str = 0,
    ):
        self.connection_string = connection_string
        self.baud   = baud
        self.master = None

        self.rc_channels    = {}
        self._in_tracking   = False   # True while flight mode is TRACKING

        self.seeker = Seeker(source=source)

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

    # ── TRACKING MAVLink message ──────────────────────────────────────────────

    def send_tracking(self, errorx: float, errory: float):
        """Send MAVLink TRACKING message (ID 230).

        errorx: normalised horizontal error [-1, 1]  (positive = target right)
        errory: normalised vertical error   [-1, 1]  (positive = target above)
        """
        msg = MAVLink_tracking_message(errorx, errory)
        self.master.mav.send(msg)

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
        try:
            while True:
                # ── 1. Grab frame & run pink CamShift tracker ─────────────────
                ok, frame = self.seeker.read_frame()
                if not ok:
                    print("[Ctrl] End of stream.")
                    break

                annotated, cx, cy = self.seeker.track(frame)
                errorx, errory    = self.seeker.error_xy(cx, cy, frame.shape)
                target_locked     = errorx is not None

                # ── 2. Poll RC (non-blocking) ─────────────────────────────────
                self._poll_rc()
                ch6_on = self._ch6_active()

                # ── 3. Mode management ────────────────────────────────────────
                if ch6_on and target_locked and not self._in_tracking:
                    self.set_mode_tracking()
                    self._in_tracking = True

                elif not ch6_on and self._in_tracking:
                    self.set_mode_loiter()
                    self._in_tracking = False

                elif not target_locked and self._in_tracking:
                    self._in_tracking = False

                # ── 4. Feed TRACKING error while active ───────────────────────
                if self._in_tracking and target_locked:
                    self.send_tracking(errorx, errory)

                # ── 5. Annotate HUD ───────────────────────────────────────────
                ch6_pwm = self.rc_channels.get("ch6", 0)
                status  = "TRACKING" if self._in_tracking else (
                    "CH6 OFF" if not ch6_on else "NO TARGET"
                )

                cv2.putText(annotated, f"CH6: {ch6_pwm} pwm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated, f"Mode: {status}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if target_locked:
                    cv2.putText(
                        annotated,
                        f"ex={errorx:+.3f}  ey={errory:+.3f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )

                cv2.imshow(self.seeker.window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[Ctrl] Quit.")
                    break
                elif key == ord("r"):
                    self.seeker._roi_hist  = None
                    self.seeker._track_win = None
                    print("[Ctrl] Tracker reset.")

        finally:
            self.seeker.close()
