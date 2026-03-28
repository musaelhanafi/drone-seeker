"""
joystick_handler.py — Joystick input → RC PWM channel values.

Wraps pygame joystick reads and exposes the latest channel values
as a dict {ch1..ch5: pwm_int}.  Intended to be polled by a background
thread in SeekerCtrl.

Axis mapping (matches test_joystick.py defaults):
  Axis 0 → CH1 aileron   (centred)
  Axis 1 → CH2 elevator  (centred, auto-inverted)
  Axis 2 → CH3 throttle  (full-range 1000-2000)
  Axis 3 → CH4 rudder    (centred)
  Axis 4 → CH5 mode      (snapped to ArduPilot flight-mode bands)
"""

try:
    import pygame
    _PYGAME = True
except ImportError:
    _PYGAME = False

UINT16_MAX = 65535

_FLTMODE_CENTRES = [1165, 1295, 1425, 1555, 1685, 1815]


def _axis_pwm(v: float, invert: bool = False) -> int:
    if invert:
        v = -v
    return int(max(1000, min(2000, 1500 + v * 500)))


def _thr_pwm(v: float, invert: bool = False) -> int:
    if invert:
        v = -v
    return int(max(1000, min(2000, 1000 + (v + 1.0) * 500)))


def _fltmode_pwm(v: float) -> int:
    raw = int(max(1000, min(2000, 1500 + v * 500)))
    return min(_FLTMODE_CENTRES, key=lambda c: abs(c - raw))


class JoystickHandler:
    """Manages a single pygame joystick and converts axes to PWM values."""

    def __init__(
        self,
        joy_index: int = 0,
        axes: tuple[int, int, int, int, int] = (0, 1, 2, 3, 4),
        invert_axes: set[int] | None = None,
        thr_invert: bool = True,
    ):
        if not _PYGAME:
            raise RuntimeError("pygame is not installed — run: pip3 install pygame")

        self.joy_index   = joy_index
        self.roll_ax, self.pitch_ax, self.thr_ax, self.yaw_ax, self.mode_ax = axes
        self.invert_axes = invert_axes or set()
        self.thr_invert  = thr_invert

        self._joy: "pygame.joystick.Joystick | None" = None

    def open(self) -> str:
        """Initialise pygame and open the joystick.  Returns the device name."""
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError("No joysticks detected")
        if self.joy_index >= count:
            raise RuntimeError(
                f"Joystick index {self.joy_index} out of range (found {count})"
            )
        self._joy = pygame.joystick.Joystick(self.joy_index)
        self._joy.init()
        name = self._joy.get_name()
        print(f"[JOY] Using [{self.joy_index}] {name}  "
              f"axes={self._joy.get_numaxes()}  buttons={self._joy.get_numbuttons()}")
        return name

    def close(self):
        if self._joy is not None:
            self._joy.quit()
            self._joy = None
        pygame.joystick.quit()
        pygame.quit()

    def read_channels(self) -> dict[str, int]:
        """Pump pygame events and return the current {ch1..ch5: pwm} dict."""
        if self._joy is None:
            raise RuntimeError("JoystickHandler not opened — call open() first")

        pygame.event.pump()

        ch1 = _axis_pwm(self._joy.get_axis(self.roll_ax),
                        self.roll_ax in self.invert_axes)
        ch2 = _axis_pwm(self._joy.get_axis(self.pitch_ax),
                        self.pitch_ax in self.invert_axes or True)  # always inverted
        ch3 = _thr_pwm(self._joy.get_axis(self.thr_ax),
                       self.thr_ax in self.invert_axes or self.thr_invert)
        ch4 = _axis_pwm(self._joy.get_axis(self.yaw_ax),
                        self.yaw_ax in self.invert_axes)

        n_axes = self._joy.get_numaxes()
        ch5 = (_fltmode_pwm(self._joy.get_axis(self.mode_ax))
               if self.mode_ax < n_axes else 1165)

        return {"ch1": ch1, "ch2": ch2, "ch3": ch3, "ch4": ch4, "ch5": ch5}
