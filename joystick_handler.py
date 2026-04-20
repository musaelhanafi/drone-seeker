"""
joystick_handler.py — Joystick input → RC PWM channel values.

Wraps pygame joystick reads and exposes the latest channel values
as a dict {ch1..ch6: pwm_int}.  Intended to be polled by a background
thread in SeekerCtrl.

Channel mapping:
  Axis 0 → CH1 aileron   (centred 1000-2000)
  Axis 1 → CH2 elevator  (centred 1000-2000)
  Axis 2 → CH3 throttle  (full-range 1000-2000)
  Axis 3 → CH4 rudder    (centred 1000-2000)
  Button 6 → CH5 1000, Button 7 → CH5 2000, neither → CH5 1500 (3-pos switch)
  Button 4 or 5, or axis 4/5 > 0.5 → CH6  (2000 active, 1000 released)

Run standalone to test:
    python3 joystick_handler.py [--index N]
"""

import argparse
import sys
import time

try:
    import pygame
    _PYGAME = True
except ImportError:
    _PYGAME = False

UINT16_MAX = 65535

_FLTMODE_CENTRES = [1165, 1295, 1425, 1555, 1685, 1815]

_CH_LABELS = {
    "ch1": "Roll     ",
    "ch2": "Pitch    ",
    "ch3": "Throttle ",
    "ch4": "Yaw      ",
    "ch5": "Mode     ",
    "ch6": "Arm/Ch6  ",
}


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


def _pwm_bar(pwm: int, width: int = 20) -> str:
    pos = max(0, min(width, int((pwm - 1000) / 1000 * width)))
    return "[" + "=" * pos + " " * (width - pos) + "]"


class JoystickHandler:
    """Manages a single pygame joystick and converts axes/buttons to PWM values."""

    def __init__(
        self,
        joy_index: int = 0,
        axes: tuple[int, int, int, int, int, int] = (0, 1, 2, 3, 4, 5),
        invert_axes: set[int] | None = None,
        thr_invert: bool = False,
    ):
        if not _PYGAME:
            raise RuntimeError("pygame is not installed — run: pip3 install pygame")

        self.joy_index   = joy_index
        self.roll_ax, self.pitch_ax, self.thr_ax, self.yaw_ax, self.mode_ax, self.ch6_ax = axes
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

    def pump(self):
        """Pump pygame events — must be called from the main thread on macOS."""
        pygame.event.pump()

    def read_channels(self) -> dict[str, int]:
        """Return the current {ch1..ch6: pwm} dict.
        Caller must ensure pump() has been called from the main thread first."""
        if self._joy is None:
            raise RuntimeError("JoystickHandler not opened — call open() first")

        ch1 = _axis_pwm(self._joy.get_axis(self.roll_ax),
                        self.roll_ax in self.invert_axes)
        ch2 = _axis_pwm(self._joy.get_axis(self.pitch_ax),
                        not (self.pitch_ax in self.invert_axes))
        ch3 = _thr_pwm(self._joy.get_axis(self.thr_ax),
                       self.thr_ax in self.invert_axes or self.thr_invert)
        ch4 = _axis_pwm(self._joy.get_axis(self.yaw_ax),
                        self.yaw_ax in self.invert_axes)

        n_buttons = self._joy.get_numbuttons()
        n_axes    = self._joy.get_numaxes()

        # CH5: 3-position switch
        #   button 6 pressed → 1000 (low)
        #   button 7 pressed → 2000 (high)
        #   neither pressed  → 1500 (middle)
        btn6 = 6 < n_buttons and self._joy.get_button(6)
        btn7 = 7 < n_buttons and self._joy.get_button(7)
        ch5 = 1000 if btn6 else (2000 if btn7 else 1500)

        # CH6: button 4 or 5, OR axis 4 or 5 > 0.5 (triggers mapped as axes)
        btn46 = any(i < n_buttons and self._joy.get_button(i) for i in (4, 5))
        ax46  = any(i < n_axes    and self._joy.get_axis(i) > 0.5  for i in (4, 5))
        ch6   = 2000 if btn46 or ax46 else 1000

        return {"ch1": ch1, "ch2": ch2, "ch3": ch3, "ch4": ch4, "ch5": ch5, "ch6": ch6}

    def run_test(self):
        """Live display of raw axes, buttons, hats and mapped PWM channels.
        Blocks until Q is pressed or the window is closed."""
        name      = self._joy.get_name()
        n_axes    = self._joy.get_numaxes()
        n_buttons = self._joy.get_numbuttons()
        n_hats    = self._joy.get_numhats()

        pygame.display.set_mode((1, 1))
        pygame.display.set_caption(f"Joystick: {name}")

        print(f"\nJoystick: {name}")
        print(f"  Axes: {n_axes}  Buttons: {n_buttons}  Hats: {n_hats}")
        print("─" * 60)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return

            self.pump()
            ch = self.read_channels()

            axes    = [self._joy.get_axis(i)   for i in range(n_axes)]
            buttons = [self._joy.get_button(i)  for i in range(n_buttons)]
            hats    = [self._joy.get_hat(i)     for i in range(n_hats)]

            print("\033[H\033[J", end="")
            print(f"Joystick: {name}   [Q to quit]\n")

            print("── Raw Axes ──────────────────────────────────────")
            for i, v in enumerate(axes):
                pos = int((v + 1) / 2 * 20)
                bar = "[" + "=" * pos + " " * (20 - pos) + "]"
                print(f"  Axis {i}: {bar}  {v:+.3f}")

            if buttons:
                print("\n── Buttons ───────────────────────────────────────")
                print("  " + "  ".join(
                    f"B{i}:{'█' if v else '·'}" for i, v in enumerate(buttons)
                ))

            if hats:
                print("\n── Hats ──────────────────────────────────────────")
                for i, v in enumerate(hats):
                    print(f"  Hat {i}: {v}")

            print("\n── Mapped PWM Channels ───────────────────────────")
            for key, label in _CH_LABELS.items():
                pwm = ch[key]
                print(f"  {label}  {_pwm_bar(pwm)}  {pwm} µs")

            time.sleep(0.05)


if __name__ == "__main__":
    if not _PYGAME:
        print("pygame not installed — run: pip3 install pygame")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Test joystick input and PWM mapping.")
    parser.add_argument("--index", type=int, default=0, help="Joystick index (default 0)")
    args = parser.parse_args()

    joy = JoystickHandler(joy_index=args.index)
    joy.open()
    try:
        joy.run_test()
    except KeyboardInterrupt:
        pass
    finally:
        joy.close()
        print("\nDone.")
