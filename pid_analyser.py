"""pid_analyser.py — Estimate PID gains for roll and pitch from tracking.csv.

Two modes
---------
PID mode (preferred):
    Requires firmware built with the GCS_MAVLink send_pid_tuning patch and
    mode_tracking using update_all().  The CSV must have non-zero pid_roll_*
    and pid_pitch_* columns (fly with --debug after rebuilding ArduPlane).

    Regression:  [e, ∫e·dt, ė] @ [P, I, D]^T ≈ pid_P + pid_I + pid_D  (cd)
    This matches ArduPlane's actual computation exactly.

Servo mode (fallback):
    Used automatically when PID columns are all zero (old firmware / CSV).
    Regresses aileron / elevator against camera error and its integral and
    derivative.  The result is the end-to-end gain (tracking PID × attitude
    controller × airframe), not the isolated PID gains.

Usage
-----
    python3 pid_analyser.py [tracking.csv]
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# ── Must match ArduPlane parameters ──────────────────────────────────────────
TRK_DBAND_DEG    = 0.573   # TRK_DBAND  (degrees)
TRK_PITCH_OFFSET = 3.0     # TRK_PITCH_OFFSET (degrees)
TRK_MAX_DEG      = 30.0    # TRK_MAX_DEG — used only in servo-fallback mode

MIN_ROWS = 20


# ── Data loading ──────────────────────────────────────────────────────────────

def load(path: str) -> dict[str, np.ndarray]:
    cols = [
        "timestamp_s", "errorx", "errory",
        "aileron", "elevator",
        "roll_deg", "pitch_deg", "roll_rate_dps", "pitch_rate_dps",
        "pid_roll_desired", "pid_roll_P", "pid_roll_I", "pid_roll_D",
        "pid_pitch_desired", "pid_pitch_P", "pid_pitch_I", "pid_pitch_D",
        "alt_agl_m",
    ]
    data = {k: [] for k in cols}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in cols:
                v = row.get(k, "").strip()
                data[k].append(float(v) if v else float("nan"))
    return {k: np.array(v) for k, v in data.items()}


def finite_mask(d: dict, *keys) -> np.ndarray:
    mask = np.ones(len(d["timestamp_s"]), dtype=bool)
    for k in keys:
        mask &= np.isfinite(d[k])
    return mask


def pid_columns_available(d: dict) -> bool:
    """Return True when at least one PID term column has non-zero data."""
    for k in ("pid_roll_P", "pid_roll_I", "pid_roll_D",
              "pid_pitch_P", "pid_pitch_I", "pid_pitch_D"):
        if np.any(d[k] != 0.0):
            return True
    return False


# ── Regression helpers ────────────────────────────────────────────────────────

def _regress(t: np.ndarray, e: np.ndarray,
             output: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares regression of output against [e, ∫e·dt, ė]."""
    dt         = np.clip(np.diff(t, prepend=t[0]), 1e-4, 0.5)
    integral   = np.cumsum(e * dt)
    derivative = np.gradient(e, t)
    X = np.column_stack([e, integral, derivative])
    result, _, _, _ = np.linalg.lstsq(X, output, rcond=None)
    P, I, D = result
    rms = float(np.sqrt(np.mean((output - X @ result) ** 2)))
    return float(P), float(I), float(D), rms


def identify_pid(t, e_deg, pid_P, pid_I, pid_D):
    """PID mode: regress against actual ArduPlane PID term outputs (cd)."""
    return _regress(t, e_deg, pid_P + pid_I + pid_D)


def identify_servo(t, e_deg, servo_norm):
    """Servo mode: regress against normalised servo output (aileron/elevator)."""
    return _regress(t, e_deg, servo_norm)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pid_axis(axes_row, t, e_deg, angle_deg,
                  pid_P, pid_I, pid_D, P, I, D, axis_name):
    dt         = np.clip(np.diff(t, prepend=t[0]), 1e-4, 0.5)
    integral   = np.cumsum(e_deg * dt)
    derivative = np.gradient(e_deg, t)
    pid_total  = pid_P + pid_I + pid_D
    predicted  = P * e_deg + I * integral + D * derivative

    ax = axes_row[0]
    ax.plot(t, e_deg,     label="PID input (deg)",    color="tab:blue")
    ax.plot(t, angle_deg, label="actual angle (deg)", color="tab:green", alpha=0.7)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title(f"{axis_name} — PID input vs actual angle")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes_row[1]
    ax.plot(t, pid_total, label="PID total (cd)",  color="tab:orange")
    ax.plot(t, predicted, label="regression fit",  color="tab:red", linestyle="--")
    ax.plot(t, pid_P,     label="P term",          color="tab:blue",   alpha=0.5, linewidth=0.8)
    ax.plot(t, pid_I,     label="I term",          color="tab:green",  alpha=0.5, linewidth=0.8)
    ax.plot(t, pid_D,     label="D term",          color="tab:purple", alpha=0.5, linewidth=0.8)
    ax.set_title(
        f"{axis_name} — PID terms vs regression fit\n"
        f"P={P:.4f}  I={I:.6f}  D={D:.4f}"
    )
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)


def plot_servo_axis(axes_row, t, e_deg, angle_deg,
                    servo, P, I, D, axis_name, servo_label):
    dt         = np.clip(np.diff(t, prepend=t[0]), 1e-4, 0.5)
    integral   = np.cumsum(e_deg * dt)
    derivative = np.gradient(e_deg, t)
    predicted  = P * e_deg + I * integral + D * derivative

    ax = axes_row[0]
    ax.plot(t, e_deg,     label="error (deg)",       color="tab:blue")
    ax.plot(t, angle_deg, label="actual angle (deg)", color="tab:green", alpha=0.7)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title(f"{axis_name} — error vs actual angle  [servo fallback]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes_row[1]
    ax.plot(t, servo,     label=servo_label,         color="tab:orange")
    ax.plot(t, predicted, label="regression fit",    color="tab:red", linestyle="--")
    ax.set_title(
        f"{axis_name} — {servo_label} vs regression fit  [end-to-end gain]\n"
        f"P={P:.6f}  I={I:.8f}  D={D:.6f}"
    )
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tracking.csv"
    try:
        d = load(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)

    if len(d["timestamp_s"]) < MIN_ROWS:
        print(f"Only {len(d['timestamp_s'])} rows — need at least {MIN_ROWS}.")
        sys.exit(1)

    pid_mode = pid_columns_available(d)

    print(f"Parameters used:")
    print(f"  TRK_DBAND_DEG    = {TRK_DBAND_DEG}")
    print(f"  TRK_PITCH_OFFSET = {TRK_PITCH_OFFSET}")
    if not pid_mode:
        print("\n[WARNING] PID columns are all zero — falling back to servo regression.")
        print("          Rebuild ArduPlane with the send_pid_tuning patch,")
        print("          re-fly with --debug, then re-run for true PID gains.\n")
    else:
        print(f"  mode             = PID (ArduPlane term outputs)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"{'PID' if pid_mode else 'Servo (fallback)'} Analysis — {path}",
        fontsize=13
    )

    # ── Roll axis ─────────────────────────────────────────────────────────────
    if pid_mode:
        need_r = ["pid_roll_desired", "pid_roll_P", "pid_roll_I", "pid_roll_D", "roll_deg"]
        mask_r = finite_mask(d, *need_r)
        mask_r &= np.abs(d["pid_roll_desired"]) > TRK_DBAND_DEG
        e_r    = d["pid_roll_desired"][mask_r]
    else:
        need_r = ["errorx", "aileron", "roll_deg"]
        mask_r = finite_mask(d, *need_r)
        # Reconstruct error in degrees; exclude deadband samples
        e_raw  = d["errorx"] * TRK_MAX_DEG
        mask_r &= np.abs(e_raw) > TRK_DBAND_DEG
        e_r    = e_raw[mask_r]

    if mask_r.sum() >= MIN_ROWS:
        t_r = d["timestamp_s"][mask_r] - d["timestamp_s"][mask_r][0]
        if pid_mode:
            P_r, I_r, D_r, rms_r = identify_pid(
                t_r, e_r,
                d["pid_roll_P"][mask_r],
                d["pid_roll_I"][mask_r],
                d["pid_roll_D"][mask_r],
            )
            label = "cd"
        else:
            P_r, I_r, D_r, rms_r = identify_servo(t_r, e_r, d["aileron"][mask_r])
            label = "servo"
        print(f"\n── Roll estimate ({mask_r.sum()} samples) ──")
        print(f"  P = {P_r:.6f}")
        print(f"  I = {I_r:.8f}")
        print(f"  D = {D_r:.6f}")
        print(f"  RMS residual ({label}) = {rms_r:.6f}")
    else:
        print(f"\nNot enough active roll samples ({mask_r.sum()}).")
        P_r = I_r = D_r = None

    # ── Pitch axis ────────────────────────────────────────────────────────────
    if pid_mode:
        need_p = ["pid_pitch_desired", "pid_pitch_P", "pid_pitch_I", "pid_pitch_D", "pitch_deg"]
        mask_p = finite_mask(d, *need_p)
        e_p    = d["pid_pitch_desired"][mask_p]
    else:
        need_p = ["errory", "elevator", "pitch_deg"]
        mask_p = finite_mask(d, *need_p)
        # Reconstruct pitch error: apply pitch offset (same as mode_tracking.cpp)
        e_p    = (d["errory"] * TRK_MAX_DEG - TRK_PITCH_OFFSET)[mask_p]

    if mask_p.sum() >= MIN_ROWS:
        t_p = d["timestamp_s"][mask_p] - d["timestamp_s"][mask_p][0]
        if pid_mode:
            P_p, I_p, D_p, rms_p = identify_pid(
                t_p, e_p,
                d["pid_pitch_P"][mask_p],
                d["pid_pitch_I"][mask_p],
                d["pid_pitch_D"][mask_p],
            )
            label = "cd"
        else:
            P_p, I_p, D_p, rms_p = identify_servo(t_p, e_p, d["elevator"][mask_p])
            label = "servo"
        print(f"\n── Pitch estimate ({mask_p.sum()} samples) ──")
        print(f"  P = {P_p:.6f}")
        print(f"  I = {I_p:.8f}")
        print(f"  D = {D_p:.6f}")
        print(f"  RMS residual ({label}) = {rms_p:.6f}")
    else:
        print(f"\nNot enough pitch samples ({mask_p.sum()}).")
        P_p = I_p = D_p = None

    # ── Plots ─────────────────────────────────────────────────────────────────
    if P_r is not None:
        if pid_mode:
            plot_pid_axis(axes[0], t_r, e_r,
                          d["roll_deg"][mask_r],
                          d["pid_roll_P"][mask_r],
                          d["pid_roll_I"][mask_r],
                          d["pid_roll_D"][mask_r],
                          P_r, I_r, D_r, "Roll")
        else:
            plot_servo_axis(axes[0], t_r, e_r,
                            d["roll_deg"][mask_r],
                            d["aileron"][mask_r],
                            P_r, I_r, D_r, "Roll", "aileron")

    if P_p is not None:
        if pid_mode:
            plot_pid_axis(axes[1], t_p, e_p,
                          d["pitch_deg"][mask_p],
                          d["pid_pitch_P"][mask_p],
                          d["pid_pitch_I"][mask_p],
                          d["pid_pitch_D"][mask_p],
                          P_p, I_p, D_p, "Pitch")
        else:
            plot_servo_axis(axes[1], t_p, e_p,
                            d["pitch_deg"][mask_p],
                            d["elevator"][mask_p],
                            P_p, I_p, D_p, "Pitch", "elevator")

    for ax in axes.flat:
        ax.set_xlabel("time (s)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
