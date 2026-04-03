"""terminal_analyse.py — Analyse the terminal phase of a tracking run.

Loads tracking.csv (written by seekerctrl.py with --debug) and produces
four panels focused on the terminal phase (terminal == 1) and the seconds
just before it:

  1. Altitude + airspeed + throttle vs time
  2. errorx / errory (camera error) vs time
  3. pitch_deg + roll_deg vs time
  4. Elevator + aileron (surface) vs time

A vertical dashed line marks the terminal phase entry.

Usage
-----
    python3 terminal_analyse.py [tracking.csv]
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# How many seconds before terminal phase to include in the overview window.
PRE_TERMINAL_S = 10.0


def load(path: str) -> dict[str, np.ndarray]:
    cols = [
        "timestamp_s", "errorx", "errory",
        "aileron", "elevator",
        "roll_deg", "pitch_deg", "roll_rate_dps", "pitch_rate_dps",
        "pid_roll_desired", "pid_roll_P", "pid_roll_I", "pid_roll_D",
        "pid_pitch_desired", "pid_pitch_P", "pid_pitch_I", "pid_pitch_D",
        "alt_agl_m", "airspeed_ms", "throttle_pct",
        "target_locked", "terminal",
    ]
    data = {k: [] for k in cols}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in cols:
                v = row.get(k, "").strip()
                if v == "":
                    data[k].append(float("nan"))
                else:
                    data[k].append(float(v))
    return {k: np.array(v) for k, v in data.items()}


def find_terminal_entry(d: dict) -> float | None:
    """Return timestamp of first terminal==1 sample, or None."""
    idx = np.where(d["terminal"] == 1.0)[0]
    if len(idx) == 0:
        return None
    return float(d["timestamp_s"][idx[0]])


def print_summary(d: dict, t_entry: float | None):
    n       = len(d["timestamp_s"])
    t0, t1  = d["timestamp_s"][0], d["timestamp_s"][-1]
    dur     = t1 - t0
    locked  = int(np.nansum(d["target_locked"]))
    term    = int(np.nansum(d["terminal"]))

    print(f"\n{'─'*55}")
    print(f"  File duration   : {dur:.1f} s  ({n} rows)")
    print(f"  Target locked   : {locked} rows  ({100*locked/n:.0f}%)")
    print(f"  Terminal phase  : {term} rows  ({100*term/n:.0f}%)")

    if t_entry is not None:
        mask_t = d["terminal"] == 1.0
        alt_t  = d["alt_agl_m"][mask_t]
        spd_t  = d["airspeed_ms"][mask_t]
        ex_t   = d["errorx"][mask_t]
        ey_t   = d["errory"][mask_t]
        print(f"\n  ── Terminal phase entry ──")
        print(f"  Entry time      : t+{t_entry - t0:.1f} s")
        print(f"  Alt at entry    : {alt_t[0]:.1f} m AGL")
        print(f"  Speed at entry  : {spd_t[0]:.1f} m/s")
        print(f"\n  ── Terminal phase stats ──")
        print(f"  Alt range       : {np.nanmin(alt_t):.1f} – {np.nanmax(alt_t):.1f} m")
        print(f"  Speed range     : {np.nanmin(spd_t):.1f} – {np.nanmax(spd_t):.1f} m/s")
        print(f"  |errorx| mean   : {np.nanmean(np.abs(ex_t)):.3f}  max: {np.nanmax(np.abs(ex_t)):.3f}")
        print(f"  |errory| mean   : {np.nanmean(np.abs(ey_t)):.3f}  max: {np.nanmax(np.abs(ey_t)):.3f}")

        # Estimate miss distance from final errorx/errory and altitude
        last_alt = alt_t[-1]
        last_ex  = ex_t[-1]
        last_ey  = ey_t[-1]
        if np.isfinite(last_alt) and np.isfinite(last_ex) and np.isfinite(last_ey):
            from math import tan, radians, sqrt
            # errorx/errory are normalised [-1,1]; full-scale = TRK_MAX_DEG = 30°
            TRK_MAX_DEG = 30.0
            ang_x = last_ex * TRK_MAX_DEG
            ang_y = last_ey * TRK_MAX_DEG
            miss_x = abs(last_alt * tan(radians(ang_x)))
            miss_y = abs(last_alt * tan(radians(ang_y)))
            miss   = sqrt(miss_x**2 + miss_y**2)
            print(f"\n  ── Estimated miss at last frame ──")
            print(f"  Alt             : {last_alt:.1f} m")
            print(f"  Lateral (x)     : {miss_x:.1f} m")
            print(f"  Vertical (y)    : {miss_y:.1f} m")
            print(f"  Total miss      : {miss:.1f} m")
    else:
        print("\n  No terminal phase data (terminal column all 0).")
        print("  Set _TRK_TERM_ALT > 0 in seekerctrl.py and re-fly.")
    print(f"{'─'*55}\n")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tracking.csv"
    try:
        d = load(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)

    if len(d["timestamp_s"]) < 5:
        print("Not enough data rows.")
        sys.exit(1)

    t_entry = find_terminal_entry(d)
    print_summary(d, t_entry)

    # ── Select window: PRE_TERMINAL_S before entry through end ────────────────
    t0_full = d["timestamp_s"][0]
    if t_entry is not None:
        t_window_start = t_entry - PRE_TERMINAL_S
    else:
        t_window_start = t0_full

    mask = d["timestamp_s"] >= t_window_start
    t    = d["timestamp_s"][mask] - t0_full   # normalise to zero

    def s(key):
        return d[key][mask]

    t_entry_rel = (t_entry - t0_full) if t_entry is not None else None

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Terminal Phase Analysis — {path}", fontsize=13)
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    def vline(ax):
        if t_entry_rel is not None:
            ax.axvline(t_entry_rel - (t_entry - t_entry), color="red",
                       linewidth=1.2, linestyle="--", label="terminal entry")

    def _vline(ax, t_ref):
        if t_entry_rel is not None:
            ax.axvline(t_entry_rel, color="red",
                       linewidth=1.2, linestyle="--", label="terminal entry")

    # ── Panel 1: altitude, airspeed, throttle ─────────────────────────────────
    ax1  = fig.add_subplot(gs[0])
    ax1b = ax1.twinx()
    ax1c = ax1.twinx()
    ax1c.spines["right"].set_position(("axes", 1.12))

    ax1.plot(t, s("alt_agl_m"),    color="tab:blue",   label="alt AGL (m)")
    ax1b.plot(t, s("airspeed_ms"), color="tab:orange",  label="airspeed (m/s)", linestyle="--")
    ax1c.plot(t, s("throttle_pct"),color="tab:green",   label="throttle (%)",   linestyle=":")
    if t_entry_rel is not None:
        ax1.axvline(t_entry_rel, color="red", linewidth=1.2, linestyle="--", label="terminal entry")
    ax1.set_ylabel("alt (m)");  ax1b.set_ylabel("speed (m/s)");  ax1c.set_ylabel("throttle (%)")
    ax1.set_title("Altitude / Airspeed / Throttle")
    lines = (ax1.get_legend_handles_labels()[0] +
             ax1b.get_legend_handles_labels()[0] +
             ax1c.get_legend_handles_labels()[0])
    labels = (ax1.get_legend_handles_labels()[1] +
              ax1b.get_legend_handles_labels()[1] +
              ax1c.get_legend_handles_labels()[1])
    ax1.legend(lines, labels, fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: camera error ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, s("errorx"), color="tab:blue",   label="errorx")
    ax2.plot(t, s("errory"), color="tab:orange",  label="errory")
    ax2.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    if t_entry_rel is not None:
        ax2.axvline(t_entry_rel, color="red", linewidth=1.2, linestyle="--")
    # shade locked / unlocked
    locked = s("target_locked")
    for i in range(len(t) - 1):
        if locked[i] == 0:
            ax2.axvspan(t[i], t[i+1], color="grey", alpha=0.15)
    ax2.set_ylabel("normalised [-1,1]")
    ax2.set_title("Camera Error  (grey = lock lost)")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: attitude ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, s("pitch_deg"), color="tab:blue",   label="pitch (deg)")
    ax3.plot(t, s("roll_deg"),  color="tab:orange",  label="roll (deg)")
    ax3.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    if t_entry_rel is not None:
        ax3.axvline(t_entry_rel, color="red", linewidth=1.2, linestyle="--")
    ax3.set_ylabel("degrees")
    ax3.set_title("Aircraft Attitude")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: control surfaces ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(t, s("elevator"), color="tab:blue",   label="elevator")
    ax4.plot(t, s("aileron"),  color="tab:orange",  label="aileron")
    ax4.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    if t_entry_rel is not None:
        ax4.axvline(t_entry_rel, color="red", linewidth=1.2, linestyle="--")
    ax4.set_ylabel("normalised")
    ax4.set_xlabel("time (s)")
    ax4.set_title("Control Surfaces (elevon demix)")
    ax4.legend(fontsize=7, loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
