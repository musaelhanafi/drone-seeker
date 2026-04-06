"""terminal_analyse.py — Analyse the terminal phase of a tracking run.

Loads a raw tracking.csv (written by seekerctrl.py with --debug) and produces
four panels focused on the terminal phase.  Before plotting the script applies
a two-stage cut to the raw data:

  Stage 1 — first pass only
    Scan from the first target-lock forward tracking the running minimum of
    dist_m.  Stop at the last row before dist_m first rises significantly
    (> max(2 m, 20 % of minimum)) above that minimum.  This isolates the first
    approach pass and discards any overshoot / second pass.

  Stage 2 — trim trailing climb
    Within the stage-1 window find the global minimum of alt_rel_m and discard
    everything after it.

The figure shows four panels (altitude, camera error, attitude, surfaces) over
the stage-2 window.  Vertical lines mark terminal-phase entry (red), nearest
distance (cyan), and lowest altitude (magenta).

Usage
-----
    python3 terminal_analyse.py [tracking.csv]
    python3 terminal_analyse.py original_tracking.csv
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Data loading ──────────────────────────────────────────────────────────────

def load(path: str) -> dict[str, np.ndarray]:
    cols = [
        "timestamp_s", "errorx", "errory",
        "aileron", "elevator",
        "roll_deg", "pitch_deg", "roll_rate_dps", "pitch_rate_dps",
        "pid_roll_desired", "pid_roll_P", "pid_roll_I", "pid_roll_D",
        "pid_pitch_desired", "pid_pitch_P", "pid_pitch_I", "pid_pitch_D",
        "alt_rel_m", "airspeed_ms", "throttle_pct",
        "nav_pitch_deg",
        "target_locked", "terminal",
        "dist_m",
    ]
    data = {k: [] for k in cols}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in cols:
                v = row.get(k, "").strip()
                data[k].append(float("nan") if v == "" else float(v))
    return {k: np.array(v) for k, v in data.items()}


def slice_data(d: dict, start: int, end: int) -> dict[str, np.ndarray]:
    """Return a new dict with every array sliced to [start:end+1]."""
    return {k: v[start:end + 1] for k, v in d.items()}


# ── Cut algorithm ─────────────────────────────────────────────────────────────

def find_first_lock(d: dict) -> int | None:
    """Return row index of first target_locked==1 sample, or None."""
    idx = np.where(d["target_locked"] == 1.0)[0]
    return int(idx[0]) if len(idx) > 0 else None


def find_first_pass_end(d: dict, first_lock_idx: int,
                        threshold_abs: float = 2.0,
                        threshold_rel: float = 0.20) -> int:
    """Stage 1: last row of the first approach pass.

    Scans forward from first_lock_idx tracking the running minimum of dist_m.
    Returns the index of that minimum the first time dist_m rises above it by
    more than max(threshold_abs, running_min * threshold_rel).
    """
    dist = d["dist_m"]
    running_min = float("inf")
    running_min_idx = first_lock_idx

    for i in range(first_lock_idx, len(dist)):
        v = dist[i]
        if np.isnan(v):
            continue
        if v <= running_min:
            running_min = v
            running_min_idx = i
        elif v > running_min + max(threshold_abs, running_min * threshold_rel):
            return running_min_idx

    return running_min_idx   # never rose — use last minimum (end of data)


def find_lowest_alt_idx(d: dict) -> int:
    """Stage 2: last row index at the global minimum alt_rel_m."""
    alt = d["alt_rel_m"]
    min_val = np.nanmin(alt)
    last_idx = int(np.where(alt == min_val)[0][-1])
    return last_idx


def apply_cuts(d: dict) -> tuple[dict, int, int]:
    """Apply stage-1 and stage-2 cuts.  Returns (cut_data, pass_end_idx, lowest_alt_idx).

    Both returned indices are relative to the *cut* dataset (row 0 = original row 0).
    """
    first_lock_idx = find_first_lock(d)
    if first_lock_idx is None:
        # No lock at all — return full data unchanged
        return d, len(d["timestamp_s"]) - 1, int(np.nanargmin(d["alt_rel_m"]))

    # Stage 1: isolate first approach pass
    pass_end_idx = find_first_pass_end(d, first_lock_idx)

    # Stage 2: trim trailing climb — keep up to lowest alt within stage-1 window
    d_stage1  = slice_data(d, 0, pass_end_idx)
    lowest_in_stage1 = find_lowest_alt_idx(d_stage1)

    # Final cut: rows 0 … lowest_in_stage1 (inclusive)
    d_cut = slice_data(d_stage1, 0, lowest_in_stage1)

    # pass_end and lowest_alt indices within d_cut
    pass_end_in_cut  = min(pass_end_idx,      len(d_cut["timestamp_s"]) - 1)
    lowest_in_cut    = lowest_in_stage1       # already trimmed to this

    return d_cut, pass_end_in_cut, lowest_in_cut


# ── Statistics helpers ────────────────────────────────────────────────────────

def find_terminal_entry(d: dict) -> float | None:
    """Return timestamp of first terminal==1 sample, or None."""
    idx = np.where(d["terminal"] == 1.0)[0]
    return float(d["timestamp_s"][idx[0]]) if len(idx) > 0 else None


def _find_min_col(d: dict, col: str) -> tuple[int, float]:
    idx = int(np.nanargmin(d[col]))
    return idx, float(d["timestamp_s"][idx])


def find_lowest_alt(d: dict) -> tuple[int, float]:
    return _find_min_col(d, "alt_rel_m")


def find_nearest_dist(d: dict) -> tuple[int, float]:
    return _find_min_col(d, "dist_m")


# ── Terminal output ───────────────────────────────────────────────────────────

def print_summary(d: dict, t_entry: float | None, first_lock_idx: int | None):
    n      = len(d["timestamp_s"])
    t0, t1 = d["timestamp_s"][0], d["timestamp_s"][-1]
    dur    = t1 - t0
    locked = int(np.nansum(d["target_locked"]))
    term   = int(np.nansum(d["terminal"]))

    unlocked     = n - locked
    pct_locked   = 100.0 * locked   / n if n else 0.0
    pct_unlocked = 100.0 * unlocked / n if n else 0.0

    print(f"\n{'─'*55}")
    print(f"  File duration   : {dur:.1f} s  ({n} rows)")
    print(f"  Target locked   : {locked} rows  ({pct_locked:.1f}%)")
    print(f"  Track lost      : {unlocked} rows  ({pct_unlocked:.1f}%)")
    print(f"  Terminal phase  : {term} rows  ({100*term/n:.0f}%)")

    if first_lock_idx is not None:
        fl_t    = d["timestamp_s"][first_lock_idx]
        fl_alt  = d["alt_rel_m"][first_lock_idx]
        fl_spd  = d["airspeed_ms"][first_lock_idx]
        fl_dist = d["dist_m"][first_lock_idx]
        print(f"\n  ── First track acquisition ──")
        print(f"  Time            : t+{fl_t - t0:.1f} s")
        print(f"  Alt above target: {fl_alt:.1f} m")
        print(f"  Speed           : {fl_spd * 3.6:.1f} km/h")
        if np.isfinite(fl_dist):
            print(f"  Distance        : {fl_dist:.1f} m")

    if t_entry is not None:
        mask_t    = d["terminal"] == 1.0
        alt_t     = d["alt_rel_m"][mask_t]
        spd_t     = d["airspeed_ms"][mask_t]
        ex_t      = d["errorx"][mask_t]
        ey_t      = d["errory"][mask_t]
        locked_t  = d["target_locked"][mask_t]
        dist_t    = d["dist_m"][mask_t]
        n_t       = len(locked_t)
        lost_t    = int(np.nansum(locked_t == 0.0))
        pct_lost_t = 100.0 * lost_t / n_t if n_t else 0.0

        print(f"\n  ── Terminal phase entry ──")
        print(f"  Entry time      : t+{t_entry - t0:.1f} s")
        print(f"  Alt above target: {alt_t[0]:.1f} m")
        print(f"  Speed at entry  : {spd_t[0] * 3.6:.1f} km/h")
        if np.isfinite(dist_t[0]):
            print(f"  Distance        : {dist_t[0]:.1f} m")
        print(f"\n  ── Terminal phase stats ──")
        print(f"  Duration        : {n_t} rows")
        print(f"  Track lost      : {lost_t} rows  ({pct_lost_t:.1f}%)")
        print(f"  Alt range       : {np.nanmin(alt_t):.1f} – {np.nanmax(alt_t):.1f} m")
        print(f"  Speed range     : {np.nanmin(spd_t) * 3.6:.1f} – {np.nanmax(spd_t) * 3.6:.1f} km/h")
        print(f"  |errorx| mean   : {np.nanmean(np.abs(ex_t)):.3f}  max: {np.nanmax(np.abs(ex_t)):.3f}")
        print(f"  |errory| mean   : {np.nanmean(np.abs(ey_t)):.3f}  max: {np.nanmax(np.abs(ey_t)):.3f}")
    else:
        print("\n  No terminal phase data (terminal column all 0).")

    print(f"{'─'*55}\n")


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot_figure(d: dict, path: str,
                 t0_full: float,
                 t_entry: float | None,
                 t_lowest: float | None,
                 t_nearest: float | None):
    t = d["timestamp_s"] - t0_full

    def s(key):
        return d[key]

    def vline(ax, ts, color, ls, label=None):
        if ts is not None:
            ax.axvline(ts - t0_full, color=color, linewidth=1.2,
                       linestyle=ls, label=label)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Terminal Phase — {path}", fontsize=13)
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    # ── Panel 1: altitude, airspeed, throttle ─────────────────────────────────
    ax1  = fig.add_subplot(gs[0])
    ax1b = ax1.twinx()
    ax1c = ax1.twinx()
    ax1c.spines["right"].set_position(("axes", 1.12))

    ax1.plot(t, s("alt_rel_m"),          color="tab:blue",   label="alt rel (m)")
    ax1b.plot(t, s("airspeed_ms") * 3.6, color="tab:orange", label="airspeed (km/h)", linestyle="--")
    ax1c.plot(t, s("throttle_pct"),       color="tab:green",  label="throttle (%)",    linestyle=":")
    vline(ax1, t_entry,   "red",     "--", "terminal entry")
    vline(ax1, t_nearest, "cyan",    ":",  "nearest dist")
    vline(ax1, t_lowest,  "magenta", ":",  "lowest alt")
    ax1.set_ylabel("alt rel (m)"); ax1b.set_ylabel("speed (km/h)"); ax1c.set_ylabel("throttle (%)")
    ax1.set_title("Altitude / Airspeed / Throttle")
    lines  = (ax1.get_legend_handles_labels()[0] +
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
    vline(ax2, t_entry,   "red",     "--")
    vline(ax2, t_nearest, "cyan",    ":")
    vline(ax2, t_lowest,  "magenta", ":")
    locked = s("target_locked")
    for i in range(len(t) - 1):
        if locked[i] == 0:
            ax2.axvspan(t[i], t[i + 1], color="grey", alpha=0.15)
    ax2.set_ylabel("normalised [-1,1]")
    ax2.set_title("Camera Error  (grey = lock lost)")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: attitude ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, s("pitch_deg"),     color="tab:blue",   label="pitch (deg)")
    ax3.plot(t, s("roll_deg"),      color="tab:orange",  label="roll (deg)")
    ax3.plot(t, s("nav_pitch_deg"), color="tab:green",   label="nav_pitch (deg)", linestyle="--")
    ax3.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    vline(ax3, t_entry,   "red",     "--")
    vline(ax3, t_nearest, "cyan",    ":")
    vline(ax3, t_lowest,  "magenta", ":")
    ax3.set_ylabel("degrees")
    ax3.set_title("Aircraft Attitude")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: control surfaces ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(t, s("elevator"), color="tab:blue",   label="elevator")
    ax4.plot(t, s("aileron"),  color="tab:orange",  label="aileron")
    ax4.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    vline(ax4, t_entry,   "red",     "--")
    vline(ax4, t_nearest, "cyan",    ":",  "nearest dist")
    vline(ax4, t_lowest,  "magenta", ":",  "lowest alt")
    ax4.set_ylabel("normalised")
    ax4.set_xlabel("time (s)")
    ax4.set_title("Control Surfaces (elevon demix)")
    ax4.legend(fontsize=7, loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tracking.csv"
    try:
        d_raw = load(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)

    if len(d_raw["timestamp_s"]) < 5:
        print("Not enough data rows.")
        sys.exit(1)

    # Apply stage-1 (first pass) and stage-2 (lowest alt) cuts
    d, _pass_end, _lowest = apply_cuts(d_raw)

    t_entry        = find_terminal_entry(d)
    first_lock_idx = find_first_lock(d)
    _, t_lowest    = find_lowest_alt(d)
    _, t_nearest   = find_nearest_dist(d)

    print_summary(d, t_entry, first_lock_idx)

    t0_full = d_raw["timestamp_s"][0]
    _plot_figure(d, path, t0_full=t0_full,
                 t_entry=t_entry, t_lowest=t_lowest, t_nearest=t_nearest)
    plt.show()


if __name__ == "__main__":
    main()
