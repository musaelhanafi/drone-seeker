"""tracking_analyse.py — Plot tracking.csv flight data.

Usage:
    python3 tracking_analyse.py [tracking.csv]

Produces three side-by-side subplots:
    1. errorx  vs aileron
    2. errory  vs elevator
    3. pitch_deg vs geo_pitch_deg
"""

import sys
import csv
import matplotlib.pyplot as plt


def load_csv(path: str) -> dict[str, list[float]]:
    cols = {
        "timestamp_s": [], "errorx": [], "errory": [],
        "aileron": [], "elevator": [],
        "pitch_deg": [], "geo_pitch_deg": [],
    }
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in cols:
                val = row.get(key, "").strip()
                cols[key].append(float(val) if val else float("nan"))
    return cols


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tracking.csv"
    try:
        d = load_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)

    t = d["timestamp_s"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Tracking Analysis — {path}", fontsize=13)

    # ── 1. errorx vs aileron ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, d["errorx"],  label="errorx",  color="tab:blue")
    ax.plot(t, d["aileron"], label="aileron",  color="tab:orange")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("normalised / scaled")
    ax.set_title("errorx vs aileron")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── 2. errory vs elevator ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, d["errory"],   label="errory",   color="tab:blue")
    ax.plot(t, d["elevator"], label="elevator",  color="tab:orange")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("normalised / scaled")
    ax.set_title("errory vs elevator")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── 3. pitch_deg vs geo_pitch_deg ─────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, d["pitch_deg"],     label="pitch (actual)",   color="tab:blue")
    ax.plot(t, d["geo_pitch_deg"], label="geo pitch (target)", color="tab:red",
            linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("degrees")
    ax.set_xlabel("time (s)")
    ax.set_title("pitch_deg vs geo_pitch_deg")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
