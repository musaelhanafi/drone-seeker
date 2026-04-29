"""terminal_analyse_simple.py — Simple two-panel plot: ex/aileron and ey/pitch.

Usage
-----
    python3 terminal_analyse_simple.py [tracking.csv]
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load(path: str) -> dict[str, np.ndarray]:
    cols = [
        "timestamp_s", "errorx", "errory",
        "aileron", "elevator",
    ]
    data = {k: [] for k in cols}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in cols:
                v = row.get(k, "").strip()
                data[k].append(float("nan") if v == "" else float(v))
    return {k: np.array(v) for k, v in data.items()}


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

    d  = d_raw
    t0 = d["timestamp_s"][0]
    t  = d["timestamp_s"] - t0

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(f"ex/aileron  &  ey/pitch — {path}", fontsize=12)
    gs = gridspec.GridSpec(2, 1, hspace=0.45)

    # ── Panel 1: errorx vs aileron ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, d["errorx"],  color="tab:blue",   label="ex (errorx)", linewidth=1.2)
    ax1.plot(t, d["aileron"], color="tab:orange",  label="aileron",     linewidth=1.2, linestyle="--")
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("normalised [-1, 1]")
    ax1.set_title("ex  vs  Aileron")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: errory vs elevator ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, d["errory"],   color="tab:blue",   label="ey (errory)", linewidth=1.2)
    ax2.plot(t, d["elevator"], color="tab:red",    label="elevator",    linewidth=1.2, linestyle="--")
    ax2.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("normalised [-1, 1]")
    ax2.set_xlabel("time (s)")
    ax2.set_title("ey  vs  Elevator")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
