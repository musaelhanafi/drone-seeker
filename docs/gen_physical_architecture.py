"""Generate chart_00_physical_architecture.png — no overlapping arrows."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── helpers ───────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=10):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          linewidth=1.2, edgecolor="#555", facecolor=color, zorder=3)
    ax.add_patch(rect)
    cx, cy = x + w/2, y + h/2
    if sublabel:
        ax.text(cx, cy+0.18, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)
        ax.text(cx, cy-0.22, sublabel, ha="center", va="center",
                fontsize=8.5, color="white", zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)

def container(ax, x, y, w, h, title, facecolor, edgecolor, fontsize=10.5, zorder=0):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          linewidth=1.8, edgecolor=edgecolor, facecolor=facecolor, zorder=zorder)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.28, title,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#444", zorder=zorder+1)

def simple_arrow(ax, x0, y0, x1, y1, label="", two_way=False,
                 dashed=False, loff=(0, 0.12)):
    style = "<->" if two_way else "->"
    ls = (0, (4, 3)) if dashed else "solid"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color="#333", lw=1.4,
                                linestyle=ls,
                                connectionstyle="arc3,rad=0"), zorder=5)
    if label:
        mx, my = (x0+x1)/2 + loff[0], (y0+y1)/2 + loff[1]
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=8.5, color="#333", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))

def routed_arrow(ax, points, label="", label_seg=None,
                 two_way=False, dashed=False):
    """Draw a multi-segment (L/U-shaped) arrow along waypoints."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ls = (0, (4, 3)) if dashed else "solid"

    # Draw all segments as lines
    ax.plot(xs, ys, color="#333", linewidth=1.4, linestyle=ls, zorder=5,
            solid_capstyle="round")

    # Arrowhead at destination
    ax.annotate("", xy=(xs[-1], ys[-1]),
                xytext=(xs[-2]+(xs[-1]-xs[-2])*0.01,
                        ys[-2]+(ys[-1]-ys[-2])*0.01),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.4), zorder=5)
    if two_way:
        ax.annotate("", xy=(xs[0], ys[0]),
                    xytext=(xs[1]+(xs[0]-xs[1])*0.01,
                            ys[1]+(ys[0]-ys[1])*0.01),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.4), zorder=5)

    if label:
        # Place label on the specified segment index (default middle)
        if label_seg is None:
            label_seg = len(xs) // 2
        i = min(label_seg, len(xs)-2)
        mx = (xs[i] + xs[i+1]) / 2
        my = (ys[i] + ys[i+1]) / 2
        # offset label away from line
        if abs(ys[i] - ys[i+1]) < 0.01:   # horizontal
            dy = 0.16
        else:                               # vertical
            dy = 0.0
        ax.text(mx, my + dy, label, ha="center", va="bottom",
                fontsize=8.5, color="#333", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

# ── layout constants ──────────────────────────────────────────────────────────
# DARAT: x 0.2–7.0   UDARA: x 8.6–15.8   gap x 7.0–8.6
# Rows:  top y≈5.4 (RC),  mid y≈4.05 (telemetry),  low y 0.7–3.4 (laptop / Pi)

DL, DR = 0.2, 7.0     # Darat left/right
UL, UR = 8.6, 15.8    # Udara left/right

# ── DARAT container ───────────────────────────────────────────────────────────
container(ax, DL, 0.4, DR-DL, 6.0, "Darat", "#eef4fb", "#2c6fad", zorder=0)

# RC Transmitter (top strip)
box(ax, 0.45, 4.9, 6.3, 1.0, "#4A90D9", "RC Transmitter")

# Telemetry RX (mid, above QGroundControl)
box(ax, 0.65, 3.6, 2.6, 0.9, "#8E7CC3", "Telemetry RX", fontsize=9)

# Laptop sub-container
container(ax, 0.45, 0.7, 6.3, 2.7, "Laptop", "#f0faf5", "#1d7a68", fontsize=9, zorder=1)

# QGroundControl  (left)
box(ax, 0.65, 1.4, 2.6, 1.0, "#2ECC71", "QGroundControl", fontsize=8.5)

# X-Plane  (right)
box(ax, 3.75, 1.4, 2.7, 1.0, "#7B5EA7", "X-Plane", fontsize=9)

# Telemetry RX -> QGroundControl (MAVLink over telemetry radio)
simple_arrow(ax, 1.95, 3.6, 1.95, 2.4, label="MAVLink", loff=(0.55, 0))

# ── UDARA container ───────────────────────────────────────────────────────────
container(ax, UL, 0.4, UR-UL, 6.0, "Udara", "#fff8ec", "#b85c1a", zorder=0)

# RC Receiver  (top-left of Udara)
box(ax, 8.85, 4.9, 2.9, 1.0, "#5BA3E0", "RC Receiver", sublabel="(SBUS)")

# Flight Controller  (top-right of Udara)
box(ax, 12.1, 4.9, 3.4, 1.0, "#E07B39", "Flight Controller", sublabel="(ArduPlane)")

# Telemetry TX  (mid-left of Udara)
box(ax, 8.85, 3.6, 2.6, 0.9, "#8E7CC3", "Telemetry TX", fontsize=9)

# RC RX -> FC
simple_arrow(ax, 11.75, 5.4, 12.1, 5.4)

# FC -> Telemetry TX  (TELEM1, routed to the mid-left radio)
routed_arrow(ax,
    [(12.1, 4.6), (11.6, 4.6), (11.6, 4.05), (11.45, 4.05)],
    label="TELEM1", label_seg=0)

# Raspberry PI sub-container (iptables + Kamera + Seeker)
container(ax, 8.85, 0.7, 6.65, 2.7, "Raspberry PI", "#fef9e7", "#d4ac0d",
          fontsize=9, zorder=1)

# iptables forward  (upper-left of Pi — bridges pppd link to the X-Plane app)
box(ax, 9.05, 1.7, 2.7, 0.9, "#16A085", "iptables\nforward", fontsize=8.5)

# Seeker  (upper-right of Pi, below FC)
box(ax, 12.4, 1.7, 2.9, 0.9, "#E05C5C", "Seeker", fontsize=9)

# Kamera  (below Seeker)
box(ax, 12.4, 0.75, 2.9, 0.8, "#F0C040", "Kamera", fontsize=9)

# FC <-> Seeker : UART (MAVLink) — clean vertical, right column
simple_arrow(ax, 13.85, 4.9, 13.85, 2.6, label="UART", two_way=True, loff=(0.62, 0))

# FC <-> iptables : pppd TELEM2 — routed above the Pi, dropping in left of the title
routed_arrow(ax,
    [(12.65, 4.9), (12.65, 3.5), (10.4, 3.5), (10.4, 2.6)],
    label="pppd TELEM2", label_seg=0, two_way=True)

# Kamera -> Seeker
simple_arrow(ax, 13.85, 1.55, 13.85, 1.7)

# ── cross-container arrows — routed to avoid overlap ─────────────────────────

# 1. RC TX → RC RX : 2.4 GHz RF  (horizontal through gap, clear)
simple_arrow(ax, 6.75, 5.4, 8.85, 5.4, label="2.4 GHz RF")

# 2. Telemetry TX ↔ Telemetry RX : telemetry RF  (horizontal, clear band)
simple_arrow(ax, 8.85, 4.05, 3.25, 4.05, label="telemetry RF", two_way=True)

# 3. iptables ↔ X-Plane : WiFi, iptables forward (X-Plane HITL data)
#    The Raspberry Pi forwards X-Plane data to the X-Plane app on the laptop.
simple_arrow(ax, 9.05, 2.15, 6.45, 1.9, label="WiFi · iptables", two_way=True, loff=(0, 0.22))

# ── save ─────────────────────────────────────────────────────────────────────
out = "chart_00_physical_architecture.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
