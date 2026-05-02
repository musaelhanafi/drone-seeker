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
# DARAT: x 0.2–7.0   UDARA: x 8.6–15.8
# Routing lanes: top y=6.6, bottom y=0.15, gap x=7.0–8.6

DL, DR = 0.2, 7.0     # Darat left/right
UL, UR = 8.6, 15.8    # Udara left/right
TOP_Y  = 6.55          # upper routing lane
BOT_Y  = 0.18          # lower routing lane

# ── DARAT container ───────────────────────────────────────────────────────────
container(ax, DL, 0.4, DR-DL, 6.0, "Darat", "#eef4fb", "#2c6fad", zorder=0)

# RC Transmitter (top strip)
box(ax, 0.45, 5.1, 6.3, 1.0, "#4A90D9", "RC Transmitter")

# Laptop sub-container
container(ax, 0.45, 0.65, 6.3, 4.1, "Laptop", "#f0faf5", "#1d7a68", fontsize=9, zorder=1)

# QGroundControl  (left column, bottom)
box(ax, 0.65, 0.85, 2.7, 1.0, "#2ECC71", "QGroundControl", fontsize=9)

# X-Plane  (right column, bottom)
box(ax, 3.7,  0.85, 2.8, 1.0, "#7B5EA7", "X-Plane", fontsize=9)

# Monitor  (spans full width, above QGC/XP)
box(ax, 0.65, 2.2, 5.85, 1.0, "#5B7FD4", "Monitor", fontsize=9)

# X-Plane -> Monitor
simple_arrow(ax, 5.1, 1.85, 3.575, 2.2, label="display")

# ── UDARA container ───────────────────────────────────────────────────────────
container(ax, UL, 0.4, UR-UL, 6.0, "Udara", "#fff8ec", "#b85c1a", zorder=0)

# RC Receiver  (top-left of Udara)
box(ax, 8.85, 5.1, 3.0, 1.0, "#5BA3E0", "RC Receiver", sublabel="(SBUS/PWM)")

# Flight Controller  (top-right of Udara)
box(ax, 12.2, 5.1, 3.3, 1.0, "#E07B39", "Flight Controller", sublabel="(ArduPlane)")

# RC RX -> FC
simple_arrow(ax, 11.85, 5.6, 12.2, 5.6)

# Companion sub-container
container(ax, 8.85, 0.65, 6.65, 4.1, "Companion Computer", "#fef9e7", "#d4ac0d",
          fontsize=9, zorder=1)

# Kamera  (top-left of Companion)
box(ax, 9.05, 3.0, 2.5, 1.0, "#F0C040", "Kamera", fontsize=9)

# Seeker  (below Kamera)
box(ax, 9.05, 1.65, 2.5, 1.0, "#E05C5C", "Seeker", fontsize=9)

# Kamera -> Seeker
simple_arrow(ax, 10.3, 3.0, 10.3, 2.65, label="")

# MAVProxy  (right of Companion)
box(ax, 12.2, 2.1, 3.0, 1.0, "#3A9E8A", "MAVProxy", fontsize=9)

# Seeker -> MAVProxy : MAVLink UDP
simple_arrow(ax, 11.55, 2.15, 12.2, 2.35, label="MAVLink UDP", loff=(0.1, 0.12))

# MAVProxy -> FC : serial  (vertical, no overlap)
simple_arrow(ax, 13.7, 3.1, 13.7, 5.1, label="serial", loff=(0.35, 0))

# ── cross-container arrows — routed to avoid overlap ─────────────────────────

# 1. RC TX → RC RX : 2.4 GHz RF  (horizontal through gap, clear)
simple_arrow(ax, 6.75, 5.6, 8.85, 5.6, label="2.4 GHz RF")

# 2. FC ↔ X-Plane : pppd/Telem2 — routed via TOP lane
#    FC top (13.85, 6.1) → up → (13.85, TOP_Y) → left → (5.1, TOP_Y) → down → XP top (5.1, 1.85)
routed_arrow(ax,
    [(13.85, 6.1), (13.85, TOP_Y), (5.1, TOP_Y), (5.1, 1.85)],
    label="pppd / Telem2", label_seg=1,
    two_way=True)

# 3. Monitor -.→ Kamera : visual — horizontal through gap at y=2.7
#    Monitor right (6.5, 2.7) → gap → Kamera left (9.05, 3.5)
routed_arrow(ax,
    [(6.5, 2.7), (7.8, 2.7), (7.8, 3.5), (9.05, 3.5)],
    label="visual", label_seg=0,
    dashed=True)

# 4. MAVProxy → QGC : WiFi — routed via BOTTOM lane
#    MAVProxy bottom (13.7, 2.1) → down → (13.7, BOT_Y) → left → (2.0, BOT_Y) → up → QGC bottom (2.0, 0.85)
routed_arrow(ax,
    [(13.7, 2.1), (13.7, BOT_Y), (2.0, BOT_Y), (2.0, 0.85)],
    label="WiFi", label_seg=1)

# ── save ─────────────────────────────────────────────────────────────────────
out = "chart_00_physical_architecture.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
