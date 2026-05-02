"""Regenerate chart_02_initial_architecture.png"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.2)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── helpers ──────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=11):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          linewidth=1.2, edgecolor="#555", facecolor=color,
                          zorder=2)
    ax.add_patch(rect)
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.19, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=3)
        ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=9, color="white", zorder=3)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=3)

def container(ax, x, y, w, h, title, icon="", fontsize=10, zorder=0):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.12",
                          linewidth=1.5, edgecolor="#CCAA00",
                          facecolor="#FFFDE7", zorder=zorder)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.22, f"{icon}  {title}",
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#555", zorder=zorder + 1)

def sub_container(ax, x, y, w, h, title, zorder=1):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          linewidth=1.2, edgecolor="#CCAA00",
                          facecolor="#FFF9C4", zorder=zorder)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.19, title,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="#777", zorder=zorder + 1)

def arrow(ax, x0, y0, x1, y1, label="", two_way=False,
          rad=0.0, label_offset=(0, 0.13), label_ha="center"):
    style = "<->" if two_way else "->"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color="#333", lw=1.5,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=4)
    if label:
        mx = (x0 + x1) / 2 + label_offset[0]
        my = (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, ha=label_ha, va="bottom",
                fontsize=9, color="#333", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))

# ── containers ───────────────────────────────────────────────────────────────
container(ax, 0.2,  0.3, 6.2, 4.6, "Darat",   icon="\u2302", zorder=0)
container(ax, 7.6,  0.3, 5.0, 4.6, "Udara",   icon="\u2708", zorder=0)

# ── Laptop sub-container ──────────────────────────────────────────────────────
sub_container(ax, 0.45, 1.4, 5.7, 1.95, "\U0001F4BB  Laptop", zorder=1)

# ── component boxes ───────────────────────────────────────────────────────────
# RC Transmitter  (top of Darat)
box(ax, 0.5,  3.55, 5.7, 0.95, "#5B7FD4", "RC Transmitter")

# X-Plane  (left inside Laptop)
box(ax, 0.65, 1.62, 2.4, 0.95, "#7B5EA7", "X-Plane", sublabel="(Simulator HITL)")

# MAVProxy  (right inside Laptop)
box(ax, 3.3,  1.62, 2.65, 0.95, "#3A9E8A", "MAVProxy")

# QGC  (below Laptop, inside Darat)
box(ax, 0.5,  0.42, 5.7, 0.82, "#2ECC71", "QGroundControl")

# RC Receiver  (Pesawat)
box(ax, 7.8,  3.55, 2.1, 0.95, "#4A90D9", "RC Receiver", sublabel="(SBUS/PWM)")

# Flight Controller  (Pesawat)
box(ax, 10.2, 3.55, 2.1, 0.95, "#E07B39", "Flight Controller", sublabel="(ArduPlane)")

# ── arrows ────────────────────────────────────────────────────────────────────
# 1. RC TX → RC RX : 2.4 GHz RF
arrow(ax, 6.2, 4.025, 7.8, 4.025, label="2.4 GHz RF")

# 2. RC RX → FC
arrow(ax, 9.9, 4.025, 10.2, 4.025)

# 3. FC ↔ X-Plane : pppd / Telem  (diagonal, bidirectional)
arrow(ax, 10.2, 3.65, 3.05, 2.57,
      label="pppd / Telem", two_way=True,
      rad=0.0, label_offset=(0.3, 0.13))

# 4. FC → MAVProxy : USB serial  (curved to avoid overlap with pppd arrow)
arrow(ax, 10.2, 3.55, 5.95, 2.1,
      label="USB serial", two_way=False,
      rad=-0.25, label_offset=(0.5, -0.28), label_ha="center")

# 5. MAVProxy → QGC : MAVLink UDP  (short vertical)
arrow(ax, 4.625, 1.62, 4.625, 1.24,
      label="MAVLink UDP", two_way=False,
      rad=0.0, label_offset=(0.55, 0.0))

# ── save ─────────────────────────────────────────────────────────────────────
out = "chart_02_initial_architecture.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
