"""replay_tracking.py — Replay tracking video with live matplotlib overlay from CSV.

Playback stops automatically at the "hit" point (same cut logic as terminal_analyse:
first approach pass end → lowest alt).  A cyan vertical line marks the hit on all plots.

Usage:
    python3 replay_tracking.py tracking_20260421_140942.csv
    python3 replay_tracking.py tracking_20260421_140942.csv tracking_20260421_140942.mp4
    python3 replay_tracking.py tracking_20260421_140942.csv --speed 2.0 --trail 200
    python3 replay_tracking.py tracking_20260421_140942.csv --save replay_out.mp4

Video is auto-detected if a .mp4/.avi file with the same timestamp prefix exists.
"""

import argparse
import os
import sys
import time
import csv
import datetime

import subprocess

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.animation import FuncAnimation

# ── re-use cut algorithm from terminal_analyse ────────────────────────────────
from terminal_analyse import (
    find_first_lock,
    find_first_pass_end,
    find_lowest_alt_idx,
)

_DARK_BG   = "#ffffff"
_PANEL_BG  = "#f5f5f5"
_SPINE_COL = "#cccccc"
_TICK_COL  = "#333333"

_SAVE_FPS  = 30


def _style_ax(ax):
    ax.set_facecolor(_PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(_SPINE_COL)
    ax.tick_params(colors=_TICK_COL, labelsize=7)
    ax.xaxis.label.set_color(_TICK_COL)
    ax.yaxis.label.set_color(_TICK_COL)
    ax.title.set_color("#222222")


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                v = v.strip()
                if k == "target_locked":
                    parsed[k] = v not in ("", "0", "False", "false")
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = float("nan")
            rows.append(parsed)
    return rows


def find_video(csv_path: str) -> str | None:
    base = os.path.splitext(csv_path)[0]
    for ext in (".mp4", ".avi", ".mkv"):
        p = base + ext
        if os.path.exists(p):
            return p
    return None


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r.get(key, float("nan")) for r in rows])


def find_hit_idx(rows: list[dict]) -> int:
    """Apply the same two-stage cut as terminal_analyse and return the hit row index."""
    d = {k: col(rows, k) for k in
         ("timestamp_s", "target_locked", "dist_m", "alt_rel_m")}
    d["target_locked"] = col(rows, "target_locked").astype(float)

    first_lock = find_first_lock(d)
    if first_lock is None:
        return len(rows) - 1

    pass_end = find_first_pass_end(d, first_lock)

    d_stage1 = {k: v[:pass_end + 1] for k, v in d.items()}
    lowest   = find_lowest_alt_idx(d_stage1)
    return lowest


def main():
    ap = argparse.ArgumentParser(description="Replay tracking video with CSV overlay")
    ap.add_argument("csv",     help="tracking_<ts>.csv file")
    ap.add_argument("video",   nargs="?", help="video file (auto-detected if omitted)")
    ap.add_argument("--speed", type=float, default=1.0,  help="playback speed (default 1.0)")
    ap.add_argument("--trail", type=int,   default=150,  help="CSV rows shown in plots (default 150)")
    ap.add_argument("--save",  metavar="OUT.mp4",        help="record the replay window to this file")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        print("Empty CSV"); sys.exit(1)

    t_abs   = col(rows, "timestamp_s")
    t_rel   = t_abs - t_abs[0]
    total   = len(rows)

    # ── detect hit (stop point) ───────────────────────────────────────────────
    hit_idx = find_hit_idx(rows)
    hit_t   = t_rel[hit_idx]
    print(f"[HIT]  row {hit_idx}/{total-1}  t={hit_t:.2f}s  "
          f"dist={rows[hit_idx].get('dist_m', float('nan')):.1f}m  "
          f"alt={rows[hit_idx].get('alt_rel_m', float('nan')):.1f}m")

    video_path = args.video or find_video(args.csv)
    cap = None
    vid_fps = 30.0
    vid_frames = 0
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        vid_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[VIDEO] {video_path}  {vid_frames} frames @ {vid_fps:.1f} fps")
    else:
        print("[WARN] No video found — CSV data only")

    # ── output recorder (ffmpeg pipe, opened on first frame) ─────────────────
    if args.save:
        save_path = args.save
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"replay_{ts}.mp4"

    save_state = {"proc": None, "path": save_path}

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8), facecolor=_DARK_BG)
    try:
        fig.canvas.manager.set_window_title("Tracking Replay")
    except Exception:
        pass
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.50, wspace=0.35,
                            left=0.06, right=0.97, top=0.92, bottom=0.07)

    ax_vid  = fig.add_subplot(gs[:2, 0])
    ax_err  = fig.add_subplot(gs[0, 1])
    ax_att  = fig.add_subplot(gs[1, 1])
    ax_srv  = fig.add_subplot(gs[2, 0])
    ax_misc = fig.add_subplot(gs[2, 1])

    for ax in (ax_vid, ax_err, ax_att, ax_srv, ax_misc):
        _style_ax(ax)

    ax_vid.set_title("Camera Feed", fontsize=9)
    ax_vid.axis("off")
    blank  = np.full((480, 640, 3), 245, dtype=np.uint8)
    im_vid = ax_vid.imshow(blank, aspect="auto")

    ax_err.set_title("Tracking Error  (normalised ±1)", fontsize=8)
    ax_err.set_ylim(-1.35, 1.35)
    ax_err.axhline(0, color=_SPINE_COL, lw=0.5)
    ln_ex, = ax_err.plot([], [], color="#0077cc", lw=1.1, label="errorx")
    ln_ey, = ax_err.plot([], [], color="#dd2222", lw=1.1, label="errory")
    ax_err.legend(fontsize=6, facecolor="#eeeeee", labelcolor="#222222", loc="upper right")

    ax_att.set_title("Roll / Pitch  (deg)", fontsize=8)
    ax_att.axhline(0, color=_SPINE_COL, lw=0.5)
    ln_roll,  = ax_att.plot([], [], color="#cc8800", lw=1.1, label="roll")
    ln_pitch, = ax_att.plot([], [], color="#228822", lw=1.1, label="pitch")
    ax_att.legend(fontsize=6, facecolor="#eeeeee", labelcolor="#222222", loc="upper right")

    ax_srv.set_title("Aileron / Elevator  (normalised)", fontsize=8)
    ax_srv.set_ylim(-1.15, 1.15)
    ax_srv.axhline(0, color=_SPINE_COL, lw=0.5)
    ln_ail, = ax_srv.plot([], [], color="#cc6600", lw=1.1, label="aileron")
    ln_ele, = ax_srv.plot([], [], color="#8833cc", lw=1.1, label="elevator")
    ax_srv.legend(fontsize=6, facecolor="#eeeeee", labelcolor="#222222", loc="upper right")

    ax_misc.set_title("Alt / Speed / Dist", fontsize=8)
    ax_misc2 = ax_misc.twinx()
    ax_misc2.tick_params(colors=_TICK_COL, labelsize=7)
    for sp in ax_misc2.spines.values():
        sp.set_color(_SPINE_COL)
    ln_alt,  = ax_misc.plot([], [],  color="#009977", lw=1.1, label="alt_m")
    ln_spd,  = ax_misc.plot([], [],  color="#cc2222", lw=1.1, label="spd_ms")
    ln_dist, = ax_misc2.plot([], [], color="#4444cc", lw=1.0, ls="--", label="dist_m")
    ax_misc.legend(fontsize=6,  facecolor="#eeeeee", labelcolor="#222222", loc="upper left")
    ax_misc2.legend(fontsize=6, facecolor="#eeeeee", labelcolor="#222222", loc="upper right")

    # static hit-line on each plot axis
    for ax_l in (ax_err, ax_att, ax_srv, ax_misc):
        ax_l.axvline(hit_t, color="#0088aa", lw=0.9, ls=":", alpha=0.8, label="hit")

    hud = fig.text(0.50, 0.965, "", ha="center", va="top",
                   color="#222222", fontsize=8, family="monospace")

    # ── Animation state ───────────────────────────────────────────────────────
    state = {"wall_start": None, "idx": 0}

    def _grab_frame() -> np.ndarray:
        """Return BGR frame via PNG round-trip — reliable across all backends."""
        import io
        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=fig.dpi, facecolor=fig.get_facecolor())
        bio.seek(0)
        arr = np.frombuffer(bio.getvalue(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _write_frame(bgr: np.ndarray):
        if save_state["proc"] is None:
            h, w = bgr.shape[:2]
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(_SAVE_FPS),
                "-i", "pipe:0",
                "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                save_state["path"],
            ]
            save_state["proc"] = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[SAVE] recording → {save_state['path']}")
        try:
            save_state["proc"].stdin.write(bgr.tobytes())
        except BrokenPipeError:
            pass

    def _update_video(idx: int, elapsed_s: float):
        if cap is None:
            return
        target_frame = min(int(elapsed_s * vid_fps), vid_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        im_vid.set_data(rgb)
        im_vid.set_extent([0, w, h, 0])
        ax_vid.set_xlim(0, w)
        ax_vid.set_ylim(h, 0)

    def _render_frame(elapsed: float, idx: int):
        """Update all plot elements for a given elapsed time and CSV index."""
        start = max(0, idx - args.trail)
        sl    = rows[start:idx + 1]
        t_sl  = t_rel[start:idx + 1]

        _update_video(idx, elapsed)

        def _set(ln, key):
            ln.set_data(t_sl, col(sl, key))

        _set(ln_ex,    "errorx");  _set(ln_ey,   "errory")
        _set(ln_roll,  "roll_deg"); _set(ln_pitch,"pitch_deg")
        _set(ln_ail,   "aileron");  _set(ln_ele,  "elevator")
        _set(ln_alt,   "alt_rel_m"); _set(ln_spd, "groundspeed_ms")
        _set(ln_dist,  "dist_m")

        t0_sl = t_sl[0] if len(t_sl) else 0
        t1_sl = max(t_sl[-1], t0_sl + 1) if len(t_sl) else 1

        for ax_l in (ax_err, ax_srv):
            ax_l.set_xlim(t0_sl, t1_sl)
        for ax_l in (ax_att, ax_misc):
            ax_l.set_xlim(t0_sl, t1_sl)
            ax_l.relim(); ax_l.autoscale_view()
        ax_misc2.relim(); ax_misc2.autoscale_view()

        row    = rows[idx]
        locked = row.get("target_locked", False)
        ex     = row.get("errorx", float("nan"))
        ey     = row.get("errory", float("nan"))
        ex_s   = f"{ex:+.3f}" if not np.isnan(ex) else "  ----"
        ey_s   = f"{ey:+.3f}" if not np.isnan(ey) else "  ----"
        lock_s = "LOCKED" if locked else " LOST "
        hud.set_text(
            f"t={t_rel[idx]:7.2f}s  |  err=({ex_s}, {ey_s})  |  "
            f"{lock_s}  |  dist={row.get('dist_m', 0):.0f}m  "
            f"alt={row.get('alt_rel_m', 0):.1f}m  "
            f"spd={row.get('groundspeed_ms', 0):.1f}m/s"
        )
        hud.set_color("#007744" if locked else "#cc2222")
        if idx >= hit_idx:
            hud.set_text(hud.get_text() + "  [HIT]")
            hud.set_color("#cc2222")

    def _close_save():
        proc = save_state["proc"]
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        save_state["proc"] = None
        print(f"[SAVE] done → {save_state['path']}")

    if args.save:
        # ── Offline render: step at exactly 1/_SAVE_FPS so playback speed is correct
        total_frames = int(hit_t * _SAVE_FPS) + 1
        print(f"[SAVE] rendering {total_frames} frames at {_SAVE_FPS} fps …")
        for fn in range(total_frames):
            elapsed = fn / _SAVE_FPS
            idx = min(int(np.searchsorted(t_rel, elapsed)), hit_idx)
            _render_frame(elapsed, idx)
            _write_frame(_grab_frame())
            if fn % _SAVE_FPS == 0:
                print(f"[SAVE]   {elapsed:.1f}s / {hit_t:.1f}s", end="\r")
        print()
        _close_save()
        plt.close(fig)
    else:
        # ── Live playback via FuncAnimation ──────────────────────────────────
        def animate(_frame_num):
            now = time.monotonic()
            if state["wall_start"] is None:
                state["wall_start"] = now
            elapsed = (now - state["wall_start"]) * args.speed
            idx = min(int(np.searchsorted(t_rel, elapsed)), hit_idx)
            state["idx"] = idx
            _render_frame(elapsed, idx)
            if idx >= hit_idx:
                ani.event_source.stop()
            return (im_vid, ln_ex, ln_ey, ln_roll, ln_pitch,
                    ln_ail, ln_ele, ln_alt, ln_spd, ln_dist, hud)

        interval_ms = max(16, int(1000 / (_SAVE_FPS * args.speed)))
        ani = FuncAnimation(fig, animate, interval=interval_ms,
                            blit=False, cache_frame_data=False)
        plt.show()

    if cap:
        cap.release()


if __name__ == "__main__":
    main()
