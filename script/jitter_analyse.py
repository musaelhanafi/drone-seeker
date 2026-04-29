#!/usr/bin/env python
"""jitter_analyse.py — Analisis data jitter dari jitter_collect.py + rekomendasi PID.

Membaca CSV hasil jitter_collect.py dan menampilkan:
  1. Statistik jitter (mean, std, P50, P95, P99, max)
  2. Histogram distribusi interval antar pesan
  3. Grafik roll/pitch/yaw: raw vs EMA vs moving-average
  4. Jitter over time (interval_ms vs waktu)
  5. Perbandingan RMS noise: raw vs EMA vs MA
  6. Spektrum FFT noise per axis
  7. Rekomendasi parameter PID untuk minimisasi jitter

Usage
-----
    python jitter_analyse.py [jitter_data.csv]
"""

from __future__ import print_function, division

import sys
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── PID tuning thresholds ─────────────────────────────────────────────────────
# Noise RMS (deg) above which D-gain reduction is recommended.
_D_NOISE_THRESHOLD = 0.3
# Rate noise (deg/s) above which D-gain reduction is recommended.
_D_RATE_THRESHOLD  = 5.0
# Oscillation peak prominence (relative to median spectrum) to flag P-gain issue.
_OSC_PROMINENCE    = 4.0
# Frequency band considered "signal" for fixed-wing maneuvers (Hz).
_SIGNAL_BAND_HZ    = (0.1, 3.0)


# ── Load CSV ──────────────────────────────────────────────────────────────────

FLOAT_COLS = [
    "recv_time_s", "msg_time_ms", "interval_ms",
    "roll_raw_deg",  "pitch_raw_deg",  "yaw_raw_deg",
    "roll_ema_deg",  "pitch_ema_deg",  "yaw_ema_deg",
    "roll_ma_deg",   "pitch_ma_deg",   "yaw_ma_deg",
    "roll_rate_raw", "pitch_rate_raw",
    "srv1_raw", "srv2_raw",
]


def load(path):
    data = {k: [] for k in FLOAT_COLS}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in FLOAT_COLS:
                v = row.get(k, "").strip()
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(float("nan"))
    return {k: np.array(v) for k, v in data.items()}


# ── Spectrum analysis ─────────────────────────────────────────────────────────

def analyse_spectrum(x, t):
    """FFT of x sampled at times t.

    Returns (freqs, power, noise_peak_hz, signal_bw_hz, osc_detected).
    """
    mask = np.isfinite(x) & np.isfinite(t)
    x, t = x[mask], t[mask]
    if len(x) < 16:
        return None, None, None, None, False

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None, None, None, None, False

    x_det = x - np.mean(x)
    n     = len(x_det)
    fft   = np.fft.rfft(x_det * np.hanning(n))
    power = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=dt)

    # Signal bandwidth: frequency below which 90% of low-band power sits
    lo, hi = _SIGNAL_BAND_HZ
    band   = (freqs >= lo) & (freqs <= hi)
    sig_bw = hi
    if band.any():
        cum   = np.cumsum(power[band])
        idx90 = np.searchsorted(cum, 0.9 * cum[-1])
        sig_bw = float(freqs[band][min(idx90, band.sum() - 1)])
        sig_bw = max(sig_bw, lo)

    # Noise peak: strongest frequency above signal band
    noise_band = freqs > max(sig_bw * 1.5, hi)
    noise_peak_hz = None
    if noise_band.any():
        noise_peak_hz = float(freqs[noise_band][np.argmax(power[noise_band])])

    # Oscillation: a spike in 2-20 Hz that is _OSC_PROMINENCE x median spectrum
    osc_band = (freqs >= 2.0) & (freqs <= 20.0)
    osc_det  = False
    if osc_band.any():
        p_osc    = power[osc_band]
        med      = float(np.median(power))
        if med > 0 and np.max(p_osc) > _OSC_PROMINENCE * med:
            osc_det = True

    return freqs, power, noise_peak_hz, sig_bw, osc_det


# ── PID recommendations ───────────────────────────────────────────────────────

def recommend_pid(d):
    t = d["recv_time_s"] - d["recv_time_s"][0]
    dt_arr = np.diff(t)
    dt     = float(np.median(dt_arr[dt_arr > 0])) if len(dt_arr) else 0.04
    fs     = 1.0 / dt if dt > 0 else 25.0

    print("=" * 60)
    print("REKOMENDASI PID — minimisasi jitter")
    print("=" * 60)
    print("  Sample rate aktual : %.1f Hz" % fs)
    print()

    for axis, rate_key in (("roll", "roll_rate_raw"), ("pitch", "pitch_rate_raw")):
        raw  = d["%s_raw_deg" % axis]
        ema  = d["%s_ema_deg" % axis]
        rate = d[rate_key]
        mask = np.isfinite(raw) & np.isfinite(ema)

        n_rms     = noise_rms(raw[mask], ema[mask])
        rate_rms  = float(np.sqrt(np.nanmean(rate[np.isfinite(rate)] ** 2)))

        freqs, power, noise_peak_hz, sig_bw, osc_det = analyse_spectrum(
            raw, t
        )

        print("  ── %s ──" % axis.upper())
        print("    Noise RMS (raw-EMA)  : %.4f deg" % n_rms)
        print("    Rate RMS             : %.4f deg/s" % rate_rms)
        if sig_bw is not None:
            print("    Signal bandwidth     : %.2f Hz" % sig_bw)
        if noise_peak_hz is not None:
            print("    Dominant noise freq  : %.2f Hz" % noise_peak_hz)
        print()

        # ── FLTD recommendation ───────────────────────────────────────────────
        if sig_bw is not None:
            fltd = max(2.0, min(sig_bw * 2.0, 10.0))
            print("    [FLTD] Derivative filter cutoff:")
            print("      → RLL2SRV_FLTD / PTCH2SRV_FLTD = %.1f Hz" % fltd)
            print("        (pasang 2x signal bandwidth; atenuasi noise di atas %.1f Hz)" % fltd)

        # ── D-gain recommendation ─────────────────────────────────────────────
        print("    [D-gain]")
        if n_rms > _D_NOISE_THRESHOLD:
            scale = _D_NOISE_THRESHOLD / n_rms
            print("      → Kurangi D gain ~%.0f%%  (noise RMS %.4f deg > threshold %.1f deg)" % (
                (1.0 - scale) * 100, n_rms, _D_NOISE_THRESHOLD))
            print("        Contoh: jika RLL2SRV_D=0.08 → set %.4f" % (0.08 * scale))
        else:
            print("      → D gain OK  (noise RMS %.4f deg <= threshold %.1f deg)" % (
                n_rms, _D_NOISE_THRESHOLD))

        if rate_rms > _D_RATE_THRESHOLD and n_rms > _D_NOISE_THRESHOLD:
            print("      → Rate noise juga tinggi (%.1f deg/s) — pastikan FLTD dipasang sebelum menaikkan D" % rate_rms)

        # ── P-gain recommendation ─────────────────────────────────────────────
        print("    [P-gain]")
        if osc_det:
            print("      → Osilasi terdeteksi di spektrum 2-20 Hz")
            print("        → Kurangi P gain ~10-20%% lalu cek ulang")
            print("        → Atau naikkan FLTD untuk filter sebelum D-term")
        else:
            print("      → Tidak ada osilasi terdeteksi — P gain OK")

        print()

    # ── Timing jitter → scheduler ─────────────────────────────────────────────
    iv = d["interval_ms"]
    iv = iv[np.isfinite(iv) & (iv > 0)]
    if len(iv) > 0:
        p95 = float(np.percentile(iv, 95))
        mean_iv = float(np.mean(iv))
        print("  ── TIMING JITTER ──")
        if p95 > mean_iv * 1.5:
            print("    → Jitter P95=%.1f ms >> mean=%.1f ms" % (p95, mean_iv))
            print("      → Pertimbangkan turunkan SCHED_LOOP_RATE atau kurangi beban GCS stream")
        else:
            print("    → Timing jitter normal (P95/mean = %.2f)" % (p95 / mean_iv))
        print()

    print("  Catatan: rekomendasi di atas bersifat indikatif.")
    print("  Selalu verifikasi dengan flight test setelah perubahan parameter.")
    print("=" * 60)
    print()


# ── Statistik ─────────────────────────────────────────────────────────────────

def rms(x):
    return float(np.sqrt(np.nanmean(x ** 2)))


def noise_rms(raw, filtered):
    """RMS selisih raw - filtered sebagai proxy noise yang dihilangkan."""
    return float(np.sqrt(np.nanmean((raw - filtered) ** 2)))


def print_stats(d):
    intervals = d["interval_ms"]
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]

    print("=" * 55)
    print("STATISTIK JITTER (interval antar pesan ATTITUDE)")
    print("=" * 55)
    print("  Jumlah sampel : %d" % len(intervals))
    print("  Mean          : %.2f ms" % np.mean(intervals))
    print("  Std dev       : %.2f ms" % np.std(intervals))
    print("  Min           : %.2f ms" % np.min(intervals))
    print("  P50 (median)  : %.2f ms" % np.percentile(intervals, 50))
    print("  P95           : %.2f ms" % np.percentile(intervals, 95))
    print("  P99           : %.2f ms" % np.percentile(intervals, 99))
    print("  Max           : %.2f ms" % np.max(intervals))

    dur = d["recv_time_s"][-1] - d["recv_time_s"][0]
    print("  Durasi rekam  : %.1f s" % dur)
    print("  Rate aktual   : %.1f Hz" % (len(intervals) / dur if dur > 0 else 0))

    print()
    print("EFEKTIVITAS FILTER (RMS noise yang dihilangkan)")
    print("-" * 55)
    for axis in ("roll", "pitch", "yaw"):
        raw = d["%s_raw_deg" % axis]
        ema = d["%s_ema_deg" % axis]
        ma  = d["%s_ma_deg"  % axis]
        mask = np.isfinite(raw) & np.isfinite(ema) & np.isfinite(ma)
        if mask.sum() < 2:
            continue
        n_ema = noise_rms(raw[mask], ema[mask])
        n_ma  = noise_rms(raw[mask], ma[mask])
        print("  %-5s  EMA delta RMS=%.4f deg   MA delta RMS=%.4f deg" % (
            axis.capitalize(), n_ema, n_ma))

    print()


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(d, path):
    t = d["recv_time_s"] - d["recv_time_s"][0]
    intervals = d["interval_ms"]
    valid_iv  = intervals[np.isfinite(intervals) & (intervals > 0)]

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle("Analisis Jitter MAVLink — %s" % path, fontsize=13, y=0.99)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ── 1. Histogram interval ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(valid_iv, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
    ax1.axvline(np.mean(valid_iv), color="red", linestyle="--", linewidth=1.2,
                label="mean=%.1f ms" % np.mean(valid_iv))
    ax1.axvline(np.percentile(valid_iv, 95), color="orange", linestyle=":",
                linewidth=1.2, label="P95=%.1f ms" % np.percentile(valid_iv, 95))
    ax1.set_title("Distribusi interval antar pesan")
    ax1.set_xlabel("interval (ms)")
    ax1.set_ylabel("frekuensi")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. Jitter over time ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    mask_iv = np.isfinite(intervals) & (intervals > 0)
    ax2.plot(t[mask_iv], intervals[mask_iv], color="steelblue",
             linewidth=0.6, alpha=0.8, label="interval_ms")
    ax2.axhline(np.mean(valid_iv), color="red", linestyle="--",
                linewidth=1.0, label="mean")
    ax2.axhline(np.percentile(valid_iv, 95), color="orange", linestyle=":",
                linewidth=1.0, label="P95")
    ax2.set_title("Jitter over time")
    ax2.set_xlabel("waktu (s)")
    ax2.set_ylabel("interval (ms)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── 3-5. Roll / Pitch / Yaw: raw vs EMA vs MA ─────────────────────────────
    axes_data = [
        ("roll",  "Roll (deg)"),
        ("pitch", "Pitch (deg)"),
        ("yaw",   "Yaw (deg)"),
    ]
    for col_idx, (axis, ylabel) in enumerate(axes_data):
        ax = fig.add_subplot(gs[1, col_idx])
        raw = d["%s_raw_deg" % axis]
        ema = d["%s_ema_deg" % axis]
        ma  = d["%s_ma_deg"  % axis]
        mask = np.isfinite(raw)
        ax.plot(t[mask], raw[mask], color="grey",   linewidth=0.5, alpha=0.6,
                label="raw")
        ax.plot(t[mask], ema[mask], color="tab:blue", linewidth=1.0,
                label="EMA")
        ax.plot(t[mask], ma[mask],  color="tab:orange", linewidth=1.0,
                linestyle="--", label="MA")
        ax.set_title("%s — raw vs filter" % axis.capitalize())
        ax.set_xlabel("waktu (s)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── 6-8. Noise residual (raw - filtered) per axis ─────────────────────────
    for col_idx, (axis, ylabel) in enumerate(axes_data):
        ax = fig.add_subplot(gs[2, col_idx])
        raw = d["%s_raw_deg" % axis]
        ema = d["%s_ema_deg" % axis]
        ma  = d["%s_ma_deg"  % axis]
        mask = np.isfinite(raw) & np.isfinite(ema) & np.isfinite(ma)
        res_ema = (raw - ema)[mask]
        res_ma  = (raw - ma)[mask]
        ax.plot(t[mask], res_ema, color="tab:blue",   linewidth=0.6,
                alpha=0.8, label="raw-EMA  rms=%.3f" % rms(res_ema))
        ax.plot(t[mask], res_ma,  color="tab:orange", linewidth=0.6,
                alpha=0.8, linestyle="--",
                label="raw-MA  rms=%.3f" % rms(res_ma))
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_title("%s — noise residual" % axis.capitalize())
        ax.set_xlabel("waktu (s)")
        ax.set_ylabel("delta (deg)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── 9-11. FFT noise spectrum per axis ────────────────────────────────────
    for col_idx, (axis, _) in enumerate(axes_data):
        ax = fig.add_subplot(gs[3, col_idx])
        raw  = d["%s_raw_deg" % axis]
        mask = np.isfinite(raw)
        freqs, power, noise_peak_hz, sig_bw, osc_det = analyse_spectrum(
            raw[mask], t[mask]
        )
        if freqs is not None:
            ax.semilogy(freqs, power, color="tab:blue", linewidth=0.8)
            if sig_bw is not None:
                ax.axvline(sig_bw, color="green", linestyle="--", linewidth=1.0,
                           label="sig BW %.1f Hz" % sig_bw)
            if noise_peak_hz is not None:
                ax.axvline(noise_peak_hz, color="red", linestyle=":", linewidth=1.0,
                           label="noise peak %.1f Hz" % noise_peak_hz)
            if osc_det:
                ax.set_title("%s — FFT spectrum  [OSC!]" % axis.capitalize(),
                             color="red")
            else:
                ax.set_title("%s — FFT spectrum" % axis.capitalize())
            ax.set_xlabel("freq (Hz)")
            ax.set_ylabel("power")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, which="both")
        else:
            ax.set_title("%s — FFT (insufficient data)" % axis.capitalize())

    plt.savefig(path.replace(".csv", "_jitter.png"), dpi=120, bbox_inches="tight")
    print("Plot disimpan ke %s" % path.replace(".csv", "_jitter.png"))
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "jitter_data.csv"
    try:
        d = load(path)
    except IOError:
        print("File tidak ditemukan: %s" % path)
        sys.exit(1)

    n = len(d["recv_time_s"])
    if n < 10:
        print("Data terlalu sedikit (%d baris). Perlu minimal 10." % n)
        sys.exit(1)

    print_stats(d)
    recommend_pid(d)
    plot(d, path)


if __name__ == "__main__":
    main()
