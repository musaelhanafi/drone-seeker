#!/usr/bin/env python
"""jitter_collect.py — Kumpulkan data MAVLink dan terapkan EMA untuk mengurangi jitter.

Rekam ATTITUDE + SERVO_OUTPUT_RAW dengan timestamp penerimaan host, hitung
inter-arrival jitter, dan simpan nilai mentah serta nilai terfilter ke CSV.

Usage
-----
    python jitter_collect.py [options]

Options
-------
    --connection  STRING   MAVLink connection string (default: udpin:0.0.0.0:14560)
    --baud        INT      Baud rate untuk koneksi serial (default: 57600)
    --alpha       FLOAT    Faktor EMA 0..1; lebih kecil = lebih halus (default: 0.2)
    --window      INT      Ukuran window moving-average (default: 5)
    --output      FILE     Nama file CSV output (default: jitter_data.csv)
    --duration    INT      Durasi rekam dalam detik; 0 = sampai Ctrl-C (default: 0)
    --rate        INT      Frekuensi request ATTITUDE ke FC dalam Hz (default: 25)

Output CSV columns
------------------
    recv_time_s        waktu penerimaan host (epoch seconds)
    msg_time_ms        time_boot_ms dari FC
    interval_ms        selisih recv_time antar pesan (jitter proxy)
    roll_raw_deg       roll mentah dari ATTITUDE
    pitch_raw_deg      pitch mentah dari ATTITUDE
    yaw_raw_deg        yaw mentah dari ATTITUDE
    roll_ema_deg       roll setelah EMA
    pitch_ema_deg      pitch setelah EMA
    yaw_ema_deg        yaw setelah EMA
    roll_ma_deg        roll setelah moving-average
    pitch_ma_deg       pitch setelah moving-average
    yaw_ma_deg         yaw setelah moving-average
    roll_rate_raw      rollspeed mentah (deg/s)
    pitch_rate_raw     pitchspeed mentah (deg/s)
    srv1_raw           servo1_raw (us)
    srv2_raw           servo2_raw (us)
"""

from __future__ import print_function, division

import argparse
import collections
import csv
import math
import sys
import time

from pymavlink import mavutil


# ── EMA helper ────────────────────────────────────────────────────────────────

class EMA(object):
    """Exponential Moving Average filter."""

    def __init__(self, alpha):
        self.alpha = alpha
        self._value = None

    def update(self, x):
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self):
        return self._value if self._value is not None else 0.0


# ── Moving-average helper ─────────────────────────────────────────────────────

class MovingAverage(object):
    def __init__(self, window):
        self._buf = collections.deque(maxlen=window)

    def update(self, x):
        self._buf.append(x)
        return sum(self._buf) / len(self._buf)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MAVLink jitter collector")
    p.add_argument("--connection", default="udpin:0.0.0.0:14570")
    p.add_argument("--baud", type=int, default=57600)
    p.add_argument("--alpha", type=float, default=0.2,
                   help="EMA alpha (0..1); lebih kecil = lebih halus")
    p.add_argument("--window", type=int, default=5,
                   help="Ukuran window moving-average")
    p.add_argument("--output", default="jitter_data.csv")
    p.add_argument("--duration", type=int, default=0,
                   help="Durasi rekam (detik); 0 = sampai Ctrl-C")
    p.add_argument("--rate", type=int, default=25,
                   help="Frekuensi request ATTITUDE ke FC (Hz)")
    return p.parse_args()


# ── Koneksi MAVLink ───────────────────────────────────────────────────────────

def connect(connection_string, baud):
    print("Connecting to %s ..." % connection_string)
    master = mavutil.mavlink_connection(connection_string, baud=baud)
    master.wait_heartbeat(timeout=10)
    print("Heartbeat received — sysid=%d compid=%d" % (
        master.target_system, master.target_component))
    return master


def request_message_rate(master, msg_id, rate_hz):
    """Minta FC mengirim pesan msg_id pada rate_hz."""
    interval_us = int(1e6 / rate_hz) if rate_hz > 0 else -1
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        interval_us,
        0, 0, 0, 0, 0,
    )


# ── CSV writer ────────────────────────────────────────────────────────────────

FIELDS = [
    "recv_time_s",
    "msg_time_ms",
    "interval_ms",
    "roll_raw_deg", "pitch_raw_deg", "yaw_raw_deg",
    "roll_ema_deg", "pitch_ema_deg", "yaw_ema_deg",
    "roll_ma_deg",  "pitch_ma_deg",  "yaw_ma_deg",
    "roll_rate_raw", "pitch_rate_raw",
    "srv1_raw", "srv2_raw",
]


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    master = connect(args.connection, args.baud)

    # Request pesan yang dibutuhkan
    request_message_rate(master, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, args.rate)
    request_message_rate(master, mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, args.rate)

    ema_roll  = EMA(args.alpha)
    ema_pitch = EMA(args.alpha)
    ema_yaw   = EMA(args.alpha)

    ma_roll   = MovingAverage(args.window)
    ma_pitch  = MovingAverage(args.window)
    ma_yaw    = MovingAverage(args.window)

    prev_recv  = None
    n_written  = 0
    start_time = time.time()

    print("Menyimpan ke %s  (alpha=%.2f  window=%d)" % (
        args.output, args.alpha, args.window))
    print("Tekan Ctrl-C untuk berhenti.")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        srv1_raw = 0
        srv2_raw = 0

        try:
            while True:
                if args.duration > 0 and (time.time() - start_time) >= args.duration:
                    break

                msg = master.recv_match(
                    type=["ATTITUDE", "SERVO_OUTPUT_RAW"],
                    blocking=True,
                    timeout=1.0,
                )
                if msg is None:
                    continue

                recv_now = time.time()
                mtype    = msg.get_type()

                if mtype == "SERVO_OUTPUT_RAW":
                    srv1_raw = msg.servo1_raw
                    srv2_raw = msg.servo2_raw
                    continue  # hanya tulis saat ATTITUDE diterima

                if mtype != "ATTITUDE":
                    continue

                # ── Hitung interval antar pesan ───────────────────────────────
                if prev_recv is None:
                    interval_ms = 0.0
                else:
                    interval_ms = (recv_now - prev_recv) * 1000.0
                prev_recv = recv_now

                # ── Raw values ────────────────────────────────────────────────
                roll_raw  = math.degrees(msg.roll)
                pitch_raw = math.degrees(msg.pitch)
                yaw_raw   = math.degrees(msg.yaw) % 360.0
                roll_rate = math.degrees(msg.rollspeed)
                pitch_rate= math.degrees(msg.pitchspeed)

                # ── EMA filter ────────────────────────────────────────────────
                roll_ema  = ema_roll.update(roll_raw)
                pitch_ema = ema_pitch.update(pitch_raw)
                yaw_ema   = ema_yaw.update(yaw_raw)

                # ── Moving-average filter ──────────────────────────────────────
                roll_ma   = ma_roll.update(roll_raw)
                pitch_ma  = ma_pitch.update(pitch_raw)
                yaw_ma    = ma_yaw.update(yaw_raw)

                row = {
                    "recv_time_s"   : "%.6f" % recv_now,
                    "msg_time_ms"   : msg.time_boot_ms,
                    "interval_ms"   : "%.3f" % interval_ms,
                    "roll_raw_deg"  : "%.4f" % roll_raw,
                    "pitch_raw_deg" : "%.4f" % pitch_raw,
                    "yaw_raw_deg"   : "%.4f" % yaw_raw,
                    "roll_ema_deg"  : "%.4f" % roll_ema,
                    "pitch_ema_deg" : "%.4f" % pitch_ema,
                    "yaw_ema_deg"   : "%.4f" % yaw_ema,
                    "roll_ma_deg"   : "%.4f" % roll_ma,
                    "pitch_ma_deg"  : "%.4f" % pitch_ma,
                    "yaw_ma_deg"    : "%.4f" % yaw_ma,
                    "roll_rate_raw" : "%.4f" % roll_rate,
                    "pitch_rate_raw": "%.4f" % pitch_rate,
                    "srv1_raw"      : srv1_raw,
                    "srv2_raw"      : srv2_raw,
                }
                writer.writerow(row)
                n_written += 1

                if n_written % 100 == 0:
                    elapsed = time.time() - start_time
                    print("  %d baris | %.0f s | jitter=%.1f ms" % (
                        n_written, elapsed, interval_ms))

        except KeyboardInterrupt:
            pass

    elapsed = time.time() - start_time
    print("\nSelesai: %d baris dalam %.1f detik -> %s" % (
        n_written, elapsed, args.output))


if __name__ == "__main__":
    main()
