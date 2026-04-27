"""
test_xplane_test.py
Verifikasi bahwa X-Plane mengirim data UDP ke port 49001.

Cara pakai:
    python test_xplane_test.py

Pastikan X-Plane sudah berjalan dan dikonfigurasi:
    Settings → Net Connections → Data → kirim ke 127.0.0.1:49001
"""

import socket
import struct
import sys

HOST = "0.0.0.0"
PORT = 49005  # ArduPlane HIL receive port (changed from default 49001 to avoid conflict with X-Plane)
TIMEOUT = 5  # detik
MAX_PACKETS = 5


def parse_xplane_data(data: bytes) -> list[dict]:
    """Parse paket DATA* dari X-Plane (format 41 byte per grup)."""
    results = []
    if len(data) < 5 or data[:4] != b"DATA":
        return results

    payload = data[5:]
    num_groups = len(payload) // 36

    for i in range(num_groups):
        chunk = payload[i * 36 : (i + 1) * 36]
        idx = struct.unpack_from("<i", chunk, 0)[0]
        values = struct.unpack_from("<8f", chunk, 4)
        results.append({"index": idx, "values": values})

    return results


def main():
    print(f"[TEST] Menunggu data X-Plane di {HOST}:{PORT} (timeout={TIMEOUT}s)...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    try:
        sock.bind(("0.0.0.0", PORT))
    except OSError as e:
        print(f"[ERROR] Gagal bind port {PORT}: {e}")
        print()
        print("Cek proses yang menggunakan port ini:")
        print(f"  macOS/Linux : lsof -i UDP:{PORT}")
        print(f"  Windows     : netstat -ano | findstr {PORT}")
        sys.exit(1)

    sock.settimeout(TIMEOUT)

    received = 0
    try:
        while received < MAX_PACKETS:
            data, addr = sock.recvfrom(4096)

            if received == 0:
                print(f"[OK] Data diterima dari {addr[0]}:{addr[1]}")
                print(f"     Panjang paket: {len(data)} bytes")
                print(f"     Header (5 byte pertama): {data[:5]}")

            groups = parse_xplane_data(data)
            if groups:
                for g in groups:
                    vals = ", ".join(f"{v:.4f}" for v in g["values"])
                    print(f"     DATA index={g['index']:3d} | {vals}")
            else:
                print(f"     [WARN] Paket tidak dikenali (bukan format DATA*)")
                print(f"     Raw hex: {data[:20].hex()}")

            received += 1
            print()

    except socket.timeout:
        if received == 0:
            print(f"[FAIL] Tidak ada data masuk setelah {TIMEOUT} detik.")
            print()
            print("Kemungkinan penyebab:")
            print("  1. X-Plane belum berjalan")
            print("  2. Port output belum dikonfigurasi ke 49001")
            print("     → X-Plane: Settings → Net Connections → Data")
            print("  3. Firewall memblokir UDP port 49001")
            sys.exit(1)
        else:
            print(f"[OK] Selesai — {received} paket diterima.")

    finally:
        sock.close()

    print(f"[PASS] X-Plane berhasil mengirim data ke port {PORT}.")


if __name__ == "__main__":
    main()
