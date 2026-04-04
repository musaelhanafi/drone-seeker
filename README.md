# Drone Seeker

Pelacak target hot-pink otonom yang mengirim sinyal error ternormalisasi ke flight controller ArduPlane melalui MAVLink.
Saat RC channel 6 aktif dan target terkunci, sistem memindahkan pesawat ke **mode TRACKING** dan mengalirkan koreksi `errorx / errory` pada kecepatan frame video.

---

## Cara Kerja

```
Kamera → deteksi warna (3 metode + voting) → CamShift + Kalman
  → error_xy [-1,1] → prediktor latensi + PN lead
    → pesan error MAVLink → ArduPlane PID roll/pitch
```

Lihat diagram arsitektur lengkap di [`docs/chart_02_architecture.png`](docs/chart_02_architecture.png).

---

## Kalibrasi Warna (Sebelum Terbang)

```bash
python calibrate_color.py
```

Arahkan kamera ke target hot-pink, tekan **S** untuk menyimpan histogram ke `color_histogram.txt`.
Tanpa file ini, sistem menggunakan fallback HSV bawaan (H 130–173).

---

## Instalasi

```bash
pip install -r requirements.txt
```

---

## Menjalankan

### Dasar (webcam, port SITL default)

```bash
python main.py
```

### Kamera tertentu

```bash
python main.py --source 1
```

### File video

```bash
python main.py --source footage.mp4
```

### Koneksi serial ke flight controller nyata

```bash
python main.py --connection /dev/ttyUSB0 --baud 57600
```

### Koneksi UDP / TCP

```bash
python main.py --connection udp:127.0.0.1:14550
python main.py --connection tcp:127.0.0.1:5760
```

### Mode auto (ikuti misi, masuk TRACKING saat dekat target)

```bash
python main.py --auto --connection /dev/ttyUSB0
```

---

## Semua Argumen

| Argumen | Default | Keterangan |
|---|---|---|
| `--connection` | `udpin:0.0.0.0:14560` | String koneksi MAVLink |
| `--baud` | `57600` | Baud rate (serial saja) |
| `--source` | `0` | Indeks kamera atau path file video |
| `--auto` | off | Mode auto: masuk TRACKING saat dalam 700 m dari target dan di waypoint terakhir; ch6 rendah → STABILIZE |
| `--res W H` | — | Resolusi tangkapan kamera yang diminta (mis. `--res 1280 720`) |
| `--crop X Y W H` | — | Potong setiap frame ke ROI ini setelah tangkapan |
| `--mask-algo` | `all` | Algoritma deteksi: `gaussian`, `adaptive`, `inrange`, atau `all` (voting 2-dari-3) |
| `--no-camshift` | off | Nonaktifkan CamShift; gunakan centroid blob langsung |
| `--no-box-filter` | off | Nonaktifkan filter bentuk blob (extent/solidity/aspect) |
| `--no-prediction` | off | Nonaktifkan prediksi latensi + PN lead |
| `--histogram` | off | Tampilkan histogram kalibrasi di jendela terpisah |
| `--mask` | off | Tampilkan mask deteksi di jendela terpisah |
| `--no-hud-pitch` | off | Sembunyikan tangga pitch pada HUD |
| `--no-hud-yaw` | off | Sembunyikan tape yaw pada HUD |
| `--debug` | off | Log telemetri tracking ke `tracking.csv` saat mode TRACKING |
| `--record` | off | Rekam video beranotasi ke `tracking_<timestamp>.avi` saat mode TRACKING |

---

## Tombol Saat Berjalan

| Tombol | Aksi |
|---|---|
| `q` | Keluar |
| `r` | Reset tracker (paksa re-akuisisi) |

---

## Pipeline Deteksi

Saat tidak terkunci, setiap frame menjalankan tiga metode mask independen lalu memilih dengan **voting mayoritas (≥ 2 dari 3)**:

| Metode | Cara Kerja |
|---|---|
| **Gaussian back-projection** | `calcBackProject` pada histogram kepercayaan (μ ± 3σ) |
| **Adaptive hue threshold** | `adaptiveThreshold` blockSize=11 + LUT gerbang hue |
| **Dual inRange** | Band core (bobot penuh) + outer (bobot setengah), menangani wrap hue |

Setelah 3 deteksi berurutan, **CamShift** mengambil alih dengan filter **Kalman** untuk menghaluskan posisi dan memprediksi saat target hilang sesaat (hingga 5 frame).

Lihat: [`docs/chart_01_detection.png`](docs/chart_01_detection.png) · [`docs/chart_02_detection_state.png`](docs/chart_02_detection_state.png)

---

## MAVLink

Error tracking dikirim sebagai pesan MAVLink dari `seekerctrl.py`:

| Field | Isi |
|---|---|
| `x` | `errorx` — error horizontal ternormalisasi [-1, 1] |
| `y` | `errory` — error vertikal ternormalisasi [-1, 1] |
| `z` | `0.0` |

ArduPlane mengalikan x/y dengan `TRK_MAX_DEG × π/180` lalu menjalankan PID roll dan pitch.

---

## State Machine Mode

**Mode manual** (default):

| Kondisi | Aksi |
|---|---|
| ch6 HIGH + target terkunci | Masuk TRACKING |
| ch6 HIGH → LOW (tepi turun) | Masuk AUTO |

**Mode auto** (`--auto`):

| Kondisi | Aksi |
|---|---|
| ch6 rendah | STABILIZE |
| ch6 tinggi, belum dekat/wp terakhir | AUTO (ikuti misi) |
| ch6 tinggi, < 700 m, wp terakhir, terkunci | TRACKING |
| ch6 tinggi → rendah | STABILIZE |

Lihat: [`docs/chart_02_mode_state.png`](docs/chart_02_mode_state.png) · [`docs/chart_03_tracking_logic.png`](docs/chart_03_tracking_logic.png)

---

## Overlay HUD

| Label | Sumber |
|---|---|
| `MODE` | Mode terbang dari HEARTBEAT |
| `LOCK: ON / OFF / NO TARGET` | Status ch6 dan tracking |
| `v` / `h` / `thr` | Airspeed (km/jam), ketinggian relatif (m), throttle (%) |
| `FPS` | Frame rate rata-rata 30 frame |
| `ex` / `ey` | Error yang diprediksi (setelah PN lead) saat terkunci |
| Tape yaw | Kompas horizontal dari ATTITUDE yaw |
| Tangga pitch | Sudut pitch dari ATTITUDE |

---

## File

| File | Keterangan |
|---|---|
| `seeker.py` | Kelas `Seeker` — tangkapan background thread, deteksi warna, CamShift + Kalman, `error_xy()` |
| `seekerctrl.py` | Kelas `SeekerCtrl` — koneksi MAVLink, polling RC, state machine mode, prediksi PN lead, loop utama |
| `main.py` | Entry point — parsing argumen CLI |
| `calibrate_color.py` | Alat kalibrasi warna — simpan `color_histogram.txt` |
| `hud_display.py` | Overlay HUD — tape yaw, tangga pitch, indikator roll |
| `joystick_handler.py` | Handler joystick (opsional) |
| `tracking_analyse.py` | Analisis CSV log tracking |
| `pid_analyser.py` | Analisis respons PID dari CSV |
| `terminal_analyse.py` | Analisis fase terminal dari CSV |
| `test_camera.py` | Uji tangkapan kamera |
| `test_detect_color.py` | Uji pipeline deteksi warna |
