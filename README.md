# Drone Seeker

Autonomous pink-object tracker that feeds normalised error signals to an ArduPlane flight controller over MAVLink.
When RC channel 6 is armed and a pink target is locked, the system switches the aircraft into **TRACKING mode** and streams `errorx / errory` corrections at video frame rate.

---

## How It Works

```
Camera / Video
      │
      ▼
  Seeker.track()          ← CamShift on pink HSV blob
      │
      ├─ cx, cy           ← pixel centre of target
      │
      ▼
  error_xy()              ← normalise to [-1, 1] relative to frame centre
      │
      ▼
  SeekerCtrl.run()
      │
      ├─ poll RC ch6      ← non-blocking MAVLink RC_CHANNELS read
      │
      ├─ ch6 HIGH + target locked?
      │       YES → set_mode_tracking()   (ArduPlane custom mode 27)
      │              send_tracking(ex, ey) every frame
      │       NO  → set_mode_loiter()     (mode 5, fallback)
      │
      └─ cv2.imshow()     ← annotated frame with HUD overlay
```

### Pink Detection & CamShift Tracking

1. **Colour mask** — each frame is converted to HSV and thresholded across two bands:
   - Hot-pink / magenta: `H 145–179, S 50–255, V 80–255`
   - Rose / light-pink:  `H 0–10,   S 50–150, V 80–255`
   A morphological open + dilate removes noise.

2. **Acquisition** — the largest pink contour above 400 px² is chosen as the initial ROI.  Its hue histogram is stored.

3. **Tracking** — `cv2.CamShift()` runs a back-projection of the hue histogram every frame, refining a rotated bounding box around the target.  If the window collapses the tracker re-acquires automatically.

4. **Error signal** — the box centre `(cx, cy)` is normalised to `[-1, 1]` relative to the frame centre:
   - `errorx` positive → target is **right** of centre
   - `errory` positive → target is **above** centre

### MAVLink TRACKING Message

Custom ArduPlane message **ID 230**, CRC extra 250.
Fields: `errorx (float32)`, `errory (float32)` — both normalised `[-1, 1]`.
The flight controller scales them by ±3° (`TRACKING_MAX_DELTA_RAD`) to drive roll and pitch.

---

## Files

| File | Description |
|---|---|
| `seeker.py` | `Seeker` class — camera/video capture, pink CamShift tracker, `error_xy()` |
| `seekerctrl.py` | `SeekerCtrl` class — MAVLink connection, RC monitoring, mode management, TRACKING message |
| `main.py` | Entry point — CLI argument parsing |
| `requirements.txt` | Python dependencies |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running

### Basic (webcam, SITL default port)

```bash
python main.py
```

### Specific camera index

```bash
python main.py --source 1
```

### Play a video file instead of live camera

```bash
python main.py --source footage.mp4
```

### Serial connection to a real flight controller

```bash
python main.py --connection /dev/ttyUSB0 --baud 57600 --source 0
```

### TCP connection (e.g. MAVProxy forwarding)

```bash
python main.py --connection tcp:127.0.0.1:5760 --source 0
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--connection` | `udpin:0.0.0.0:14560` | MAVLink connection string |
| `--baud` | `57600` | Baud rate (serial only) |
| `--source` | `0` | Camera index or video file path |

---

## Keyboard Controls (during run)

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Reset CamShift tracker (force re-acquisition) |

---

## HUD Overlay

| Label | Meaning |
|---|---|
| `CH6: <pwm> pwm` | Current RC channel 6 PWM value |
| `Mode: TRACKING` | Aircraft is in TRACKING mode, errors are being sent |
| `Mode: NO TARGET` | CH6 is armed but no pink target is detected |
| `Mode: CH6 OFF` | CH6 switch is below threshold (1700 µs) |
| `ex=±X  ey=±Y` | Normalised horizontal / vertical tracking error |
