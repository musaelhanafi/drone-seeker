# Camera over LAN — OBS Virtual Camera → GStreamer → drone-seeker

Stream the OBS virtual camera from one laptop (GCS / camera laptop) to another
laptop running `drone-seeker` over a local network using GStreamer UDP.

---

## Architecture

```
[GCS Laptop]                            [Seeker Laptop]
 OBS Studio                              drone-seeker (main.py)
  └─ Virtual Camera (/dev/video10)            └─ cv2.VideoCapture(pipeline)
      └─ gst-launch sender  ──UDP:5600──►  gst-launch receiver (GStreamer)
```

---

## Requirements

Both laptops must have GStreamer installed:

```bash
sudo apt install \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav
```

OpenCV on the seeker laptop must be built with GStreamer support. Verify:

```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer
# Expected: GStreamer:                   YES
```

---

## Step 1 — GCS Laptop: start OBS virtual camera

Launch OBS with virtual camera auto-start:

```bash
~/.local/bin/obs-virtual-cam.sh
```

Verify `/dev/video10` exists:

```bash
v4l2-ctl --list-devices | grep -A2 "OBS Virtual"
```

---

## Step 2 — GCS Laptop: stream /dev/video10 over UDP

Replace `<SEEKER_IP>` with the seeker laptop's IP address.

```bash
gst-launch-1.0 \
    v4l2src device=/dev/video10 \
    ! video/x-raw,framerate=30/1 \
    ! videoconvert \
    ! x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast \
    ! rtph264pay config-interval=1 pt=96 \
    ! udpsink host=<SEEKER_IP> port=5600
```

**Lower-latency alternative (no transcoding, raw MJPEG):**

```bash
gst-launch-1.0 \
    v4l2src device=/dev/video10 \
    ! video/x-raw,framerate=30/1 \
    ! videoconvert \
    ! jpegenc quality=85 \
    ! rtpjpegpay \
    ! udpsink host=<SEEKER_IP> port=5600
```

---

## Step 3 — Seeker Laptop: run drone-seeker with GStreamer source

Use `--udpsrc PORT` to receive a UDP stream. The pipeline is built automatically.

**H.264 (matches sender step 2 default):**

```bash
python3 main.py \
    --udpsrc 5600 \
    --connection udpin:0.0.0.0:14560
```

**MJPEG (matches low-latency alternative):**

```bash
python3 main.py \
    --udpsrc 5600 --udpsrc-codec mjpeg \
    --connection udpin:0.0.0.0:14560
```

`--udpsrc` overrides `--source` and sets `drop=1 max-buffers=1` automatically to
keep the pipeline at the latest frame and prevent buffer buildup.

**Manual pipeline (advanced):** pass a full GStreamer pipeline via `--source` if
you need custom parameters:

```bash
python3 main.py \
    --source "udpsrc port=5600 ! application/x-rtp,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=1 max-buffers=1" \
    --connection udpin:0.0.0.0:14560
```

---

## Troubleshooting

### No frames received
- Check firewall on the GCS laptop: `sudo ufw allow 5600/udp`
- Confirm the seeker laptop's IP with `ip addr show`
- Verify `/dev/video10` is streaming: `ffplay /dev/video10`

### OpenCV not built with GStreamer
Install the OpenCV package that includes GStreamer:
```bash
sudo apt install python3-opencv
# Or rebuild OpenCV from source with -DWITH_GSTREAMER=ON
```

### High latency
- Use MJPEG pipeline instead of H.264 (eliminates encoder delay)
- Add `! queue max-size-buffers=1 leaky=downstream` before `appsink` to drop stale frames

### Check LAN connectivity
```bash
# From seeker laptop:
ping <GCS_LAPTOP_IP>
# From GCS laptop, check port is sending:
sudo tcpdump -i any udp port 5600
```

---

## Quick reference

| Parameter | Value |
|-----------|-------|
| OBS virtual camera device | `/dev/video10` |
| Stream port | `5600/udp` |
| Codec | H.264 (default) or MJPEG (low latency) |
| `--udpsrc PORT` | Receive UDP stream (builds pipeline automatically) |
| `--udpsrc-codec` | `h264` (default) or `mjpeg` |
| `--source` (manual) | Full GStreamer pipeline string (advanced) |
