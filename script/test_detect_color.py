#!/usr/bin/env python3
"""
Detect + track test harness with recording — same vision pipeline as main.py, no MAVLink.

This drives the Seeker vision engine directly (the same object SeekerCtrl wraps
in main.py) and runs its built-in detect → track → display loop. It mirrors
main.py's source / tracker / detection options so behaviour is identical to
production; only the flight-control / MAVLink layer is omitted.

Uses the CALIBRATED colour detector: it loads color_histogram.txt (produced by
calibrate_color.py / app_calibrate.py), fits a Gaussian to the target hue and
detects with a confidence back-projection + nearest-blob selection + box filter +
Kalman coasting. Video-file sources play at their native FPS (the Seeker capture
thread paces file reads); camera / UDP sources run flat-out. --record tees the
annotated frames to tracking_<timestamp>.mp4 (constant 30 fps, real-time playback).

Usage:
    python3 script/app_calibrate.py --source video.mp4     # make color_histogram.txt
    python3 script/test_detect_color.py --source video.mp4 --tracker mil,kalman --record
    python3 script/test_detect_color.py --udpsrc 5600 --udpsrc-codec h264

Controls (from Seeker.run):
    q   quit
    r   reset the tracker
"""

from __future__ import annotations
import argparse
import collections
import sys
import time
from pathlib import Path

import cv2

# Import the Seeker vision engine + recorder from the project root (one level up).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seeker import Seeker           # noqa: E402
from recorder import VideoRecorder  # noqa: E402

_SHIFT_ALGOS         = {"camshift", "meanshift"}
_APPEARANCE_TRACKERS = {"mil", "kcf"}


def _parse_tracker_opt(value: str) -> tuple[bool, str, bool, str]:
    """Parse '--tracker' into (use_camshift, shift_algo, use_kalman, tracker_name).

    Tokens (comma-separated):
      camshift / meanshift  — shift-based tracker variant (mutually exclusive)
      kalman                — enable Kalman filter
      mil / kcf             — appearance tracker (mutually exclusive with each
                              other AND with shift). KCF is ~5–10× faster than
                              MIL on typical footage but weaker on scale change.
    Examples:
      camshift,kalman   → CamShift + Kalman  (default)
      meanshift,kalman  → MeanShift + Kalman
      mil               → MIL tracker, no Kalman
      mil,kalman        → MIL tracker + Kalman
      kcf,kalman        → KCF tracker + Kalman  (fastest appearance path)
    """
    tokens = {t.strip().lower() for t in value.split(",") if t.strip()}
    _VALID = {"kalman"} | _SHIFT_ALGOS | _APPEARANCE_TRACKERS
    unknown = tokens - _VALID
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown tracker token(s): {', '.join(sorted(unknown))}. "
            f"Valid: kalman, {', '.join(sorted(_SHIFT_ALGOS))}, "
            f"{', '.join(sorted(_APPEARANCE_TRACKERS))}"
        )
    shift = tokens & _SHIFT_ALGOS
    if len(shift) > 1:
        raise argparse.ArgumentTypeError(
            "Cannot combine 'camshift' and 'meanshift' — they are mutually exclusive."
        )
    appearance = tokens & _APPEARANCE_TRACKERS
    if len(appearance) > 1:
        raise argparse.ArgumentTypeError(
            f"Cannot combine {' and '.join(repr(t) for t in sorted(appearance))} "
            f"— appearance trackers are mutually exclusive."
        )
    tracker_name = next(iter(appearance), "")
    shift_algo   = next(iter(shift), "camshift")
    use_camshift = bool(shift) and not tracker_name
    use_kalman   = "kalman" in tokens
    if tracker_name and shift:
        raise argparse.ArgumentTypeError(
            f"Cannot combine '{tracker_name}' with '{shift_algo}' — they are mutually exclusive."
        )
    return use_camshift, shift_algo, use_kalman, tracker_name


def _parse_source(value: str) -> int | str:
    """Return an int for camera index, or str for file path / pipeline."""
    try:
        return int(value)
    except ValueError:
        return value


def _build_udpsrc_pipeline(port: int, codec: str) -> str:
    """GStreamer pipeline receiving an RTP stream on `port`, decoding on the CPU
    (the Raspberry Pi has no hardware video decoder) and handing BGR frames to
    OpenCV via appsink. drop=1 max-buffers=1 keeps only the freshest frame.
    Ported from main.py so the harness consumes the same UDP source as the seeker."""
    if codec == "mjpeg":
        return (
            f"udpsrc port={port} "
            "! application/x-rtp,encoding-name=JPEG "
            "! rtpjpegdepay ! jpegdec ! videoconvert "
            "! appsink drop=1 max-buffers=1"
        )
    return (
        f"udpsrc port={port} "
        "! application/x-rtp,payload=96 "
        "! rtph264depay ! avdec_h264 ! videoconvert "
        "! appsink drop=1 max-buffers=1"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect + track test with recording — same pipeline as main.py, no MAVLink",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", default="0",
                        help="Camera index (e.g. 0), video file path, or GStreamer "
                             "pipeline string. Overridden by --udpsrc.")
    parser.add_argument("--udpsrc", type=int, default=None, metavar="PORT",
                        help="Receive camera stream from a UDP port via GStreamer "
                             "(e.g. --udpsrc 5600). Overrides --source.")
    parser.add_argument("--udpsrc-codec", default="h264", choices=["h264", "mjpeg"],
                        metavar="CODEC",
                        help="Codec for --udpsrc: h264 (default) or mjpeg.")
    parser.add_argument("--res", type=int, nargs=2, default=None, metavar=("W", "H"),
                        help="Request this capture resolution (e.g. --res 1280 720)")
    parser.add_argument("--outres", type=int, nargs=2, default=None, metavar=("W", "H"),
                        help="Scale each frame to this size (e.g. --outres 640 360). "
                             "Applied BEFORE --crop; downscaling cuts CPU/heat.")
    parser.add_argument("--histogram", action="store_true", default=False,
                        help="Show the calibration histogram in a separate window")
    parser.add_argument("--mask", action="store_true", default=False,
                        help="Show the detection mask in a separate window")
    parser.add_argument("--mask-algo", default="all",
                        choices=["gaussian", "adaptive", "inrange", "all"],
                        help="Detection mask algorithm: gaussian, adaptive, inrange, "
                             "or all (2-of-3 vote, default)")
    parser.add_argument("--no-display", action="store_true", default=False,
                        help="Headless: no OpenCV preview window (also disables the "
                             "histogram/mask windows). --record still writes the mp4.")
    parser.add_argument("--threads", type=int, default=None, metavar="N",
                        help="Cap CPU threads (OpenCV/BLAS) to N to flatten peak "
                             "current (e.g. --threads 2). Default: uncapped.")
    parser.add_argument("--tracker", default="camshift,kalman", metavar="TOKENS",
                        help="Comma-separated tracking components (default: "
                             "camshift,kalman). Tokens: camshift  meanshift  mil  "
                             "kalman. Examples: 'camshift,kalman'  'meanshift,kalman'  "
                             "'mil'  'mil,kalman'.")
    parser.add_argument("--no-box-filter", action="store_true", default=False,
                        help="Disable the box-like shape filter; accept any blob "
                             "regardless of shape")
    parser.add_argument("--crop", type=str, nargs=4, default=None,
                        metavar=("X", "Y", "W", "H"),
                        help="Crop each frame to this ROI (e.g. --crop 320 180 640 360). "
                             "Use - for W or H to mean 'rest of dimension after offset'.")
    parser.add_argument("--flip", action="store_true", default=False,
                        help="Flip captured frames 180° (both axes) for upside-down "
                             "cameras. Applied at libcamera level for Picamera2 sources "
                             "and via cv2.flip(-1) for cv2/UDP/file sources.")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-stage loop timing (capture / detect / tracker / display)")
    parser.add_argument("--record", action="store_true",
                        help="Record the annotated output to tracking_<timestamp>.mp4")
    return parser.parse_args()


def _run_with_recording(seeker: Seeker):
    """Mirror Seeker.run()'s detect → track → display loop, but tee each annotated
    frame to a VideoRecorder. Recording lives in the harness (not Seeker) so the
    production vision engine stays record-free. The recorder locks writes to the
    wall clock (dup/drop to a constant 30 fps) so playback is real-time."""
    seeker.open()
    recorder = None
    started = False
    prev_time = time.time()
    frame_times = collections.deque(maxlen=30)
    try:
        while True:
            curr_time = time.time()
            frame_times.append(curr_time - prev_time)
            prev_time = curr_time
            fps = 1.0 / (sum(frame_times) / len(frame_times))

            _seq, ok, frame = seeker.read_frame()
            if frame is None:
                # None before any frame = capture thread warming up; None *after*
                # frames have flowed = end of a video file → stop (else spin forever).
                if started:
                    print("[Seeker] End of stream.")
                    break
                continue
            if not ok:
                print("[Seeker] End of stream.")
                break
            started = True

            annotated, cx, cy = seeker.track(frame)
            errorx, errory    = seeker.error_xy(cx, cy, frame.shape)
            if errorx is not None:
                cv2.putText(annotated, f"ex={errorx:+.3f}  ey={errory:+.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 233, 0), 2)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 233, 0), 2)

            if recorder is None:
                recorder = VideoRecorder()
                recorder.open(annotated.shape, 30.0)
            recorder.write(annotated)

            if seeker.display:
                cv2.imshow(seeker.window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0xFF
            if key == ord("q"):
                print("[Seeker] Quit.")
                break
            elif key == ord("r"):
                seeker._track_win    = None
                seeker._detect_count = 0
                seeker._win_w_ema    = 0.0
                seeker._win_h_ema    = 0.0
                print("[Seeker] Tracker reset.")
    finally:
        if recorder is not None:
            recorder.close()
        seeker.close()


def main():
    args = parse_args()

    if args.threads:
        cv2.setNumThreads(args.threads)
        print(f"[test] thread cap = {args.threads} (OpenCV/BLAS)")

    if args.udpsrc is not None:
        source = _build_udpsrc_pipeline(args.udpsrc, args.udpsrc_codec)
    else:
        source = _parse_source(args.source)

    try:
        use_camshift, shift_algo, use_kalman, tracker_name = _parse_tracker_opt(args.tracker)
    except argparse.ArgumentTypeError as e:
        print(f"error: --tracker: {e}")
        raise SystemExit(1)

    seeker = Seeker(
        source=source,
        capture_width=args.res[0] if args.res else None,
        capture_height=args.res[1] if args.res else None,
        crop=tuple(None if v == '-' else int(v) for v in args.crop) if args.crop else None,
        outres=tuple(args.outres) if args.outres else None,
        show_histogram=args.histogram,
        show_mask=args.mask,
        mask_algo=args.mask_algo,
        use_camshift=use_camshift,
        shift_algo=shift_algo,
        box_filter=not args.no_box_filter,
        use_kalman=use_kalman,
        tracker=tracker_name,
        flip=args.flip,
        display=not args.no_display,
    )
    seeker.profile = args.profile

    try:
        if args.record:
            _run_with_recording(seeker)
        else:
            seeker.run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
