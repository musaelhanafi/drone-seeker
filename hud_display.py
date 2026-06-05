import cv2
import math
import numpy as np


class HudDisplay():

    def __init__(self, offsety=0, size=120, use_cuda=False,
                 show_pitch=True, show_yaw=True):
        self.offsety    = offsety
        self.size       = size
        self.use_cuda   = use_cuda
        self.show_pitch = show_pitch
        self.show_yaw   = show_yaw

    @staticmethod
    def transform(x, y, theta):
        ct = math.cos(theta)
        st = math.sin(theta)
        return int(x * ct - y * st), int(x * st + y * ct)

    def draw_hud(self, is_enabled, frame, lat, lon, yaw, pitch, roll,
                 pitch_offset_norm=0.0):
        # The attitude overlay (pitch ladder + roll pointer/label + roll arc) is
        # one unit, all gated by show_pitch (--no-hud-pitch). Skipping it here
        # also avoids the alloc / warpAffine / blend when it's off.
        if is_enabled and self.show_pitch:
            rows, cols = frame.shape[0], frame.shape[1]
            # HUD anchor — centre of the pitch ladder / roll arc.
            cx = int(3 * cols / 4)
            cy = int(rows / 2 - self.offsety)
            # All HUD content lives within ~1.35*size of the anchor, and roll
            # only rotates it about the anchor, so render into that ROI box
            # instead of the whole frame — the zero alloc, warpAffine and blend
            # all shrink ~5-6x at 720p. The blend is masked to HUD pixels only,
            # so the rest of the feed stays full-bright (not dimmed to 60%).
            R   = int(1.35 * self.size) + 30
            rx  = max(0, cx - R);    ry  = max(0, cy - R)
            rx2 = min(cols, cx + R); ry2 = min(rows, cy + R)
            rw  = rx2 - rx;          rh  = ry2 - ry
            if rw > 0 and rh > 0:
                zero = np.zeros((rh, rw, 3), dtype="uint8")
                lx = cx - rx;  ly = cy - ry        # anchor in ROI-local coords

                roll = -roll
                cr = math.cos(roll * math.pi / 180)
                sr = math.sin(roll * math.pi / 180)
                # Rotate about the ROI-local anchor (lx, ly).
                xx1, yy1 = self.transform(lx, ly, roll * math.pi / 180)
                M = np.float32([
                    [cr, -sr, -(xx1 - lx)],
                    [sr,  cr, -(yy1 - ly)],
                ])

                self.draw_center(zero, lx, ly, roll)
                self.draw_pitch(zero, lx, ly, pitch)

                zero = cv2.warpAffine(zero, M, (rw, rh))

                cv2.line(zero,
                         (int(2 * self.size / 3) + lx, ly),
                         (int(-2 * self.size / 3) + lx, ly),
                         (0, 255, 255), 2, cv2.LINE_8)
                axes = (int(1.1 * self.size), int(1.1 * self.size))
                cv2.ellipse(zero, (lx, ly), axes, 270, -60, 60, (0, 0, 255), 4)

                # Masked blend: only HUD pixels are composited (0.6*frame +
                # 0.4*zero). cv2.copyTo with an 8-bit mask is a C++ masked copy
                # straight into the strided frame view — ~12x faster than numpy
                # boolean fancy-indexing (2.9 ms → 0.23 ms at 720p) and
                # pixel-identical. The grayscale of `zero` is nonzero exactly on
                # the HUD pixels, so it doubles as the copy mask.
                roi      = frame[ry:ry2, rx:rx2]
                blended  = cv2.addWeighted(roi, 0.6, zero, 0.4, 0.0)
                hud_mask = cv2.cvtColor(zero, cv2.COLOR_BGR2GRAY)
                cv2.copyTo(blended, hud_mask, roi)

        if self.show_yaw:
            self.draw_yaw(frame, lat, lon, yaw)

        h, w = frame.shape[:2]
        self.draw_center_cross(frame, w, h, pitch_offset_norm)

    def draw_yaw(self, frame, lat, lon, yaw):
        x = int(frame.shape[1] / 2)
        y = int(3 * frame.shape[0] / 4)

        if yaw > 180:
            yaw = yaw - 360
        pp    = int(yaw / 5.0) * 5
        delta = int(yaw - pp)

        latlon = "Location: %7.5f, %8.5f" % (lat, lon)
        cv2.putText(frame, latlon, (x - 120, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for idx in range(15):
            yy = idx - 7
            dd = yy * 5 + pp
            if dd < 0:
                dd = 360 + dd

            deg = "%3d" % dd
            if dd == 0:   deg = "  N "
            if dd == 90:  deg = "  E "
            if dd == 180: deg = "  S "
            if dd == 270: deg = "  W "

            sx  = 10 if (yy * 5 + pp) % 10 != 0 else 15
            xx1 = 15 * yy - delta * 3

            color = (0, 0, 255) if idx == 7 else (0, 255, 0)
            size  = 4           if idx == 7 else 2

            cv2.line(frame, (xx1 + x, y - sx), (xx1 + x, y + sx), color, size, cv2.LINE_8)
            if (yy * 5 + pp) % 20 == 0 or deg in ("  E ", "  W "):
                cv2.putText(frame, deg, (xx1 + x - 20, y + sx + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_pitch(self, frame, x, y, pitch):
        pp    = int(pitch / 5.0) * 5
        rem   = 1 if (int(pitch / 5.0)) % 2 == 0 else 0
        delta = int(pitch - pp)

        for idx in range(7):
            yy = idx - 3
            dd = yy * 5 + pp
            deg = "%3d" % dd

            sx    = 50 if dd == 0 else (15 if dd % 10 == 0 else 10)
            color = (0, 0, 255) if dd == 0 else (0, 255, 0)
            thick = 4           if dd == 0 else 2

            oy = yy * 15 - delta * 3
            cv2.line(frame, (x - sx, y + oy), (x + sx, y + oy), color, thick)
            if idx % 2 == rem:
                cv2.putText(frame, deg, (x + 20, y + oy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    @staticmethod
    def draw_center_cross(frame, w, h, pitch_offset_norm=0.0,
                          box=80, arm=16, color=(0, 0, 233), thickness=3):
        """Corner-bracket crosshair at the effective pitch-offset aim point."""
        cx = w // 2
        cy = h // 2 - int(round(pitch_offset_norm * h / 2))
        for x, y, dx, dy in [
            (cx - box, cy - box, +1, +1),
            (cx + box, cy - box, -1, +1),
            (cx + box, cy + box, -1, -1),
            (cx - box, cy + box, +1, -1),
        ]:
            cv2.line(frame, (x, y), (x + arm * dx, y), color, thickness)
            cv2.line(frame, (x, y), (x, y + arm * dy), color, thickness)
        cs = 24
        cv2.line(frame, (cx - cs, cy), (cx + cs, cy), color, thickness)
        cv2.line(frame, (cx, cy - cs), (cx, cy + cs), color, thickness)

    def draw_center(self, frame, x, y, roll):
        cv2.line(frame, (x, y - int(1.1 * self.size)), (x, y),
                 (0, 255, 255), 2, cv2.LINE_8)

        deg = "%3d" % (-roll)
        cv2.putText(frame, deg, (x - 15, y - int(1.2 * self.size)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
