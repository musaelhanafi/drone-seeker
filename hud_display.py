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

    def draw_hud(self, is_enabled, frame, lat, lon, yaw, pitch, roll):
        if is_enabled:
            zero = np.zeros((frame.shape[0], frame.shape[1], 3), dtype="uint8")

            roll = -roll
            cr = math.cos(roll * math.pi / 180)
            sr = math.sin(roll * math.pi / 180)
            rows, cols = frame.shape[0], frame.shape[1]
            xx1, yy1 = self.transform(3 * cols / 4, rows / 2 - self.offsety, roll * math.pi / 180)

            M = np.float32([
                [cr, -sr, -(xx1 - 3 * cols / 4)],
                [sr,  cr, -(yy1 - rows / 2 + self.offsety)],
            ])

            self.draw_center(zero, roll)
            if self.show_pitch:
                self.draw_pitch(zero, pitch)

            center = (int(3 * frame.shape[1] / 4), int(frame.shape[0] / 2 - self.offsety))
            axes   = (int(1.1 * self.size), int(1.1 * self.size))
            color  = (0, 0, 255)

            zero = cv2.warpAffine(zero, M, (cols, rows))

            x  = int(3 * frame.shape[1] / 4)
            y  = int(frame.shape[0] / 2 - self.offsety)

            if self.show_pitch:
                cv2.line(zero,
                         (int(2 * self.size / 3) + x, y),
                         (int(-2 * self.size / 3) + x, y),
                         (0, 255, 255), 2, cv2.LINE_8)
                cv2.ellipse(zero, center, axes, 270, -60, 60, color, 4)
            cv2.addWeighted(frame, 0.6, zero, 0.4, 0.0, frame)

        if self.show_yaw:
            self.draw_yaw(frame, lat, lon, yaw)

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

    def draw_pitch(self, frame, pitch):
        x = int(3 * frame.shape[1] / 4)
        y = int(frame.shape[0] / 2 - self.offsety)

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

    def draw_center(self, frame, roll):
        x = int(3 * frame.shape[1] / 4)
        y = int(frame.shape[0] / 2 - self.offsety)

        cv2.line(frame, (x, y - int(1.1 * self.size)), (x, y),
                 (0, 255, 255), 2, cv2.LINE_8)

        deg = "%3d" % (-roll)
        cv2.putText(frame, deg, (x - 15, y - int(1.2 * self.size)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
