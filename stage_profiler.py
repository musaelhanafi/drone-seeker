"""stage_profiler.py — lightweight per-stage timing for the seeker loop.

Accumulates wall-clock cost of each loop stage (capture → track → control →
hud → display) and prints a periodic summary: mean ms/stage, % of frame, and
implied FPS. Near-zero overhead when disabled (a couple of attribute checks).

Usage in a frame loop:

    prof = StageProfiler(enabled=args.profile)
    while True:
        prof.begin()
        frame = read()            ; prof.lap("capture")
        out   = track(frame)      ; prof.lap("track")
        prof.note("  detect",  seeker.t_detect_ms)   # sub-stage (inside track)
        prof.note("  tracker", seeker.t_track_ms)    #   not added to the total
        ...                       ; prof.lap("display")
        prof.frame_end()

`lap()` stages are summed into the per-frame total; `note()` sub-stages are
reported indented for attribution but excluded from the total to avoid double
counting (they already live inside a lap'd stage).
"""

import collections
import time


class StageProfiler:
    def __init__(self, enabled=True, period_s=2.0, logfile=None):
        self.enabled = enabled
        self.period_s = period_s
        self.logfile = logfile          # path to append reports to (or None)
        self._fh = None                 # lazily-opened file handle
        self._acc = collections.OrderedDict()    # name -> summed seconds (top-level)
        self._notes = collections.OrderedDict()  # name -> summed ms (sub-stage)
        self._frames = 0
        self._t = 0.0
        self._last_report = time.perf_counter()

    def begin(self):
        if not self.enabled:
            return
        self._t = time.perf_counter()

    def lap(self, name):
        if not self.enabled:
            return
        now = time.perf_counter()
        self._acc[name] = self._acc.get(name, 0.0) + (now - self._t)
        self._t = now

    def note(self, name, ms):
        """Record a sub-stage already measured (in ms) elsewhere, e.g. inside track()."""
        if not self.enabled:
            return
        self._notes[name] = self._notes.get(name, 0.0) + float(ms)

    def frame_end(self):
        if not self.enabled:
            return
        self._frames += 1
        now = time.perf_counter()
        if now - self._last_report >= self.period_s and self._frames:
            self._report()
            self._acc.clear()
            self._notes.clear()
            self._frames = 0
            self._last_report = now

    def _report(self):
        n = self._frames
        total_ms = sum(self._acc.values()) / n * 1000.0
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        stamp = time.strftime("%H:%M:%S")
        lines = ["[PROF %s] %3d frames  %6.2f ms/frame  ->  %4.1f FPS"
                 % (stamp, n, total_ms, fps)]
        for name, sec in self._acc.items():
            ms = sec / n * 1000.0
            pct = 100.0 * ms / total_ms if total_ms else 0.0
            lines.append("   %-14s %7.2f ms  %5.1f%%" % (name, ms, pct))
        for name, ms_sum in self._notes.items():
            lines.append("   %-14s %7.2f ms  (sub)" % (name, ms_sum / n))
        text = "\n".join(lines)
        print(text, flush=True)
        if self.logfile:
            if self._fh is None:
                self._fh = open(self.logfile, "a", buffering=1)   # line-buffered
                self._fh.write("# pipeline profile — started %s\n"
                               % time.strftime("%Y-%m-%d %H:%M:%S"))
            self._fh.write(text + "\n")

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
