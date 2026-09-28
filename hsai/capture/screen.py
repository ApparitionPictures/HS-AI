"""High-rate screen capture (DXGI desktop duplication via bettercam/dxcam, mss fallback).

``get_frame()`` blocks until the capture thread has a new frame, which paces the
agent loop at the capture rate without any manual sleeping.
"""
from __future__ import annotations

import sys
import time
from typing import Optional, Tuple

import numpy as np


class ScreenCapture:
    def __init__(self, backend: str = "auto", monitor_index: int = 0, target_fps: int = 60,
                 region: Optional[Tuple[int, int, int, int]] = None):
        self.backend_name = backend
        self.monitor_index = monitor_index
        self.target_fps = target_fps
        self.region = region          # absolute pixel (left, top, right, bottom) or None
        self._cam = None
        self._mss = None
        self.width = 0
        self.height = 0
        self.frames = 0
        self._last_t = 0.0
        self.backend = "none"

    # ---------------------------------------------------------------------
    def start(self) -> "ScreenCapture":
        order = [self.backend_name] if self.backend_name != "auto" else ["bettercam", "dxcam", "mss"]
        last_err: Optional[Exception] = None
        for name in order:
            try:
                if name in ("bettercam", "dxcam"):
                    mod = __import__(name)
                    self._cam = mod.create(output_idx=self.monitor_index, output_color="BGR", max_buffer_len=4)
                    self._cam.start(region=self.region, target_fps=self.target_fps, video_mode=True)
                    frame = self._cam.get_latest_frame()
                    self.height, self.width = frame.shape[:2]
                    self.backend = name
                    return self
                if name == "mss":
                    import mss
                    self._mss = mss.mss()
                    mon = self._mss.monitors[self.monitor_index + 1]
                    if self.region:
                        l, t, r, b = self.region
                        self._mon = {"left": l, "top": t, "width": r - l, "height": b - t}
                    else:
                        self._mon = mon
                    self.width, self.height = self._mon["width"], self._mon["height"]
                    self.backend = "mss"
                    return self
            except Exception as e:  # try next backend
                last_err = e
                self._cam = None
                self._mss = None
        raise RuntimeError(f"no screen capture backend available ({last_err})")

    def stop(self) -> None:
        try:
            if self._cam is not None:
                self._cam.stop()
                try:
                    self._cam.release()
                except Exception:
                    pass
        except Exception:
            pass
        self._cam = None
        self._mss = None

    # ---------------------------------------------------------------------
    def get_frame(self) -> Optional[np.ndarray]:
        """Blocks for the next frame. Returns HxWx3 BGR uint8 (or None on failure)."""
        if self._cam is not None:
            frame = self._cam.get_latest_frame()
            self.frames += 1
            return frame
        if self._mss is not None:
            # mss has no frame pacing: emulate the target rate
            period = 1.0 / self.target_fps
            now = time.perf_counter()
            if self._last_t and now - self._last_t < period:
                time.sleep(max(0.0, period - (now - self._last_t)))
            self._last_t = time.perf_counter()
            shot = self._mss.grab(self._mon)
            self.frames += 1
            return np.asarray(shot)[:, :, :3]
        return None


def monitor_size(monitor_index: int = 0) -> Tuple[int, int]:
    """(width, height) of a monitor in physical pixels."""
    try:
        import bettercam as bc  # type: ignore
        info = bc.output_info()
    except Exception:
        try:
            import dxcam as bc  # type: ignore
            info = bc.output_info()
        except Exception:
            info = ""
    # output_info is a printable string; fall back to win32 metrics
    from ..util.win32 import screen_size
    return screen_size()
