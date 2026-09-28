"""GPU-side observation builder: crop -> gray -> resize -> uint8 frame stack.

Everything after the single host->device copy stays on the GPU.  The stacked
observation is kept as uint8 (stack, H, W) so rollouts are compact; the policy
normalizes to float on the fly.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


class FramePreprocessor:
    def __init__(self, size: int, stack: int, gray: bool, crop: List[float], device: str = "cuda"):
        self.size = size
        self.stack = stack
        self.gray = gray
        self.crop = crop
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.channels = 1 if gray else 3
        self.obs = torch.zeros((stack * self.channels, size, size), dtype=torch.uint8, device=self.device)
        self._pinned: List[torch.Tensor] = []
        self._pin_i = 0
        self._shape = None
        self._weights = torch.tensor([0.114, 0.587, 0.299], device=self.device).view(1, 3, 1, 1)  # BGR -> luma
        self.last_frame_gpu: Optional[torch.Tensor] = None

    @property
    def obs_channels(self) -> int:
        return self.stack * self.channels

    def reset(self) -> None:
        self.obs.zero_()

    def _upload(self, frame: np.ndarray) -> torch.Tensor:
        if self.device.type != "cuda":
            return torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
        if self._shape != frame.shape or not self._pinned:
            self._shape = frame.shape
            self._pinned = [torch.empty(frame.shape, dtype=torch.uint8).pin_memory() for _ in range(3)]
        buf = self._pinned[self._pin_i]
        self._pin_i = (self._pin_i + 1) % len(self._pinned)
        buf.copy_(torch.from_numpy(frame))
        return buf.to(self.device, non_blocking=True)

    @torch.no_grad()
    def push(self, frame: np.ndarray) -> torch.Tensor:
        """Add one BGR uint8 frame; returns the current stacked observation (uint8, on device)."""
        h, w = frame.shape[:2]
        l, t, r, b = self.crop
        x0, y0, x1, y1 = int(l * w), int(t * h), int(r * w), int(b * h)
        gpu = self._upload(frame)                     # H W 3 uint8
        gpu = gpu[y0:y1, x0:x1]
        self.last_frame_gpu = gpu
        x = gpu.permute(2, 0, 1).unsqueeze(0).float()  # 1 3 H W
        if self.gray:
            x = (x * self._weights).sum(1, keepdim=True)
        x = F.interpolate(x, size=(self.size, self.size), mode="area")
        x = x.clamp_(0, 255).to(torch.uint8)[0]       # C h w
        c = self.channels
        self.obs[:-c] = self.obs[c:].clone()
        self.obs[-c:] = x
        return self.obs

    @staticmethod
    def to_float(obs_u8: torch.Tensor) -> torch.Tensor:
        return obs_u8.float().div_(255.0)
