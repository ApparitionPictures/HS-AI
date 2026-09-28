"""Running statistics used for reward scaling and value-target normalization."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


class RunningMeanStd:
    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        bm, bv, bc = float(x.mean()), float(x.var()), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 1e-8))

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: Dict[str, float]) -> None:
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class ReturnScaler:
    """Scales rewards by the running std of discounted returns (no mean shift),
    as in "Implementation Matters" (Engstrom et al. 2020)."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.rms = RunningMeanStd()
        self.ret = 0.0

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        rets = np.zeros(len(rewards), dtype=np.float64)
        for i, (r, d) in enumerate(zip(rewards, dones)):
            self.ret = self.ret * self.gamma + float(r)
            rets[i] = self.ret
            if d:
                self.ret = 0.0
        self.rms.update(rets)

    @property
    def scale(self) -> float:
        return 1.0 / max(self.rms.std, 1e-3)

    def state_dict(self) -> Dict:
        return {"rms": self.rms.state_dict(), "ret": self.ret}

    def load_state_dict(self, d: Dict) -> None:
        self.rms.load_state_dict(d["rms"])
        self.ret = d.get("ret", 0.0)
