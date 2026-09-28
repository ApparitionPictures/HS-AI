"""Temporally correlated exploration noise.

Colored (1/f^beta) Gaussian noise for the continuous mouse action, and the
same colored noise pushed through Phi^-1 -> Gumbel for the discrete heads via
Gumbel-max sampling.  Per-step marginals are exactly the policy's Gaussian /
categorical distributions, so PPO log-probabilities stay exact, while the
sampled actions are smooth in time (a swing lasts many frames instead of
jittering at 60 Hz).  Reference: Eberhard et al., "Pink Noise Is All You
Need", ICLR 2023.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


def powerlaw_psd_gaussian(beta: float, size: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """(size, n) array of unit-variance Gaussian noise with PSD ~ 1/f^beta along axis 1."""
    if beta <= 0.0:
        return rng.standard_normal((size, n))
    f = np.fft.rfftfreq(n)
    s = f.copy()
    s[0] = s[1] if n > 1 else 1.0
    s = s ** (-beta / 2.0)
    # keep the variance finite: the DC bin gets the lowest non-zero frequency weight
    w = s[1:].copy()
    w[-1] *= (1 + (n % 2)) / 2.0
    sigma = 2 * math.sqrt(np.sum(w ** 2)) / n
    sr = rng.standard_normal((size, len(f))) * s
    si = rng.standard_normal((size, len(f))) * s
    si[:, 0] = 0.0
    if n % 2 == 0:
        si[:, -1] = 0.0
    y = np.fft.irfft(sr + 1j * si, n=n, axis=1) / sigma
    return y


class ColoredNoise:
    def __init__(self, dim: int, beta: float = 1.0, chunk: int = 2048, seed: Optional[int] = None):
        self.dim = dim
        self.beta = float(beta)
        self.chunk = chunk
        self.rng = np.random.default_rng(seed)
        self._buf = np.zeros((dim, 0))
        self._i = 0

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)
        self._buf = np.zeros((self.dim, 0))
        self._i = 0

    def reset(self) -> None:
        self._buf = np.zeros((self.dim, 0))
        self._i = 0

    def sample(self) -> np.ndarray:
        if self._i >= self._buf.shape[1]:
            self._buf = powerlaw_psd_gaussian(self.beta, self.dim, self.chunk, self.rng)
            self._i = 0
        out = self._buf[:, self._i]
        self._i += 1
        return out.astype(np.float32)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    # vectorised erf via math.erf (small arrays only)
    return np.vectorize(math.erf)(x)


class ActionSampler:
    """Samples actions on the CPU from network outputs (batch 1) and returns the
    exact log-probability under the policy's per-step distribution."""

    def __init__(self, n_continuous: int, discrete_sizes: List[int], beta: float = 1.0, scale: float = 1.0,
                 seed: Optional[int] = None):
        self.n_c = n_continuous
        self.sizes = list(discrete_sizes)
        self.scale = float(scale)
        self.noise_c = ColoredNoise(n_continuous, beta, seed=seed)
        self.noise_d = ColoredNoise(int(sum(self.sizes)), beta, seed=None if seed is None else seed + 1)
        self.deterministic = False

    def set_beta(self, beta: float) -> None:
        self.noise_c.set_beta(beta)
        self.noise_d.set_beta(beta)

    def reset(self) -> None:
        self.noise_c.reset()
        self.noise_d.reset()

    def sample(self, mean: np.ndarray, logstd: np.ndarray, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        logstd = np.asarray(logstd, dtype=np.float64).reshape(-1)
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        std = np.exp(logstd)
        if self.deterministic:
            mouse = mean.copy()
        else:
            mouse = mean + std * self.scale * self.noise_c.sample()
        lp = float(np.sum(-0.5 * ((mouse - mean) / std) ** 2 - logstd - 0.5 * math.log(2 * math.pi)))
        disc = np.zeros(len(self.sizes), dtype=np.int64)
        g_all = None
        if not self.deterministic:
            eps = self.noise_d.sample().astype(np.float64)
            u = np.clip(_norm_cdf(eps), 1e-7, 1 - 1e-7)
            g_all = -np.log(-np.log(u))
        off = 0
        for i, n in enumerate(self.sizes):
            lg = logits[off:off + n]
            lg = lg - np.max(lg)
            logp = lg - math.log(np.sum(np.exp(lg)))
            if self.deterministic:
                a = int(np.argmax(lg))
            else:
                a = int(np.argmax(lg + g_all[off:off + n]))
            disc[i] = a
            lp += float(logp[a])
            off += n
        return mouse.astype(np.float32), disc, lp
