"""Generalized Advantage Estimation for a single actor stream with episode
boundaries (``done``) and a bootstrap value for the final step."""
from __future__ import annotations

import numpy as np


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, bootstrap_value: float,
                gamma: float, lam: float):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    next_value = bootstrap_value
    for t in range(T - 1, -1, -1):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
        next_value = values[t]
    returns = adv + values.astype(np.float32)
    return adv, returns
