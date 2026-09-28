"""Behaviour cloning from recorded human play (same file layout as the spool).

Loss = Gaussian NLL on the mouse velocity + cross-entropy on every key head,
computed over truncated sequences so the GRU learns the same temporal
structure it will use in RL.  Produces checkpoints/bc.pt that ``hsai train
--init`` can start from (and optionally regularise toward via ppo.bc_kl_coef).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

from ..model.policy import Policy
from .spool import load_rollout


def load_recordings(paths: List[Path]) -> List[Dict]:
    out = []
    for p in paths:
        r = load_rollout(p)
        if r is not None and len(r["reward"]) > 8:
            out.append(r)
    return out


def train_bc(policy: Policy, recordings: List[Dict], epochs: int = 20, seq_len: int = 32, batch: int = 32,
             lr: float = 3e-4, device: str = "cuda", log: Callable[[str], None] = print,
             amp: bool = True) -> Dict[str, float]:
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    policy = policy.to(dev).train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    chunks = []
    for r in recordings:
        T = len(r["reward"])
        for s in range(0, T - 4, seq_len):
            e = min(T, s + seq_len)
            chunks.append({k: r[k][s:e] for k in ("img", "state", "mouse", "disc", "first")})
    if not chunks:
        raise RuntimeError("no recorded data")
    log(f"[bc] {len(chunks)} sequences of <= {seq_len} steps from {len(recordings)} files")
    rng = np.random.default_rng(0)
    amp_dtype = torch.bfloat16 if (amp and dev.type == "cuda") else None
    stats = {}
    for ep in range(epochs):
        order = rng.permutation(len(chunks))
        tot, n = 0.0, 0
        t0 = time.perf_counter()
        for s in range(0, len(order), batch):
            b = [chunks[i] for i in order[s:s + batch]]
            L = max(len(c["mouse"]) for c in b)
            def stack(key, dtype=None):
                arr = np.zeros((L, len(b)) + b[0][key].shape[1:], dtype=b[0][key].dtype if dtype is None else dtype)
                for i, c in enumerate(b):
                    arr[: len(c[key]), i] = c[key]
                return torch.from_numpy(arr).to(dev)
            mask = torch.zeros(L, len(b), device=dev)
            for i, c in enumerate(b):
                mask[: len(c["mouse"]), i] = 1.0
            img = stack("img").float().div_(255.0)
            state = stack("state")
            mouse = stack("mouse")
            disc = stack("disc")
            first = stack("first", np.float32)
            h0 = policy.initial_state(len(b), dev)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                mean, logstd, logits, _ = policy.forward_seq(img, state, h0, first)
            logp, _, _ = policy.log_prob(mean.float(), logstd.float(), logits.float(), mouse, disc)
            loss = -(logp * mask).sum() / mask.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); n += 1
        stats = {"epoch": ep + 1, "nll": tot / max(1, n), "epoch_s": time.perf_counter() - t0}
        log(f"[bc] epoch {ep + 1}/{epochs} nll={stats['nll']:.3f} ({stats['epoch_s']:.1f}s)")
    policy.eval()
    return stats
