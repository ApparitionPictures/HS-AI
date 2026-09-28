"""Recurrent PPO learner core (device-agnostic; bf16 autocast on CUDA).

One ``update(rollouts)`` call consumes a list of rollout dicts (from the spool)
and performs ``epochs`` passes of minibatch PPO over truncated-BPTT sequences.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PPOCfg
from ..model.policy import Policy
from .gae import compute_gae
from .normalizers import ReturnScaler, RunningMeanStd


@dataclass
class HyperState:
    lr: float
    entropy_discrete: float
    entropy_continuous: float
    beta: float           # exploration noise colour (used by the actor)
    scale: float = 1.0    # exploration std multiplier (actor)

    def to_dict(self) -> Dict[str, float]:
        return {"lr": self.lr, "entropy_discrete": self.entropy_discrete, "entropy_continuous": self.entropy_continuous,
                "beta": self.beta, "scale": self.scale}


class PPOLearner:
    def __init__(self, policy: Policy, cfg: PPOCfg, device: str = "cuda", explore_beta: float = 1.0,
                 compile_model: bool = False, channels_last: bool = True, minibatch_sleep_ms: int = 0,
                 bc_prior: Optional[Policy] = None):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.policy = policy.to(self.device)
        self.channels_last = channels_last and self.device.type == "cuda"
        if self.channels_last:
            self.policy = self.policy.to(memory_format=torch.channels_last)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5, weight_decay=cfg.weight_decay)
        self.hyper = HyperState(lr=cfg.lr, entropy_discrete=cfg.entropy_discrete,
                                entropy_continuous=cfg.entropy_continuous, beta=explore_beta)
        self.updates = 0
        self.total_steps = 0
        self.reward_scaler = ReturnScaler(cfg.gamma)
        self.value_rms = RunningMeanStd()
        self.minibatch_sleep = minibatch_sleep_ms / 1000.0
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp) if self.device.type == "cuda" else None
        self.bc_prior = bc_prior.to(self.device).eval() if bc_prior is not None else None
        self._fwd = self.policy.forward_seq
        if compile_model:
            try:
                self._fwd = torch.compile(self.policy.forward_seq, dynamic=False)
            except Exception:
                self._fwd = self.policy.forward_seq
        self.last_stats: Dict[str, float] = {}

    # ---- hyper-parameters -------------------------------------------------
    def set_lr(self, lr: float) -> None:
        self.hyper.lr = lr
        for g in self.opt.param_groups:
            g["lr"] = lr

    def scheduled_lr(self) -> float:
        c = self.cfg
        if c.lr_decay_updates <= 0:
            return self.hyper.lr
        frac = min(1.0, self.updates / c.lr_decay_updates)
        base = self.hyper.lr
        return c.lr_min + (base - c.lr_min) * 0.5 * (1.0 + math.cos(math.pi * frac)) if base > c.lr_min else base

    def reset_optimizer(self) -> None:
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.hyper.lr, eps=1e-5, weight_decay=self.cfg.weight_decay)

    # ---- data prep --------------------------------------------------------
    def _prepare(self, rollouts: List[Dict[str, Any]]):
        c = self.cfg
        chunks = []
        for r in rollouts:
            T = len(r["reward"])
            if T < 2:
                continue
            rew = r["reward"].astype(np.float32)
            done = r["done"].astype(bool)
            if c.reward_norm:
                self.reward_scaler.update(rew, done)
                rew = rew * self.reward_scaler.scale
            values = r["value"].astype(np.float32)
            adv, ret = compute_gae(rew, values, done, float(r["bootstrap_value"]), c.gamma, c.lam)
            L = c.seq_len
            for s in range(0, T, L):
                e = min(T, s + L)
                if e - s < 4:
                    continue
                chunks.append({k: r[k][s:e] for k in ("img", "state", "mouse", "disc", "logp", "first")} |
                              {"adv": adv[s:e], "ret": ret[s:e], "old_v": values[s:e], "h0": r["h"][s]})
        return chunks

    @staticmethod
    def _stack(chunks: List[Dict[str, Any]], key: str, L: int) -> np.ndarray:
        """Stack variable-length chunks to [L, B, ...] with zero padding; returns padded array."""
        arrs = [c[key] for c in chunks]
        shape = (L, len(arrs)) + tuple(arrs[0].shape[1:])
        out = np.zeros(shape, dtype=arrs[0].dtype)
        for i, a in enumerate(arrs):
            out[: len(a), i] = a
        return out

    # ---- update -----------------------------------------------------------
    def update(self, rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        c = self.cfg
        t0 = time.perf_counter()
        chunks = self._prepare(rollouts)
        if not chunks:
            return {}
        n_steps = int(sum(len(ch["adv"]) for ch in chunks))
        self.total_steps += n_steps
        L = c.seq_len
        all_adv = np.concatenate([ch["adv"] for ch in chunks])
        adv_mean, adv_std = float(all_adv.mean()), float(all_adv.std() + 1e-8)
        all_ret = np.concatenate([ch["ret"] for ch in chunks])
        if c.value_norm:
            self.value_rms.update(all_ret)
        v_mean, v_std = (self.value_rms.mean, self.value_rms.std) if c.value_norm else (0.0, 1.0)
        self.set_lr(self.scheduled_lr())
        n_chunks = len(chunks)
        mb = max(1, n_chunks // c.minibatches)
        stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy_c": 0.0, "entropy_d": 0.0, "kl": 0.0, "clipfrac": 0.0,
                 "bc_kl": 0.0, "n_mb": 0}
        early_stop = False
        rng = np.random.default_rng(self.updates)
        for epoch in range(c.epochs):
            order = rng.permutation(n_chunks)
            for start in range(0, n_chunks, mb):
                idx = order[start:start + mb]
                batch = [chunks[i] for i in idx]
                lens = np.array([len(b["adv"]) for b in batch])
                mask = np.zeros((L, len(batch)), dtype=np.float32)
                for i, l in enumerate(lens):
                    mask[:l, i] = 1.0
                dev = self.device
                img = torch.from_numpy(self._stack(batch, "img", L)).to(dev, non_blocking=True)
                state = torch.from_numpy(self._stack(batch, "state", L)).to(dev, non_blocking=True)
                mouse = torch.from_numpy(self._stack(batch, "mouse", L)).to(dev)
                disc = torch.from_numpy(self._stack(batch, "disc", L)).to(dev)
                old_logp = torch.from_numpy(self._stack(batch, "logp", L)).to(dev)
                first = torch.from_numpy(self._stack(batch, "first", L).astype(np.float32)).to(dev)
                adv = torch.from_numpy(self._stack(batch, "adv", L)).to(dev)
                ret = torch.from_numpy(self._stack(batch, "ret", L)).to(dev)
                old_v = torch.from_numpy(self._stack(batch, "old_v", L)).to(dev)
                h0 = torch.from_numpy(np.stack([b["h0"] for b in batch]).astype(np.float32)).to(dev)
                m = torch.from_numpy(mask).to(dev)
                if c.adv_norm:
                    adv = (adv - adv_mean) / adv_std
                ret_n = (ret - v_mean) / v_std
                old_v_n = (old_v - v_mean) / v_std
                img_f = img.float().div_(255.0)
                if self.channels_last:
                    img_f = img_f.contiguous()
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                    mean, logstd, logits, v = self._fwd(img_f, state, h0, first)
                mean, logstd, logits, v = mean.float(), logstd.float(), logits.float(), v.float()
                logp, ent_c, ent_d = self.policy.log_prob(mean, logstd, logits, mouse, disc)
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - c.clip, 1.0 + c.clip) * adv
                loss_pi = -(torch.min(surr1, surr2) * m).sum() / m.sum()
                v_clipped = old_v_n + torch.clamp(v - old_v_n, -c.value_clip, c.value_clip)
                lv1 = (v - ret_n) ** 2
                lv2 = (v_clipped - ret_n) ** 2
                loss_v = 0.5 * (torch.max(lv1, lv2) * m).sum() / m.sum()
                ent_c_m = (ent_c * m).sum() / m.sum()
                ent_d_m = (ent_d * m).sum() / m.sum()
                loss = loss_pi + c.value_coef * loss_v - self.hyper.entropy_continuous * ent_c_m - self.hyper.entropy_discrete * ent_d_m
                bc_kl_val = 0.0
                if self.bc_prior is not None and c.bc_kl_coef > 0:
                    coef = c.bc_kl_coef * max(0.0, 1.0 - self.updates / max(1, c.bc_kl_decay_updates))
                    if coef > 0:
                        with torch.no_grad():
                            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                                pm, pls, plg, _ = self.bc_prior.forward_seq(img_f, state, h0, first)
                        kl = self._kl(pm.float(), pls.float(), plg.float(), mean, logstd, logits)
                        bc_kl = (kl * m).sum() / m.sum()
                        loss = loss + coef * bc_kl
                        bc_kl_val = float(bc_kl.item())
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    approx_kl = ((old_logp - logp) * m).sum() / m.sum()
                    clipfrac = (((ratio - 1.0).abs() > c.clip).float() * m).sum() / m.sum()
                stats["loss_pi"] += float(loss_pi.item()); stats["loss_v"] += float(loss_v.item())
                stats["entropy_c"] += float(ent_c_m.item()); stats["entropy_d"] += float(ent_d_m.item())
                stats["kl"] += float(approx_kl.item()); stats["clipfrac"] += float(clipfrac.item())
                stats["bc_kl"] += bc_kl_val
                stats["n_mb"] += 1
                if self.minibatch_sleep > 0:
                    time.sleep(self.minibatch_sleep)
                if c.target_kl > 0 and approx_kl.item() > 1.5 * c.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break
        self.updates += 1
        n = max(1, stats.pop("n_mb"))
        out = {k: v / n for k, v in stats.items()}
        out.update({"lr": self.hyper.lr, "steps": float(n_steps), "total_steps": float(self.total_steps),
                    "update_s": time.perf_counter() - t0, "early_stop": float(early_stop),
                    "adv_std": adv_std, "reward_scale": self.reward_scaler.scale, "v_mean": v_mean, "v_std": v_std,
                    "logstd": float(self.policy.logstd.detach().mean().item())})
        self.last_stats = out
        return out

    def _kl(self, pm, pls, plg, m, ls, lg) -> torch.Tensor:
        """KL(prior || policy) for the hybrid distribution, per step."""
        kl_c = (ls - pls + (torch.exp(2 * pls) + (pm - m) ** 2) / (2 * torch.exp(2 * ls)) - 0.5).sum(-1)
        kl_d = torch.zeros_like(kl_c)
        for a, b in zip(self.policy.split_logits(plg), self.policy.split_logits(lg)):
            pa = F.log_softmax(a, -1)
            pb = F.log_softmax(b, -1)
            kl_d = kl_d + (pa.exp() * (pa - pb)).sum(-1)
        return kl_c + kl_d

    # ---- persistence ------------------------------------------------------
    def state_extra(self) -> Dict[str, Any]:
        return {"reward_scaler": self.reward_scaler.state_dict(), "value_rms": self.value_rms.state_dict(),
                "hyper": self.hyper.to_dict(), "updates": self.updates, "total_steps": self.total_steps}

    def load_extra(self, ck: Dict[str, Any]) -> None:
        if "reward_scaler" in ck:
            self.reward_scaler.load_state_dict(ck["reward_scaler"])
        if "value_rms" in ck:
            self.value_rms.load_state_dict(ck["value_rms"])
        if "hyper" in ck:
            h = ck["hyper"]
            self.hyper = HyperState(**h)
            self.set_lr(self.hyper.lr)
        self.updates = int(ck.get("updates", 0))
        self.total_steps = int(ck.get("total_steps", 0))
        if "optimizer" in ck:
            try:
                self.opt.load_state_dict(ck["optimizer"])
            except Exception:
                pass
