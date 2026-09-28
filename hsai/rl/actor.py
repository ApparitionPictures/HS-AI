"""The actor: runs the policy against the live game at the capture rate,
writes rollouts to the spool and reloads weights published by the learner.
Also used for ``play`` (deterministic, no learning) and for the offline
simulator used in tests."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from ..config import Config
from ..env.actions import ActionSpace
from ..env.features import StateFeaturizer
from ..env.half_sword_env import KillSwitch
from ..model.backends import make_backend
from ..model.noise import ActionSampler
from ..model.policy import Policy
from .checkpoint import load_checkpoint, policy_from_checkpoint
from .spool import RolloutBuffer, SpoolWriter


class WeightWatcher:
    def __init__(self, path: Path, poll_s: float = 2.0):
        self.path = Path(path)
        self.poll_s = poll_s
        self._mtime = 0.0
        self._last_check = 0.0
        self.version = 0
        self.hyper: Dict[str, float] = {}

    def check(self) -> Optional[Dict[str, Any]]:
        now = time.perf_counter()
        if now - self._last_check < self.poll_s:
            return None
        self._last_check = now
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return None
        if m == self._mtime:
            return None
        self._mtime = m
        try:
            ck = load_checkpoint(self.path)
        except Exception:
            return None
        self.version = int(ck.get("meta", {}).get("version", self.version + 1))
        self.hyper = ck.get("hyper", {})
        return ck


def build_dims(cfg: Config):
    space = ActionSpace(cfg.actions, cfg.input.keys, cfg.input.mouse_max_speed_px_s)
    feat = StateFeaturizer(cfg.obs.bones, cfg.obs.max_enemies, space.prev_action_dim, cfg.obs.include_prev_action)
    img_channels = cfg.obs.stack * (1 if cfg.obs.gray else 3)
    return space, feat, img_channels


class ActorLoop:
    def __init__(self, cfg: Config, env, policy: Policy, space: ActionSpace, run_dir: Optional[Path],
                 status: Dict[str, Any], log: Callable[[str], None] = print, learn: bool = True,
                 backend_name: Optional[str] = None, device: str = "cuda"):
        self.cfg = cfg
        self.env = env
        self.space = space
        self.run_dir = Path(run_dir) if run_dir else None
        self.status = status
        self.log = log
        self.learn = learn
        cache = str(cfg.resolve_path(cfg.paths.checkpoints_dir) / "trt_cache")
        self.backend = make_backend(policy, backend_name or cfg.model.backend, device, cfg.model.precision,
                                    cfg.model.trt_workspace_mb, cache_dir=cache, log=log)
        self.sampler = ActionSampler(space.n_continuous, space.discrete_sizes, cfg.explore.beta, cfg.explore.scale, seed=cfg.seed)
        self.sampler.deterministic = not learn
        self.spool = SpoolWriter(cfg.resolve_path(cfg.paths.spool_dir)) if learn else None
        self.watcher = WeightWatcher(self.run_dir / "weights" / "latest.pt") if (learn and self.run_dir) else None
        self.gru_hidden = policy.spec.gru_hidden
        self.img_shape = (policy.spec.img_channels, policy.spec.img_size, policy.spec.img_size)
        self.state_dim = policy.spec.state_dim
        self.episodes: List[Dict[str, Any]] = []
        self.outcomes: List[str] = []
        self.buf: Optional[RolloutBuffer] = None
        self.pending: Optional[RolloutBuffer] = None
        self.version = 0
        self._ep_csv = None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            p = self.run_dir / "episodes.csv"
            new = not p.exists()
            self._ep_csv = open(p, "a", newline="", encoding="utf-8")
            self._ep_writer = csv.writer(self._ep_csv)
            if new:
                self._ep_writer.writerow(["time", "episode", "outcome", "eval", "seconds", "steps", "damage_dealt",
                                          "damage_taken", "end_health", "reward_sum", "version", "map"])

    # ------------------------------------------------------------------
    def _new_buffer(self) -> RolloutBuffer:
        b = RolloutBuffer(self.cfg.ppo.rollout_steps, self.img_shape, self.state_dim, self.space.n_discrete, self.gru_hidden)
        b.version = self.version
        return b

    def _maybe_reload(self, force: bool = False) -> None:
        if self.watcher is None:
            return
        if force:
            self.watcher._last_check = 0.0
        ck = self.watcher.check()
        if ck is None:
            return
        try:
            self.backend.load_weights(ck["state_dict"])
            self.version = self.watcher.version
            h = self.watcher.hyper or {}
            if "beta" in h and abs(h["beta"] - self.sampler.noise_c.beta) > 1e-6:
                self.sampler.set_beta(h["beta"])
            if "scale" in h:
                self.sampler.scale = float(h["scale"])
            self.status["weights_version"] = self.version
        except Exception as e:
            self.log(f"[actor] weight reload failed: {e!r}")

    def _log_episode(self, info: Dict[str, Any], eval_ep: bool, ep_index: int) -> None:
        rec = {"episode": ep_index, "outcome": info.get("outcome"), "eval": eval_ep, "seconds": info.get("seconds", 0.0),
               "steps": info.get("steps", 0), "damage_dealt": info.get("damage_dealt", 0.0),
               "damage_taken": info.get("damage_taken", 0.0), "end_health": info.get("end_health", 0.0),
               "reward_sum": info.get("reward_sum", 0.0), "version": self.version, "map": self.status.get("map", "")}
        self.episodes.append(rec)
        self.outcomes.append(rec["outcome"])
        if self.buf is not None:
            self.buf.episodes.append(rec)
        if self._ep_csv:
            self._ep_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S")] + [rec[k] for k in
                                     ("episode", "outcome", "eval", "seconds", "steps", "damage_dealt", "damage_taken",
                                      "end_health", "reward_sum", "version", "map")])
            self._ep_csv.flush()
        wins = sum(1 for o in self.outcomes[-20:] if o == "win")
        self.status.update({"win_rate_20": wins / max(1, len(self.outcomes[-20:])), "last_outcome": rec["outcome"],
                            "last_dealt": rec["damage_dealt"], "last_taken": rec["damage_taken"],
                            "last_reward": rec["reward_sum"], "episode": ep_index})
        self.log(f"[episode {ep_index}] {rec['outcome']:9s} dealt={rec['damage_dealt']:.0f} taken={rec['damage_taken']:.0f} "
                 f"hp_end={rec['end_health']:.0f} reward={rec['reward_sum']:.2f} {'(eval)' if eval_ep else ''}")

    # ------------------------------------------------------------------
    def run(self, hours: Optional[float] = None, max_episodes: Optional[int] = None) -> None:
        cfg = self.cfg
        t_end = time.perf_counter() + hours * 3600.0 if hours else None
        self.status.update({"state": "starting", "backend": self.backend.name})
        self._maybe_reload(force=True)
        if self.learn:
            self.buf = self._new_buffer()
        ep = 0
        fps_t = time.perf_counter()
        fps_n = 0
        try:
            while True:
                if t_end and time.perf_counter() > t_end:
                    self.status["state"] = "time limit reached"
                    break
                if max_episodes and ep >= max_episodes:
                    break
                self.status["state"] = "resetting"
                img, state, rinfo = self.env.reset()
                self.status["map"] = rinfo.get("map", "")
                ep += 1
                eval_ep = self.learn and cfg.explore.eval_every_episodes > 0 and ep % cfg.explore.eval_every_episodes == 0
                self.sampler.deterministic = eval_ep or not self.learn
                self.sampler.reset()
                self.backend.reset_state()
                first = True
                done = False
                self.status["state"] = "fighting" + (" (eval)" if eval_ep else "")
                while not done:
                    self.env.wait_if_paused()
                    h_before = self.backend.hidden() if self.learn else None
                    mean, logstd, logits, value = self.backend.step(img, state)
                    if self.pending is not None:
                        self.pending.bootstrap_value = float(value)
                        self.spool.submit(self.pending)
                        self.pending = None
                    mouse, disc, logp = self.sampler.sample(mean, logstd, logits)
                    img_cpu = img.cpu().numpy() if self.learn else None
                    nimg, nstate, r, done, info = self.env.step(mouse, disc)
                    if self.learn and not eval_ep:
                        self.buf.add(img_cpu, state, mouse, disc, logp, r, done, first, float(value), h_before)
                        if self.buf.full:
                            self.pending = self.buf
                            self.buf = self._new_buffer()
                    first = False
                    img, state = nimg, nstate
                    fps_n += 1
                    now = time.perf_counter()
                    if now - fps_t >= 1.0:
                        self.status.update({"fps": fps_n / (now - fps_t), "step_ms": info.get("step_ms", 0.0),
                                            "infer_ms": self.backend.last_ms, "steps": self.env.total_steps,
                                            "telemetry_age_ms": self.env.telemetry.age_ms(),
                                            "hours_left": (t_end - now) / 3600.0 if t_end else -1})
                        st = self.env.last_state
                        if st is not None:
                            self.status["player_hp"] = st.player.health if st.player else -1
                            self.status["enemy_hp"] = st.enemies[0].health if st.enemies else -1
                            self.status["enemies"] = len(st.enemies)
                        fps_t, fps_n = now, 0
                    if done:
                        self._log_episode(info, eval_ep, ep)
                        if self.learn and not eval_ep and self.buf is not None and self.buf.n > 0 and self.pending is None:
                            # flush partial rollout at episode end so the learner sees terminal transitions quickly
                            if self.buf.n >= max(64, self.cfg.ppo.seq_len * 4):
                                self.buf.bootstrap_value = 0.0
                                self.spool.submit(self.buf)
                                self.buf = self._new_buffer()
                        self._maybe_reload(force=True)
        except KillSwitch:
            self.status["state"] = "stopped by kill switch"
            self.log("[actor] kill switch pressed, stopping")
        except RuntimeError as e:
            self.status["state"] = f"error: {e}"
            self.log(f"[actor] stopped: {e}")
        finally:
            self.env.close()
            if self.spool:
                self.spool.close()
            if self._ep_csv:
                self._ep_csv.close()
