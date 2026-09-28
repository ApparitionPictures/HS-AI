"""Offline simulator with the same interface as HalfSwordEnv.

A toy 2-D duel: the enemy walks toward the player and hits when close; the
agent's mouse velocity 'swings' and damages the enemy when the enemy is in
front and the swing is fast.  It exists to validate the whole actor/learner
pipeline (shapes, spool, PPO, generations, backends) without the game, on any
machine, and is what the unit tests use.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..config import Config
from ..env.actions import MOVE_TABLE, ActionSpace
from ..env.episode import EpisodeTracker
from ..env.features import StateFeaturizer
from ..env.reward import Outcome, RewardShaper
from ..telemetry.state import Fighter, GameState


class _FakeTelemetry:
    def __init__(self):
        self._st: Optional[GameState] = None

    def latest(self):
        return self._st

    def age_ms(self):
        return 0.0

    def is_fresh(self):
        return True


class _FakeController:
    def __init__(self):
        import threading
        self.killed = threading.Event()
        self.paused = False


class SimEnv:
    def __init__(self, cfg: Config, space: ActionSpace, featurizer: StateFeaturizer, log=print, realtime: bool = False,
                 device: str = "cpu", seed: int = 0):
        self.cfg = cfg
        self.space = space
        self.featurizer = featurizer
        self.log = log
        self.realtime = realtime
        self.dt = 1.0 / cfg.game.target_fps
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        C = cfg.obs.stack * (1 if cfg.obs.gray else 3)
        self.obs = torch.zeros((C, cfg.obs.size, cfg.obs.size), dtype=torch.uint8, device=self.device)
        self.reward = RewardShaper(cfg.reward, self.dt)
        self.tracker = EpisodeTracker(cfg.episode, self.dt)
        self.telemetry = _FakeTelemetry()
        self.controller = _FakeController()
        self.episode_index = 0
        self.total_steps = 0
        self.last_state: Optional[GameState] = None
        self.t = 0.0
        self.seq = 0
        self.prev_action_vec = None

    # ---- toy dynamics ----------------------------------------------------
    def _state(self) -> GameState:
        self.seq += 1
        p = Fighter(health=self.php, consciousness=100.0, stamina=self.pstam, tonus=100.0, team=1,
                    pos=(self.px, self.py, 0.0), yaw=0.0, vel=(self.pvx, self.pvy, 0.0),
                    bones={"head": (self.px, self.py, 1.7), "hand_r": (self.px + 0.5, self.py + self.hand, 1.2)})
        e = Fighter(health=self.ehp, consciousness=100.0, stamina=100.0, tonus=100.0, team=2, dead=self.ehp <= 0,
                    pos=(self.ex, self.ey, 0.0), yaw=180.0, vel=(self.evx, self.evy, 0.0),
                    bones={"head": (self.ex, self.ey, 1.7)})
        st = GameState(seq=self.seq, game_time=self.t, map_name="Sim", cam_yaw=0.0, cam_pitch=0.0, n_fighters=2,
                       player=p, enemies=[e], recv_time=time.perf_counter())
        return st

    def _render(self) -> torch.Tensor:
        S = self.cfg.obs.size
        C = 1 if self.cfg.obs.gray else 3
        frame = torch.zeros((C, S, S), dtype=torch.uint8, device=self.device)
        # enemy blob whose x-position encodes the lateral offset and size encodes distance
        dx, dy = self.ex - self.px, self.ey - self.py
        dist = max(0.3, math.hypot(dx, dy))
        cx = int(S / 2 + (dy / 6.0) * S / 2)
        r = int(max(2, min(S // 3, S / (2.5 * dist))))
        y0, y1 = max(0, S // 2 - r), min(S, S // 2 + r)
        x0, x1 = max(0, cx - r), min(S, cx + r)
        if x1 > x0 and y1 > y0:
            frame[:, y0:y1, x0:x1] = 200
        # own hand position as a bar at the bottom
        hx = int(S / 2 + self.hand * S / 3)
        frame[:, S - 6:S, max(0, hx - 3):min(S, hx + 3)] = 255
        self.obs[:-C] = self.obs[C:].clone()
        self.obs[-C:] = frame
        return self.obs

    def wait_if_paused(self) -> None:
        pass

    def reset(self) -> Tuple[torch.Tensor, np.ndarray, Dict]:
        self.php, self.ehp, self.pstam = 100.0, 100.0, 100.0
        self.px, self.py, self.pvx, self.pvy = 0.0, 0.0, 0.0, 0.0
        ang = self.rng.uniform(-0.6, 0.6)
        d = self.rng.uniform(3.0, 6.0)
        self.ex, self.ey, self.evx, self.evy = d * math.cos(ang), d * math.sin(ang), 0.0, 0.0
        self.hand = 0.0
        self.t = 0.0
        self.obs.zero_()
        st = self._state()
        self.telemetry._st = st
        self.featurizer.reset()
        self.reward.reset(st)
        self.tracker.reset(st)
        self.prev_action_vec = None
        self.last_state = st
        self.episode_index += 1
        for _ in range(self.cfg.obs.stack):
            self._render()
        return self.obs, self.featurizer(st, None, 0.0, 99.0, 99.0), {"episode": self.episode_index, "map": "Sim"}

    def step(self, mouse: np.ndarray, discrete: np.ndarray):
        t0 = time.perf_counter()
        # player movement
        mi = self.space.head_index.get("move")
        f, r = MOVE_TABLE[int(discrete[mi])] if mi is not None else (0, 0)
        speed = 2.0 * (1.5 if ("sprint" in self.space.head_index and discrete[self.space.head_index["sprint"]] == 1) else 1.0)
        self.pvx, self.pvy = f * speed, r * speed
        self.px += self.pvx * self.dt
        self.py += self.pvy * self.dt
        # hand follows mouse x with inertia; swing speed = |mouse|
        self.hand += 0.6 * (float(np.clip(mouse[0], -1, 1)) * 0.8 - self.hand)
        swing = abs(float(mouse[0])) * (1.0 if ("lmb" in self.space.head_index and discrete[self.space.head_index["lmb"]] == 1) else 0.3)
        # enemy approaches and attacks
        dx, dy = self.px - self.ex, self.py - self.ey
        dist = math.hypot(dx, dy)
        if self.ehp > 0:
            if dist > 1.2:
                self.evx, self.evy = dx / dist * 1.4, dy / dist * 1.4
                self.ex += self.evx * self.dt
                self.ey += self.evy * self.dt
            else:
                self.evx = self.evy = 0.0
                if self.rng.random() < 0.04:
                    self.php -= 8.0 * (0.4 if abs(self.hand) > 0.5 else 1.0)  # a raised hand 'blocks' a bit
        # player hits when close, roughly facing, swinging fast
        if dist < 1.8 and abs(dy) < 1.0 and dx < 0 and swing > 0.5 and self.rng.random() < 0.15 * swing:
            self.ehp -= 12.0
        self.pstam = max(0.0, min(100.0, self.pstam + (1.0 - 25.0 * swing) * self.dt * 5))
        self.t += self.dt
        st = self._state()
        self.telemetry._st = st
        img = self._render()
        rew = self.reward.step(st)
        outcome = self.tracker.update(st, surrender_pressed=self.space.is_surrender(discrete))
        done = outcome != Outcome.NONE
        info: Dict = {"outcome": outcome.value, "stale": False}
        if done:
            rew += self.reward.terminal(outcome, st)
            s = self.reward.stats
            info.update({"damage_dealt": s.damage_dealt, "damage_taken": s.damage_taken, "steps": s.steps, "seconds": s.seconds,
                         "end_health": s.end_health, "reward_sum": s.reward_sum, "reward_terminal": s.reward_terminal,
                         "min_health": s.min_health, "kite_steps": s.kite_steps, "components": dict(s.components)})
        self.prev_action_vec = self.space.prev_action_vector(mouse, discrete)
        state_vec = self.featurizer(st, self.prev_action_vec, self.t / self.cfg.episode.max_seconds,
                                    self.reward.t_since_dealt, self.reward.t_since_taken)
        self.last_state = st
        self.total_steps += 1
        if self.realtime:
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t0)))
        info["step_ms"] = (time.perf_counter() - t0) * 1000.0
        return img, state_vec, float(rew), done, info

    def close(self) -> None:
        pass


def make_sim_env(cfg: Config, space: ActionSpace, feat: StateFeaturizer, log=print, realtime: bool = False, device: str = "cpu"):
    return SimEnv(cfg, space, feat, log=log, realtime=realtime, device=device, seed=cfg.seed)
