"""Reward shaping for Half Sword fights.

Priorities (in order): win, take no damage, deal damage, do it decisively.
The approach term is potential-based so it cannot change which policy is
optimal, only how fast it is found.  The kite penalty only applies after a
grace window with no damage exchanged while far away, so short retreats to
reset an engagement stay free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from ..config import RewardCfg
from ..telemetry.state import GameState


class Outcome(str, Enum):
    NONE = "none"
    WIN = "win"
    LOSS = "loss"
    TIMEOUT = "timeout"
    SURRENDER = "surrender"
    ABORT = "abort"


@dataclass
class EpisodeStats:
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    ko_dealt: float = 0.0
    ko_taken: float = 0.0
    steps: int = 0
    seconds: float = 0.0
    kite_steps: int = 0
    reward_sum: float = 0.0
    reward_terminal: float = 0.0
    min_health: float = 100.0
    outcome: Outcome = Outcome.NONE
    end_health: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)


def _sum_enemy(st: GameState, attr: str) -> float:
    return sum(max(0.0, getattr(e, attr)) for e in st.enemies)


class RewardShaper:
    def __init__(self, cfg: RewardCfg, tick_dt: float = 1.0 / 60.0):
        self.cfg = cfg
        self.dt = tick_dt
        self.stats = EpisodeStats()
        self._prev: Optional[GameState] = None
        self._prev_enemy_hp = 0.0
        self._prev_enemy_ko = 0.0
        self._prev_dist: Optional[float] = None
        self.t_since_dealt = 99.0
        self.t_since_taken = 99.0
        self._far_time = 0.0

    def reset(self, st: Optional[GameState]) -> None:
        self.stats = EpisodeStats()
        self._prev = st
        self._prev_enemy_hp = _sum_enemy(st, "health") if st else 0.0
        self._prev_enemy_ko = _sum_enemy(st, "consciousness") if st else 0.0
        self._prev_dist = self._dist(st) if st else None
        self.t_since_dealt = 99.0
        self.t_since_taken = 99.0
        self._far_time = 0.0

    @staticmethod
    def _dist(st: Optional[GameState]) -> Optional[float]:
        if st is None or st.player is None or not st.enemies:
            return None
        alive = [e for e in st.enemies if e.alive]
        target = alive[0] if alive else st.enemies[0]
        return st.distance_to(target)

    def _potential(self, dist: Optional[float]) -> float:
        if dist is None:
            return 0.0
        return -max(0.0, dist - self.cfg.engage_distance_m)

    def step(self, st: GameState) -> float:
        c = self.cfg
        comps: Dict[str, float] = {}
        r = 0.0
        prev = self._prev
        # --- damage exchange -------------------------------------------
        enemy_hp = _sum_enemy(st, "health")
        enemy_ko = _sum_enemy(st, "consciousness")
        dealt = max(0.0, self._prev_enemy_hp - enemy_hp) if prev is not None and len(prev.enemies) == len(st.enemies) else 0.0
        ko_dealt = max(0.0, self._prev_enemy_ko - enemy_ko) if prev is not None and len(prev.enemies) == len(st.enemies) else 0.0
        taken = 0.0
        ko_taken = 0.0
        if prev is not None and prev.player is not None and st.player is not None:
            taken = max(0.0, prev.player.health - st.player.health)
            ko_taken = max(0.0, prev.player.consciousness - st.player.consciousness)
        # ignore implausible jumps (respawns / heals)
        if dealt > 60: dealt = 0.0
        if taken > 60: taken = 0.0
        comps["dealt"] = c.damage_dealt * dealt + c.ko_dealt * ko_dealt
        comps["taken"] = -(c.damage_taken * taken + c.ko_taken * ko_taken)
        r += comps["dealt"] + comps["taken"]
        self.stats.damage_dealt += dealt
        self.stats.ko_dealt += ko_dealt
        self.stats.damage_taken += taken
        self.stats.ko_taken += ko_taken
        if dealt > 0 or ko_dealt > 0:
            self.t_since_dealt = 0.0
        else:
            self.t_since_dealt += self.dt
        if taken > 0 or ko_taken > 0:
            self.t_since_taken = 0.0
        else:
            self.t_since_taken += self.dt
        # --- time / stamina --------------------------------------------
        comps["time"] = -c.time_penalty
        r += comps["time"]
        if st.player is not None and st.player.stamina < c.low_stamina_frac * 100.0:
            comps["stamina"] = -c.low_stamina_penalty
            r += comps["stamina"]
        # --- engagement shaping ----------------------------------------
        dist = self._dist(st)
        if dist is not None and self._prev_dist is not None:
            comps["approach"] = c.approach_potential * (self._potential(dist) - self._potential(self._prev_dist))
            r += comps["approach"]
        if dist is not None and dist > c.kite_far_m and min(self.t_since_dealt, self.t_since_taken) > c.kite_grace_s:
            self._far_time += self.dt
            comps["kite"] = -c.kite_penalty
            r += comps["kite"]
            self.stats.kite_steps += 1
        else:
            self._far_time = 0.0
        # --- bookkeeping -----------------------------------------------
        r = max(-c.clip_step, min(c.clip_step, r))
        self._prev = st
        self._prev_enemy_hp = enemy_hp
        self._prev_enemy_ko = enemy_ko
        self._prev_dist = dist
        self.stats.steps += 1
        self.stats.seconds += self.dt
        self.stats.reward_sum += r
        if st.player is not None:
            self.stats.min_health = min(self.stats.min_health, st.player.health)
        for k, v in comps.items():
            self.stats.components[k] = self.stats.components.get(k, 0.0) + v
        return r

    def terminal(self, outcome: Outcome, st: Optional[GameState]) -> float:
        c = self.cfg
        hp = 0.0
        if st is not None and st.player is not None:
            hp = max(0.0, min(100.0, st.player.health)) / 100.0
        if outcome == Outcome.WIN:
            r = c.win + c.win_health_bonus * hp
        elif outcome == Outcome.LOSS:
            r = c.lose
        elif outcome == Outcome.TIMEOUT:
            r = c.timeout
        elif outcome == Outcome.SURRENDER:
            r = c.surrender
        else:
            r = 0.0
        self.stats.outcome = outcome
        self.stats.end_health = hp * 100.0
        self.stats.reward_terminal = r
        self.stats.reward_sum += r
        return r
