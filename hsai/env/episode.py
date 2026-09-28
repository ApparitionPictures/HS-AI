"""Terminal detection from telemetry."""
from __future__ import annotations

from typing import Optional

from ..config import EpisodeCfg
from ..telemetry.state import GameState
from .reward import Outcome


def is_fight_live(st: Optional[GameState], max_dist_m: float = 40.0) -> bool:
    """A fight is live when the player pawn exists and is alive and at least one
    hostile fighter is alive within ``max_dist_m``."""
    if st is None or st.player is None or not st.player.alive:
        return False
    for e in st.enemies:
        if e.alive and st.distance_to(e) <= max_dist_m:
            return True
    return False


class EpisodeTracker:
    def __init__(self, cfg: EpisodeCfg, dt: float = 1.0 / 60.0):
        self.cfg = cfg
        self.dt = dt
        self.reset(None)

    def reset(self, st: Optional[GameState]) -> None:
        self.t = 0.0
        self.player_ko_t = 0.0
        self.enemy_down_t = 0.0
        self.hopeless_t = 0.0
        self.map_name = st.map_name if st else ""
        self.start_enemies = len(st.enemies) if st else 0
        self.lost_track_t = 0.0

    def update(self, st: Optional[GameState], surrender_pressed: bool = False) -> Outcome:
        c = self.cfg
        self.t += self.dt
        if st is None:
            self.lost_track_t += self.dt
            return Outcome.ABORT if self.lost_track_t > 5.0 else Outcome.NONE
        self.lost_track_t = 0.0
        if self.t < c.start_grace_s:
            return Outcome.NONE
        if self.map_name and st.map_name and st.map_name != self.map_name:
            return Outcome.ABORT
        p = st.player
        if p is None:
            self.lost_track_t += self.dt
            return Outcome.NONE
        if surrender_pressed:
            return Outcome.SURRENDER
        # --- loss ---------------------------------------------------------
        if p.dead or p.health <= 0.0:
            return Outcome.LOSS
        if p.consciousness <= 0.0:
            self.player_ko_t += self.dt
            if self.player_ko_t >= c.ko_grace_s:
                return Outcome.LOSS
        else:
            self.player_ko_t = 0.0
        # --- win ----------------------------------------------------------
        enemies = st.enemies
        if enemies:
            all_down = all((e.dead or e.health <= 0.0 or e.consciousness <= 0.0) for e in enemies)
            all_dead = all((e.dead or e.health <= 0.0) for e in enemies)
            if all_down:
                self.enemy_down_t += self.dt
                need = c.enemy_dead_confirm_s if all_dead else c.ko_grace_s
                if self.enemy_down_t >= need:
                    return Outcome.WIN
            else:
                self.enemy_down_t = 0.0
        elif self.start_enemies > 0:
            # enemies vanished (despawned after death) -> count as a win once confirmed
            self.enemy_down_t += self.dt
            if self.enemy_down_t >= c.enemy_dead_confirm_s:
                return Outcome.WIN
        # --- stalemate / hopeless ----------------------------------------
        if p.health <= c.hopeless_health:
            self.hopeless_t += self.dt
            if self.hopeless_t >= c.auto_surrender_after_s:
                return Outcome.SURRENDER
        else:
            self.hopeless_t = 0.0
        if self.t >= c.max_seconds:
            return Outcome.TIMEOUT
        return Outcome.NONE
