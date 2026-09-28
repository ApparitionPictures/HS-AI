"""Builds the low-dimensional state vector from telemetry.

All positions are expressed in the player's *camera* frame (forward, right, up)
in metres, so the same mouse motion means the same thing regardless of where
the fighters stand.  Bone velocities are finite differences across frames
which tells the policy how fast each hand (weapon) is moving.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..telemetry.state import Fighter, GameState

Vec3 = Tuple[float, float, float]


def rotate_to_frame(dx: float, dy: float, yaw_deg: float) -> Tuple[float, float]:
    """World XY delta -> (forward, right) in a frame with the given yaw (Unreal: yaw about +Z, X forward)."""
    r = math.radians(yaw_deg)
    c, s = math.cos(r), math.sin(r)
    fwd = dx * c + dy * s
    right = -dx * s + dy * c
    return fwd, right


def _clip(x: float, lim: float) -> float:
    return max(-lim, min(lim, x))


class StateFeaturizer:
    POS_SCALE = 5.0
    VEL_SCALE = 5.0
    BONE_SCALE = 2.0
    BONE_VEL_SCALE = 8.0

    def __init__(self, bones: List[str], max_enemies: int, prev_action_dim: int, include_prev_action: bool = True):
        self.bones = list(bones)
        self.max_enemies = max_enemies
        self.prev_action_dim = prev_action_dim if include_prev_action else 0
        self.include_prev_action = include_prev_action
        self._prev_bones: Dict[str, Dict[str, Vec3]] = {}
        self._prev_t: Optional[float] = None
        # layout sizes
        self.player_dim = 11 + len(self.bones) * 7
        self.enemy_dim = 15 + len(self.bones) * 7
        self.time_dim = 3
        self.dim = self.player_dim + self.max_enemies * self.enemy_dim + self.time_dim + self.prev_action_dim

    def reset(self) -> None:
        self._prev_bones = {}
        self._prev_t = None

    # ------------------------------------------------------------------
    def _bone_feats(self, key: str, f: Fighter, origin: Vec3, yaw: float, dt: float, out: List[float]) -> None:
        prev = self._prev_bones.get(key, {})
        cur: Dict[str, Vec3] = {}
        for b in self.bones:
            p = f.bones.get(b)
            if p is None:
                out += [0.0] * 7
                continue
            cur[b] = p
            fwd, right = rotate_to_frame(p[0] - origin[0], p[1] - origin[1], yaw)
            up = p[2] - origin[2]
            vx = vy = vz = 0.0
            if b in prev and dt > 1e-4:
                dfwd, dright = rotate_to_frame(p[0] - prev[b][0], p[1] - prev[b][1], yaw)
                vx, vy, vz = dfwd / dt, dright / dt, (p[2] - prev[b][2]) / dt
            out += [1.0, _clip(fwd, 4) / self.BONE_SCALE, _clip(right, 4) / self.BONE_SCALE, _clip(up, 4) / self.BONE_SCALE,
                    _clip(vx, 16) / self.BONE_VEL_SCALE, _clip(vy, 16) / self.BONE_VEL_SCALE, _clip(vz, 16) / self.BONE_VEL_SCALE]
        self._prev_bones[key] = cur

    def __call__(self, st: GameState, prev_action: Optional[np.ndarray], t_frac: float,
                 t_since_dealt: float, t_since_taken: float) -> np.ndarray:
        out: List[float] = []
        dt = 1.0 / 60.0
        if self._prev_t is not None and st.game_time > self._prev_t:
            dt = min(0.25, st.game_time - self._prev_t)
        self._prev_t = st.game_time
        p = st.player
        yaw = st.cam_yaw
        if p is None:
            out += [0.0] * self.player_dim
            origin: Vec3 = (0.0, 0.0, 0.0)
        else:
            origin = p.pos
            vf, vr = rotate_to_frame(p.vel[0], p.vel[1], yaw)
            dyaw = math.radians(p.yaw - yaw)
            out += [p.health / 100.0, p.consciousness / 100.0, p.stamina / 100.0, p.tonus / 100.0,
                    1.0 if p.dead else 0.0, min(1.0, math.hypot(p.vel[0], p.vel[1]) / self.VEL_SCALE),
                    _clip(vf, 5) / self.VEL_SCALE, _clip(vr, 5) / self.VEL_SCALE,
                    _clip(st.cam_pitch, 90) / 90.0, math.sin(dyaw), math.cos(dyaw)]
            self._bone_feats("player", p, origin, yaw, dt, out)
        for i in range(self.max_enemies):
            if i >= len(st.enemies):
                out += [0.0] * self.enemy_dim
                continue
            e = st.enemies[i]
            fwd, right = rotate_to_frame(e.pos[0] - origin[0], e.pos[1] - origin[1], yaw)
            up = e.pos[2] - origin[2]
            dist = math.sqrt(fwd * fwd + right * right + up * up)
            vf, vr = rotate_to_frame(e.vel[0], e.vel[1], yaw)
            dyaw = math.radians(e.yaw - yaw)
            out += [1.0, e.health / 100.0, e.consciousness / 100.0, e.stamina / 100.0, 1.0 if e.dead else 0.0,
                    _clip(fwd, 10) / self.POS_SCALE, _clip(right, 10) / self.POS_SCALE, _clip(up, 5) / self.POS_SCALE,
                    min(1.0, dist / 10.0), 1.0 / (1.0 + dist),
                    math.sin(dyaw), math.cos(dyaw),
                    _clip(vf, 5) / self.VEL_SCALE, _clip(vr, 5) / self.VEL_SCALE, _clip(e.vel[2], 5) / self.VEL_SCALE]
            self._bone_feats(f"enemy{i}", e, origin, yaw, dt, out)
        out += [min(1.0, t_frac), min(1.0, t_since_dealt / 5.0), min(1.0, t_since_taken / 5.0)]
        if self.include_prev_action:
            if prev_action is None:
                out += [0.0] * self.prev_action_dim
            else:
                out += [float(x) for x in prev_action]
        v = np.asarray(out, dtype=np.float32)
        assert v.shape[0] == self.dim, (v.shape, self.dim)
        return v
