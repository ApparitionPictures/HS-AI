"""Game-state records exported by the HSAI Lua mod and their text encoding.

Line layout (protocol 1), fields separated by '|':

    HSAI|1|seq|game_time|map|cam_yaw|cam_pitch|n_fighters|P1|<fighter>|E<n>|<fighter>...|END
    HSAI|1|seq|game_time|map|cam_yaw|cam_pitch|n_fighters|P0|E<n>|...|END      (no player pawn)

<fighter> = health|consciousness|stamina|tonus|dead|team|x|y|z|yaw|vx|vy|vz|nb|(have|bx|by|bz)*nb

Positions are Unreal units (cm).  Python converts to metres.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PROTO = 1
CM = 0.01


@dataclass
class Fighter:
    health: float = 0.0
    consciousness: float = 0.0
    stamina: float = 0.0
    tonus: float = 0.0
    dead: bool = False
    team: int = -1
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # metres
    yaw: float = 0.0                                     # degrees
    vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # m/s
    bones: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)  # metres, only bones that exist

    @property
    def alive(self) -> bool:
        return (not self.dead) and self.health > 0.0

    @property
    def conscious(self) -> bool:
        return self.consciousness > 0.0


@dataclass
class GameState:
    seq: int = 0
    game_time: float = 0.0
    map_name: str = ""
    cam_yaw: float = 0.0
    cam_pitch: float = 0.0
    n_fighters: int = 0
    player: Optional[Fighter] = None
    enemies: List[Fighter] = field(default_factory=list)
    recv_time: float = 0.0     # perf_counter when received

    @property
    def nearest_enemy(self) -> Optional[Fighter]:
        return self.enemies[0] if self.enemies else None

    def distance_to(self, f: Fighter) -> float:
        if self.player is None:
            return 1e9
        px, py, pz = self.player.pos
        x, y, z = f.pos
        return math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)

    @property
    def any_enemy_alive(self) -> bool:
        return any(e.alive for e in self.enemies)


def _parse_fighter(parts: List[str], i: int, bone_names: List[str]) -> Tuple[Fighter, int]:
    f = Fighter()
    f.health = float(parts[i]); f.consciousness = float(parts[i + 1]); f.stamina = float(parts[i + 2])
    f.tonus = float(parts[i + 3]); f.dead = parts[i + 4] == "1"; f.team = int(float(parts[i + 5]))
    f.pos = (float(parts[i + 6]) * CM, float(parts[i + 7]) * CM, float(parts[i + 8]) * CM)
    f.yaw = float(parts[i + 9])
    f.vel = (float(parts[i + 10]) * CM, float(parts[i + 11]) * CM, float(parts[i + 12]) * CM)
    nb = int(parts[i + 13])
    i += 14
    for b in range(nb):
        have = parts[i] == "1"
        if have:
            name = bone_names[b] if b < len(bone_names) else f"bone{b}"
            f.bones[name] = (float(parts[i + 1]) * CM, float(parts[i + 2]) * CM, float(parts[i + 3]) * CM)
        i += 4
    return f, i


def parse_line(line: str, bone_names: Optional[List[str]] = None) -> Optional[GameState]:
    """Parse one telemetry line; returns None if the line is incomplete or malformed."""
    bone_names = bone_names or ["head", "hand_r", "hand_l", "pelvis"]
    line = line.strip()
    if not line.startswith("HSAI|") or not line.endswith("|END"):
        return None
    parts = line.split("|")
    try:
        if int(parts[1]) != PROTO:
            return None
        st = GameState()
        st.seq = int(parts[2])
        st.game_time = float(parts[3])
        st.map_name = parts[4]
        st.cam_yaw = float(parts[5])
        st.cam_pitch = float(parts[6])
        st.n_fighters = int(parts[7])
        i = 8
        if parts[i] == "P1":
            st.player, i = _parse_fighter(parts, i + 1, bone_names)
        elif parts[i] == "P0":
            i += 1
        else:
            return None
        if not parts[i].startswith("E"):
            return None
        n = int(parts[i][1:])
        i += 1
        for _ in range(n):
            e, i = _parse_fighter(parts, i, bone_names)
            st.enemies.append(e)
        if parts[i] != "END":
            return None
        st.recv_time = time.perf_counter()
        return st
    except (ValueError, IndexError):
        return None


def _encode_fighter(f: Fighter, bone_names: List[str]) -> List[str]:
    out = [f"{f.health:.1f}", f"{f.consciousness:.1f}", f"{f.stamina:.1f}", f"{f.tonus:.1f}",
           "1" if f.dead else "0", str(f.team)]
    out += [f"{v / CM:.1f}" for v in f.pos]
    out.append(f"{f.yaw:.1f}")
    out += [f"{v / CM:.1f}" for v in f.vel]
    out.append(str(len(bone_names)))
    for b in bone_names:
        if b in f.bones:
            out.append("1")
            out += [f"{v / CM:.1f}" for v in f.bones[b]]
        else:
            out.append("0")
            out += [f"{v / CM:.1f}" for v in f.pos]
    return out


def encode_line(st: GameState, bone_names: Optional[List[str]] = None) -> str:
    """Inverse of parse_line (used by tests and the offline simulator)."""
    bone_names = bone_names or ["head", "hand_r", "hand_l", "pelvis"]
    out = ["HSAI", str(PROTO), str(st.seq), f"{st.game_time:.2f}", st.map_name, f"{st.cam_yaw:.1f}",
           f"{st.cam_pitch:.1f}", str(st.n_fighters)]
    if st.player is not None:
        out.append("P1")
        out += _encode_fighter(st.player, bone_names)
    else:
        out.append("P0")
    out.append(f"E{len(st.enemies)}")
    for e in st.enemies:
        out += _encode_fighter(e, bone_names)
    out.append("END")
    return "|".join(out)
