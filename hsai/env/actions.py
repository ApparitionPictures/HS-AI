"""Hybrid action space: continuous mouse velocity + categorical key heads.

The same object converts model samples into an ``ActionCommand`` for the input
controller and converts recorded human input back into action indices for
behaviour cloning, so both directions always agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from ..config import ActionsCfg, KeysCfg
from ..input.controller import ActionCommand

# index -> (forward, right) components
MOVE_TABLE: List[Tuple[int, int]] = [(0, 0), (1, 0), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 1), (-1, -1), (-1, 1)]


@dataclass
class Head:
    name: str
    n: int


class ActionSpace:
    def __init__(self, actions: ActionsCfg, keys: KeysCfg, mouse_max_speed_px_s: float):
        self.keys = keys
        self.mouse_max = float(mouse_max_speed_px_s)
        self.heads: List[Head] = []
        if actions.move:
            self.heads.append(Head("move", 9))
        for name, on in (("lmb", actions.lmb), ("rmb", actions.rmb), ("thrust", actions.thrust),
                         ("sprint", actions.sprint), ("kick", actions.kick), ("crouch", actions.crouch),
                         ("surrender", actions.surrender)):
            if on:
                self.heads.append(Head(name, 2))
        self.head_index: Dict[str, int] = {h.name: i for i, h in enumerate(self.heads)}
        self.n_discrete = len(self.heads)
        self.discrete_sizes = [h.n for h in self.heads]
        self.n_continuous = 2
        # flattened one-hot size used for the previous-action feature
        self.prev_action_dim = self.n_continuous + sum(self.discrete_sizes)

    # ---- model -> game ----------------------------------------------------
    def to_command(self, mouse: Sequence[float], discrete: Sequence[int]) -> ActionCommand:
        cmd = ActionCommand()
        mx = float(np.clip(mouse[0], -1.0, 1.0))
        my = float(np.clip(mouse[1], -1.0, 1.0))
        cmd.mouse_vx = mx * self.mouse_max
        cmd.mouse_vy = my * self.mouse_max
        k = self.keys
        for head, a in zip(self.heads, discrete):
            a = int(a)
            if head.name == "move":
                f, r = MOVE_TABLE[a]
                if f > 0: cmd.keys.add(k.forward)
                if f < 0: cmd.keys.add(k.back)
                if r > 0: cmd.keys.add(k.right)
                if r < 0: cmd.keys.add(k.left)
            elif a == 1:
                if head.name == "lmb": cmd.buttons.add("L")
                elif head.name == "rmb": cmd.buttons.add("R")
                elif head.name == "thrust": cmd.keys.add(k.thrust)
                elif head.name == "sprint": cmd.keys.add(k.sprint)
                elif head.name == "kick": cmd.keys.add(k.kick)
                elif head.name == "crouch": cmd.keys.add(k.crouch)
                elif head.name == "surrender": cmd.keys.add(k.surrender)
        return cmd

    def is_surrender(self, discrete: Sequence[int]) -> bool:
        i = self.head_index.get("surrender")
        return i is not None and int(discrete[i]) == 1

    # ---- game -> model (for imitation) ----------------------------------
    def from_human(self, keys_down: Set[str], buttons: Set[str], dx_px_s: float, dy_px_s: float) -> Tuple[np.ndarray, np.ndarray]:
        mouse = np.array([np.clip(dx_px_s / self.mouse_max, -1, 1), np.clip(dy_px_s / self.mouse_max, -1, 1)], dtype=np.float32)
        k = self.keys
        disc = np.zeros(self.n_discrete, dtype=np.int64)
        for i, head in enumerate(self.heads):
            if head.name == "move":
                f = (1 if k.forward in keys_down else 0) - (1 if k.back in keys_down else 0)
                r = (1 if k.right in keys_down else 0) - (1 if k.left in keys_down else 0)
                disc[i] = MOVE_TABLE.index((f, r))
            elif head.name == "lmb": disc[i] = 1 if "L" in buttons else 0
            elif head.name == "rmb": disc[i] = 1 if "R" in buttons else 0
            elif head.name == "thrust": disc[i] = 1 if k.thrust in keys_down else 0
            elif head.name == "sprint": disc[i] = 1 if k.sprint in keys_down else 0
            elif head.name == "kick": disc[i] = 1 if k.kick in keys_down else 0
            elif head.name == "crouch": disc[i] = 1 if k.crouch in keys_down else 0
            elif head.name == "surrender": disc[i] = 1 if k.surrender in keys_down else 0
        return mouse, disc

    def prev_action_vector(self, mouse: Sequence[float], discrete: Sequence[int]) -> np.ndarray:
        v = np.zeros(self.prev_action_dim, dtype=np.float32)
        v[0] = float(mouse[0]); v[1] = float(mouse[1])
        off = 2
        for head, a in zip(self.heads, discrete):
            v[off + int(a)] = 1.0
            off += head.n
        return v

    def human_keys_of_interest(self) -> List[str]:
        k = self.keys
        return [k.forward, k.back, k.left, k.right, k.thrust, k.sprint, k.kick, k.crouch, k.surrender, k.lock_on]
