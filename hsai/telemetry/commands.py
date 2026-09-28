"""Sends commands to the Lua mod through a small command file (atomic replace)."""
from __future__ import annotations

import os
import tempfile
from typing import Iterable, List, Optional


class CommandWriter:
    def __init__(self, cmd_path: str):
        self.path = os.path.expandvars(cmd_path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def send(self, *commands: str) -> None:
        self.send_many(commands)

    def send_many(self, commands: Iterable[str]) -> None:
        lines: List[str] = [c.strip() for c in commands if c and c.strip()]
        if not lines:
            return
        existing: List[str] = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                existing = [l.rstrip("\n") for l in f if l.strip()]
        except FileNotFoundError:
            pass
        fd, tmp = tempfile.mkstemp(prefix="cmd", suffix=".tmp", dir=os.path.dirname(self.path))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(existing + lines) + "\n")
        os.replace(tmp, self.path)

    # convenience wrappers ----------------------------------------------
    def heal(self) -> None: self.send("heal")
    def heal_enemies(self) -> None: self.send("heal_enemies")
    def kill_enemies(self) -> None: self.send("kill_enemies")
    def invulnerable(self, on: bool) -> None: self.send(f"invuln|{1 if on else 0}")
    def time_dilation(self, x: float) -> None: self.send(f"timedilation|{x:.3f}")
    def spawn(self, class_path: str, distance_m: float = 3.0, team: Optional[int] = None) -> None:
        self.send(f"spawn|{class_path}|{distance_m * 100:.0f}" + (f"|{team}" if team is not None else ""))
    def despawn_spawned(self) -> None: self.send("despawn_spawned")
    def set_player(self, prop: str, value) -> None: self.send(f"set|{prop}|{value}")
    def rescan(self) -> None: self.send("rescan")
