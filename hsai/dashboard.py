"""Live terminal dashboard (rich) fed by the actor's status dict and the
learner's status JSON."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


def _fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


class Dashboard:
    def __init__(self, run_dir: Path, actor_status: Dict[str, Any], title: str = "HS-AI training"):
        self.run_dir = Path(run_dir)
        self.actor = actor_status
        self.title = title
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.console = Console()

    def _learner(self) -> Dict[str, Any]:
        p = self.run_dir / "status_learner.json"
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def render(self):
        a = self.actor
        l = self._learner()
        t = Table.grid(expand=True)
        t.add_column(ratio=1)
        t.add_column(ratio=1)
        left = Table(title="Actor (game loop)", expand=True, show_header=False)
        for k in ("state", "backend", "fps", "step_ms", "infer_ms", "episode", "generation", "steps", "win_rate_20",
                  "last_outcome", "last_dealt", "last_taken", "last_reward", "telemetry_age_ms", "map", "enemies",
                  "player_hp", "enemy_hp", "weights_version", "hours_left", "message"):
            if k in a:
                left.add_row(k, _fmt(a[k]))
        right = Table(title="Learner (PPO on GPU)", expand=True, show_header=False)
        for k in ("updates", "total_steps", "backlog", "loss_pi", "loss_v", "entropy_c", "entropy_d", "kl", "clipfrac",
                  "lr", "logstd", "update_s", "reward_scale", "gen", "gen_fitness", "elite_fitness", "elite_gen",
                  "reverts", "beta", "message"):
            if k in l:
                right.add_row(k, _fmt(l[k]))
        t.add_row(left, right)
        return Panel(t, title=self.title, subtitle="F11 pause | F12 stop", border_style="cyan")

    def start(self) -> "Dashboard":
        def loop():
            with Live(self.render(), console=self.console, refresh_per_second=4, screen=False) as live:
                while not self._stop.is_set():
                    try:
                        live.update(self.render())
                    except Exception:
                        pass
                    time.sleep(0.25)
        self._thread = threading.Thread(target=loop, name="hsai-dashboard", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
