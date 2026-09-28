"""Generation / elite loop ("keep the best, continue from it").

Every ``episodes_per_generation`` episodes the current policy gets a fitness
score.  The best-scoring generation is the *elite*.  If the score drops below
the elite by more than ``margin`` for ``patience`` generations in a row, the
policy weights are reverted to the elite and the learning hyper-parameters
are perturbed (population-based-training style explore step), which is what
makes the search restart from the strongest point instead of drifting.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import GenerationsCfg


@dataclass
class GenerationRecord:
    index: int
    episodes: int
    fitness: float
    win_rate: float
    clean: float
    dealt: float
    taken: float
    mean_reward: float
    checkpoint: str = ""
    hyper: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def fitness_of(episodes: List[Dict[str, Any]], cfg: GenerationsCfg) -> Dict[str, float]:
    n = max(1, len(episodes))
    wins = [e for e in episodes if e.get("outcome") == "win"]
    win_rate = len(wins) / n
    clean = (sum(max(0.0, e.get("end_health", 0.0)) / 100.0 for e in wins) / len(wins)) if wins else 0.0
    dealt = sum(min(100.0, e.get("damage_dealt", 0.0)) / 100.0 for e in episodes) / n
    taken = sum(min(100.0, e.get("damage_taken", 0.0)) / 100.0 for e in episodes) / n
    mean_reward = sum(e.get("reward_sum", 0.0) for e in episodes) / n
    fitness = cfg.fitness_win * win_rate + cfg.fitness_clean * win_rate * clean + cfg.fitness_dealt * dealt - cfg.fitness_taken * taken
    return {"fitness": fitness, "win_rate": win_rate, "clean": clean, "dealt": dealt, "taken": taken, "mean_reward": mean_reward}


class GenerationManager:
    def __init__(self, cfg: GenerationsCfg, run_dir: Path, seed: int = 0):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.rng = random.Random(seed)
        self.pending: List[Dict[str, Any]] = []
        self.history: List[GenerationRecord] = []
        self.elite: Optional[GenerationRecord] = None
        self.worse_streak = 0
        self.total_episodes = 0
        self.reverts = 0

    # ------------------------------------------------------------------
    def add_episodes(self, eps: List[Dict[str, Any]]) -> None:
        self.pending.extend(eps)
        self.total_episodes += len(eps)

    def ready(self) -> bool:
        return len(self.pending) >= self.cfg.episodes_per_generation

    def close_generation(self, hyper: Dict[str, float]) -> Dict[str, Any]:
        """Score the pending episodes. Returns a decision dict:
        {"record": GenerationRecord, "is_elite": bool, "revert": bool, "new_hyper": dict|None}"""
        f = fitness_of(self.pending, self.cfg)
        idx = len(self.history) + 1
        rec = GenerationRecord(index=idx, episodes=len(self.pending), fitness=f["fitness"], win_rate=f["win_rate"],
                               clean=f["clean"], dealt=f["dealt"], taken=f["taken"], mean_reward=f["mean_reward"],
                               checkpoint=f"gen_{idx:04d}.pt", hyper=dict(hyper))
        self.pending = []
        self.history.append(rec)
        is_elite = False
        revert = False
        new_hyper: Optional[Dict[str, float]] = None
        if self.elite is None or rec.fitness > self.elite.fitness:
            if rec.episodes >= self.cfg.min_episodes_for_elite or self.elite is None:
                self.elite = rec
                is_elite = True
                self.worse_streak = 0
        if not is_elite and self.elite is not None and rec.fitness < self.elite.fitness - self.cfg.margin:
            self.worse_streak += 1
            if self.worse_streak >= self.cfg.patience:
                revert = True
                self.worse_streak = 0
                self.reverts += 1
                new_hyper = self.perturb(hyper)
        elif not is_elite:
            self.worse_streak = 0
        self.save()
        return {"record": rec, "is_elite": is_elite, "revert": revert, "new_hyper": new_hyper}

    def perturb(self, hyper: Dict[str, float]) -> Dict[str, float]:
        c = self.cfg
        h = dict(hyper)
        h["lr"] = float(h["lr"] * self.rng.choice(c.lr_factors))
        h["lr"] = max(1e-6, min(3e-3, h["lr"]))
        ef = self.rng.choice(c.entropy_factors)
        h["entropy_discrete"] = float(max(1e-5, min(0.05, h["entropy_discrete"] * ef)))
        h["entropy_continuous"] = float(max(1e-5, min(0.05, h["entropy_continuous"] * ef)))
        h["beta"] = float(max(0.0, min(2.0, h["beta"] + self.rng.choice([-c.beta_delta, c.beta_delta]))))
        return h

    def top_k(self) -> List[GenerationRecord]:
        return sorted(self.history, key=lambda r: r.fitness, reverse=True)[: self.cfg.keep_top_k]

    # ------------------------------------------------------------------
    def save(self) -> None:
        d = {"history": [r.to_dict() for r in self.history], "elite": self.elite.to_dict() if self.elite else None,
             "worse_streak": self.worse_streak, "total_episodes": self.total_episodes, "reverts": self.reverts}
        p = self.run_dir / "generations.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)

    def load(self) -> None:
        p = self.run_dir / "generations.json"
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.history = [GenerationRecord(**r) for r in d.get("history", [])]
        self.elite = GenerationRecord(**d["elite"]) if d.get("elite") else None
        self.worse_streak = d.get("worse_streak", 0)
        self.total_episodes = d.get("total_episodes", 0)
        self.reverts = d.get("reverts", 0)
