"""Rollout spool: the actor writes fixed-length rollout files, the learner
consumes them.  Files are plain .npz (uncompressed) written atomically in a
background thread so the 60 Hz loop never blocks on disk."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class RolloutBuffer:
    """Pre-allocated arrays for one rollout file."""

    def __init__(self, T: int, img_shape, state_dim: int, n_disc: int, gru_hidden: int):
        self.T = T
        self.img = np.zeros((T, *img_shape), dtype=np.uint8)
        self.state = np.zeros((T, state_dim), dtype=np.float32)
        self.mouse = np.zeros((T, 2), dtype=np.float32)
        self.disc = np.zeros((T, n_disc), dtype=np.int64)
        self.logp = np.zeros((T,), dtype=np.float32)
        self.reward = np.zeros((T,), dtype=np.float32)
        self.done = np.zeros((T,), dtype=bool)
        self.first = np.zeros((T,), dtype=bool)
        self.value = np.zeros((T,), dtype=np.float32)
        self.h = np.zeros((T, gru_hidden), dtype=np.float16)   # hidden state BEFORE step t
        self.n = 0
        self.episodes: List[Dict[str, Any]] = []
        self.bootstrap_value = 0.0
        self.version = 0

    def add(self, img, state, mouse, disc, logp, reward, done, first, value, h) -> None:
        i = self.n
        self.img[i] = img
        self.state[i] = state
        self.mouse[i] = mouse
        self.disc[i] = disc
        self.logp[i] = logp
        self.reward[i] = reward
        self.done[i] = done
        self.first[i] = first
        self.value[i] = value
        self.h[i] = h
        self.n += 1

    @property
    def full(self) -> bool:
        return self.n >= self.T

    def to_dict(self) -> Dict[str, Any]:
        n = self.n
        return {"img": self.img[:n], "state": self.state[:n], "mouse": self.mouse[:n], "disc": self.disc[:n],
                "logp": self.logp[:n], "reward": self.reward[:n], "done": self.done[:n], "first": self.first[:n],
                "value": self.value[:n], "h": self.h[:n],
                "bootstrap_value": np.float32(self.bootstrap_value), "version": np.int64(self.version),
                "episodes": np.frombuffer(json.dumps(self.episodes).encode("utf-8"), dtype=np.uint8),
                "time": np.float64(time.time())}


class SpoolWriter:
    def __init__(self, spool_dir: Path):
        self.dir = Path(spool_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=8)
        self.seq = 0
        self.written = 0
        self._t = threading.Thread(target=self._loop, name="hsai-spool", daemon=True)
        self._t.start()

    def submit(self, buf: RolloutBuffer) -> None:
        d = buf.to_dict()
        try:
            self.q.put(d, timeout=2.0)
        except queue.Full:
            pass  # learner is far behind; drop this rollout

    def _loop(self) -> None:
        while True:
            d = self.q.get()
            if d is None:
                return
            self.seq += 1
            name = f"roll_{int(d['version']):06d}_{self.seq:08d}"
            tmp = self.dir / (name + ".tmp.npz")
            final = self.dir / (name + ".npz")
            try:
                np.savez(tmp, **d)
                os.replace(tmp, final)
                self.written += 1
            except Exception:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def close(self) -> None:
        self.q.put(None)


def list_spool(spool_dir: Path) -> List[Path]:
    return sorted(p for p in Path(spool_dir).glob("roll_*.npz") if not p.name.endswith(".tmp.npz"))


def load_rollout(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["episodes"] = json.loads(bytes(d["episodes"]).decode("utf-8")) if d["episodes"].size else []
        d["bootstrap_value"] = float(d["bootstrap_value"])
        d["version"] = int(d["version"])
        return d
    except Exception:
        return None
