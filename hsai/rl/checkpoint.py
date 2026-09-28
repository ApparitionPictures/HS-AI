"""Checkpoint helpers: policy spec + weights + optimizer + normalizers + meta."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..model.policy import Policy, PolicySpec


def atomic_save(obj: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path: Path, policy: Policy, optimizer=None, meta: Optional[Dict[str, Any]] = None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    obj: Dict[str, Any] = {
        "spec": policy.spec.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "meta": meta or {},
    }
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if extra:
        obj.update(extra)
    atomic_save(obj, path)


def load_checkpoint(path: Path, map_location="cpu") -> Dict[str, Any]:
    for _ in range(5):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except (EOFError, RuntimeError, OSError):
            import time
            time.sleep(0.2)
    return torch.load(path, map_location=map_location, weights_only=False)


def policy_from_checkpoint(ck: Dict[str, Any]) -> Policy:
    p = Policy(PolicySpec.from_dict(ck["spec"]))
    p.load_state_dict(ck["state_dict"])
    return p
