"""Training orchestrator: prepares the run directory and the initial weights,
launches the learner process, and runs the actor in this process."""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..config import Config, to_dict
from ..model.policy import build_policy
from .actor import ActorLoop, build_dims
from .checkpoint import load_checkpoint, policy_from_checkpoint
from .learner import learner_main, publish_weights


def make_run_dir(cfg: Config, resume: Optional[str] = None) -> Path:
    runs = cfg.resolve_path(cfg.paths.runs_dir)
    if resume:
        p = Path(resume) if Path(resume).is_absolute() else runs / resume
        if not p.exists():
            raise FileNotFoundError(f"run directory {p} not found")
        return p
    p = runs / time.strftime("%Y%m%d_%H%M%S")
    p.mkdir(parents=True, exist_ok=True)
    return p


def init_weights(cfg: Config, run_dir: Path, init: Optional[str], resume: bool) -> None:
    """Create run_dir/weights/latest.pt (version 0) from a checkpoint or a fresh policy."""
    wp = run_dir / "weights" / "latest.pt"
    if resume and wp.exists():
        return
    space, feat, img_channels = build_dims(cfg)
    extra: Dict[str, Any] = {}
    if init:
        ck = load_checkpoint(Path(init))
        policy = policy_from_checkpoint(ck)
        if cfg.ppo.bc_kl_coef > 0:
            extra["bc_prior_path"] = str(Path(init).resolve())
        expected = (img_channels, cfg.obs.size, feat.dim, list(space.discrete_sizes))
        got = (policy.spec.img_channels, policy.spec.img_size, policy.spec.state_dim, list(policy.spec.discrete_sizes))
        if expected != got:
            raise ValueError(f"checkpoint {init} was trained with a different observation/action layout {got} != {expected}")
    else:
        policy = build_policy(img_channels, cfg.obs.size, feat.dim, space.discrete_sizes, cfg.model)
    hyper = {"lr": cfg.ppo.lr, "entropy_discrete": cfg.ppo.entropy_discrete, "entropy_continuous": cfg.ppo.entropy_continuous,
             "beta": cfg.explore.beta, "scale": cfg.explore.scale}
    publish_weights(wp, policy, 0, hyper, extra)


def train(cfg: Config, make_env, hours: Optional[float] = None, init: Optional[str] = None, resume: Optional[str] = None,
          log=print, max_episodes: Optional[int] = None, dashboard: bool = True, device: str = "cuda") -> Path:
    run_dir = make_run_dir(cfg, resume)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
    init_weights(cfg, run_dir, init, resume is not None)
    spool = cfg.resolve_path(cfg.paths.spool_dir)
    if spool.exists():
        shutil.rmtree(spool, ignore_errors=True)
    spool.mkdir(parents=True, exist_ok=True)
    stop_flag = run_dir / "stop.flag"
    if stop_flag.exists():
        stop_flag.unlink()
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=learner_main, args=(to_dict(cfg), str(run_dir), os.getpid()), name="hsai-learner", daemon=True)
    proc.start()
    log(f"[train] run directory: {run_dir}")
    log(f"[train] learner process pid {proc.pid}")
    ck = load_checkpoint(run_dir / "weights" / "latest.pt")
    policy = policy_from_checkpoint(ck)
    space, feat, _ = build_dims(cfg)
    env = make_env(cfg, space, feat, log)
    status: Dict[str, Any] = {"generation": 0}
    dash = None
    if dashboard:
        from ..dashboard import Dashboard
        dash = Dashboard(run_dir, status).start()
    try:
        loop = ActorLoop(cfg, env, policy, space, run_dir, status, log=log, learn=True, device=device)
        loop.run(hours=hours, max_episodes=max_episodes)
    finally:
        stop_flag.touch()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
        if dash:
            dash.stop()
        log(f"[train] finished. Best policy: {run_dir / 'elite.pt'} (if any generation completed), latest: {run_dir / 'weights' / 'latest.pt'}")
    return run_dir
