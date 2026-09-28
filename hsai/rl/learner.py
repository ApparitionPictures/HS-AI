"""Learner process: consumes spool rollouts, runs PPO on the GPU, publishes
weights, and drives the generation/elite loop."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..config import Config, _from_dict
from .checkpoint import load_checkpoint, policy_from_checkpoint, save_checkpoint
from .generations import GenerationManager
from .ppo import PPOLearner
from .spool import list_spool, load_rollout


def _parent_alive(ppid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        SYNCHRONIZE = 0x00100000
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, ppid)
        if not h:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT -> still running
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        os.kill(ppid, 0)
        return True
    except OSError:
        return False


def publish_weights(path: Path, policy, version: int, hyper: Dict[str, float], extra: Optional[Dict[str, Any]] = None) -> None:
    save_checkpoint(path, policy, meta={"version": version, "time": time.time()}, extra={"hyper": hyper, **(extra or {})})


class LearnerRuntime:
    def __init__(self, cfg: Config, run_dir: Path, log=print):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.log = log
        self.spool_dir = cfg.resolve_path(cfg.paths.spool_dir)
        self.weights_path = self.run_dir / "weights" / "latest.pt"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ck = load_checkpoint(self.weights_path)
        self.policy = policy_from_checkpoint(ck)
        bc_prior = None
        if cfg.ppo.bc_kl_coef > 0 and ck.get("bc_prior_path"):
            try:
                bc_prior = policy_from_checkpoint(load_checkpoint(Path(ck["bc_prior_path"])))
            except Exception:
                bc_prior = None
        dev = cfg.learner.device
        if dev == "cuda" and torch.cuda.is_available() and cfg.learner.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.learner = PPOLearner(self.policy, cfg.ppo, device=dev, explore_beta=cfg.explore.beta,
                                  compile_model=cfg.learner.compile, channels_last=cfg.learner.channels_last,
                                  minibatch_sleep_ms=cfg.learner.minibatch_sleep_ms, bc_prior=bc_prior)
        self.learner.load_extra(ck)
        self.version = int(ck.get("meta", {}).get("version", 0))
        self.gm = GenerationManager(cfg.generations, self.run_dir, seed=cfg.seed)
        self.gm.load()
        self.status: Dict[str, Any] = {"updates": self.learner.updates, "message": "waiting for rollouts"}
        self._upd_csv = open(self.run_dir / "updates.csv", "a", encoding="utf-8")
        if self._upd_csv.tell() == 0:
            self._upd_csv.write("time,update,total_steps,loss_pi,loss_v,entropy_c,entropy_d,kl,clipfrac,lr,logstd,update_s\n")

    def write_status(self) -> None:
        p = self.run_dir / "status_learner.json"
        tmp = p.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.status, f)
            os.replace(tmp, p)
        except Exception:
            pass

    def publish(self) -> None:
        self.version += 1
        publish_weights(self.weights_path, self.policy, self.version, self.learner.hyper.to_dict(), self.learner.state_extra())

    def step(self) -> bool:
        """One learner iteration. Returns True if an update happened."""
        files = list_spool(self.spool_dir)
        if not files:
            return False
        maxb = self.cfg.learner.max_spool_backlog
        if len(files) > maxb:
            for p in files[:-maxb]:
                try:
                    os.remove(p)
                except OSError:
                    pass
            files = files[-maxb:]
        per_update = max(1, 1024 // max(1, self.cfg.ppo.rollout_steps))
        take = files[:per_update]
        rolls: List[Dict[str, Any]] = []
        episodes: List[Dict[str, Any]] = []
        for p in take:
            r = load_rollout(p)
            try:
                os.remove(p)
            except OSError:
                pass
            if r is None:
                continue
            rolls.append(r)
            episodes.extend(r.get("episodes", []))
        if not rolls:
            return False
        stats = self.learner.update(rolls)
        self.status.update({k: v for k, v in stats.items() if isinstance(v, (int, float))})
        self.status.update({"backlog": len(files) - len(take), "beta": self.learner.hyper.beta,
                            "elite_fitness": self.gm.elite.fitness if self.gm.elite else 0.0,
                            "elite_gen": self.gm.elite.index if self.gm.elite else 0, "reverts": self.gm.reverts,
                            "gen": len(self.gm.history) + 1, "message": "training"})
        if stats:
            self._upd_csv.write(",".join(str(x) for x in [time.strftime("%H:%M:%S"), self.learner.updates, int(stats["total_steps"]),
                                round(stats["loss_pi"], 5), round(stats["loss_v"], 5), round(stats["entropy_c"], 4),
                                round(stats["entropy_d"], 4), round(stats["kl"], 5), round(stats["clipfrac"], 4),
                                stats["lr"], round(stats["logstd"], 4), round(stats["update_s"], 3)]) + "\n")
            self._upd_csv.flush()
        if self.learner.updates % max(1, self.cfg.learner.weight_sync_every_updates) == 0:
            self.publish()
        # ---- generations ---------------------------------------------------
        self.gm.add_episodes([e for e in episodes if not e.get("eval")])
        if self.gm.ready():
            dec = self.gm.close_generation(self.learner.hyper.to_dict())
            rec = dec["record"]
            save_checkpoint(self.ckpt_dir / rec.checkpoint, self.policy, self.learner.opt,
                            meta={"version": self.version, "generation": rec.to_dict()}, extra=self.learner.state_extra())
            self.status["gen_fitness"] = rec.fitness
            msg = f"gen {rec.index}: fitness={rec.fitness:.3f} win={rec.win_rate:.2f} dealt={rec.dealt:.2f} taken={rec.taken:.2f}"
            if dec["is_elite"]:
                save_checkpoint(self.run_dir / "elite.pt", self.policy, self.learner.opt,
                                meta={"version": self.version, "generation": rec.to_dict()}, extra=self.learner.state_extra())
                msg += "  -> NEW ELITE"
            if dec["revert"]:
                elite_ck = load_checkpoint(self.run_dir / "elite.pt")
                self.policy.load_state_dict(elite_ck["state_dict"])
                self.learner.reset_optimizer()
                h = dec["new_hyper"]
                self.learner.hyper.lr = h["lr"]
                self.learner.hyper.entropy_discrete = h["entropy_discrete"]
                self.learner.hyper.entropy_continuous = h["entropy_continuous"]
                self.learner.hyper.beta = h["beta"]
                self.learner.set_lr(h["lr"])
                self.publish()
                msg += f"  -> REVERTED to elite gen {self.gm.elite.index}, new hyper {h}"
            # prune old generation checkpoints beyond top-k (+ last 2)
            keep = {r.checkpoint for r in self.gm.top_k()} | {r.checkpoint for r in self.gm.history[-2:]}
            for p in self.ckpt_dir.glob("gen_*.pt"):
                if p.name not in keep:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            self.log("[learner] " + msg)
            self.status["message"] = msg
        self.write_status()
        return True

    def close(self) -> None:
        self._upd_csv.close()


def learner_main(cfg_dict: Dict[str, Any], run_dir: str, parent_pid: int) -> None:
    cfg = _from_dict(Config, cfg_dict)
    run = Path(run_dir)
    log_path = run / "learner.log"

    def log(s: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + s + "\n")

    try:
        rt = LearnerRuntime(cfg, run, log=log)
        log(f"learner started on {rt.learner.device} with {rt.policy.count_params()} params")
        rt.write_status()
        idle = 0
        while True:
            if (run / "stop.flag").exists() or not _parent_alive(parent_pid):
                break
            did = rt.step()
            if not did:
                idle += 1
                time.sleep(0.2)
                if idle % 50 == 0:
                    rt.write_status()
            else:
                idle = 0
        rt.publish()
        rt.close()
        log("learner stopped")
    except Exception:
        log("learner crashed:\n" + traceback.format_exc())
        raise
