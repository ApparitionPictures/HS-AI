"""Command line interface.  Run ``hsai`` (or ``python -m hsai``) with no
arguments for the list of commands."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from .config import Config, load_config, parse_override, save_user_config


def _log_factory(run_log: Path | None = None):
    def log(s: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        if run_log is not None:
            try:
                with open(run_log, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
    return log


def _cfg(args) -> Config:
    overrides: Dict[str, Any] = {}
    for o in getattr(args, "override", None) or []:
        overrides.update(parse_override(o))
    return load_config(getattr(args, "config", None), overrides)


# ---------------------------------------------------------------------------
def cmd_setup(args) -> int:
    log = _log_factory()
    cfg = _cfg(args)
    from .tools.modinstall import install_all, local_dir
    log("== HS-AI setup ==")
    log("1/3 installing UE4SS + telemetry mod into the game folder")
    try:
        install_all(log, cfg.game.install_dir, cfg.game.steam_app_id, hide_console=not args.console, prefer_stable=args.stable)
    except Exception as e:
        log(f"mod install failed: {e}")
        log("You can retry later with:  hsai mod install --dir \"<Half Sword folder>\"")
    local_dir().mkdir(parents=True, exist_ok=True)
    log("2/3 system check")
    from .tools.doctor import run_doctor
    run_doctor(cfg, log, quick=True, bench=False)
    log("3/3 next steps")
    log("  a) Start Half Sword (borderless window, 1080p or 1440p, mouse sensitivity low), enter a fight, then run:  hsai doctor")
    log("  b) Record what you do after a fight ends to start the next one:   hsai macro record after_fight")
    log("  c) Optional but recommended: record 15+ minutes of your own fights:  hsai record   then  hsai pretrain")
    log("  d) Train:  hsai train --hours 2        Watch the best policy:  hsai play")
    return 0


def cmd_doctor(args) -> int:
    from .tools.doctor import run_doctor
    rep = run_doctor(_cfg(args), _log_factory(), quick=args.quick, bench=not args.no_bench)
    return 1 if rep.failures else 0


def cmd_mod(args) -> int:
    log = _log_factory()
    cfg = _cfg(args)
    from .tools import modinstall
    from .tools.gamefind import find_game_dir, find_win64_dir
    if args.action == "install":
        modinstall.install_all(log, args.dir or cfg.game.install_dir, cfg.game.steam_app_id,
                               zip_path=Path(args.zip) if args.zip else None, hide_console=not args.console,
                               prefer_stable=args.stable)
        return 0
    game = find_game_dir(cfg.game.steam_app_id, args.dir or cfg.game.install_dir)
    if game is None:
        log("game not found")
        return 1
    win64 = find_win64_dir(game)
    if args.action == "uninstall":
        modinstall.uninstall_mod(win64, log, remove_ue4ss=args.all)
    else:
        log(f"game: {game}")
        log(f"ue4ss: {'installed' if modinstall.ue4ss_installed(win64) else 'missing'}  layout root: {modinstall.ue4ss_root(win64)}")
        log(f"hsai mod: {'installed' if modinstall.mod_installed(win64) else 'missing'}, enabled: {modinstall.mod_enabled(win64)}")
        st = modinstall.local_dir() / "mod_status.txt"
        log(f"mod status file: {st.read_text() if st.exists() else 'not written yet (game never started with the mod)'}")
    return 0


def cmd_watch(args) -> int:
    from .tools.watch import run_watch
    run_watch(_cfg(args), _log_factory(), seconds=args.seconds)
    return 0


def cmd_macro(args) -> int:
    log = _log_factory()
    cfg = _cfg(args)
    from .macro import MacroPlayer, MacroRecorder, load_macro, save_macro
    mdir = cfg.resolve_path(cfg.paths.macros_dir)
    if args.action == "list":
        for p in sorted(mdir.glob("*.json")):
            m = json.loads(p.read_text(encoding="utf-8"))
            log(f"{p.stem}: {m.get('duration', 0):.1f} s, {len(m.get('events', []))} events")
        return 0
    if args.action == "record":
        log(f"Recording macro '{args.name}'. Switch to the game now.")
        log(f"Recording starts in {args.countdown} s and stops when you press {args.stop_key} (or after {args.max_seconds} s).")
        log("Do exactly what you do after a fight ends to get into the next fight (hold G, click menus, ...).")
        for i in range(args.countdown, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1.0)
        rec = MacroRecorder(stop_key=args.stop_key, max_seconds=args.max_seconds, log=log)
        m = rec.record()
        p = save_macro(mdir, args.name, m)
        log(f"saved {p}: {m['duration']:.1f} s, {len(m['events'])} events. Test it with: hsai macro play {args.name}")
        return 0
    if args.action == "play":
        m = load_macro(mdir, args.name)
        if m is None:
            log(f"macro '{args.name}' not found in {mdir}")
            return 1
        log(f"playing '{args.name}' in {args.countdown} s, switch to the game...")
        time.sleep(args.countdown)
        MacroPlayer().play(m, speed=args.speed)
        log("done")
        return 0
    return 1


def cmd_record(args) -> int:
    from .tools.record import run_record
    run_record(_cfg(args), _log_factory(), minutes=args.minutes)
    return 0


def cmd_pretrain(args) -> int:
    log = _log_factory()
    cfg = _cfg(args)
    import torch
    from .model.policy import build_policy
    from .rl.actor import build_dims
    from .rl.bc import load_recordings, train_bc
    from .rl.checkpoint import save_checkpoint
    from .rl.spool import list_spool
    rec_dir = cfg.resolve_path(cfg.paths.recordings_dir)
    files = list_spool(rec_dir)
    if not files:
        log(f"no recordings in {rec_dir}. Run `hsai record` first.")
        return 1
    space, feat, img_ch = build_dims(cfg)
    policy = build_policy(img_ch, cfg.obs.size, feat.dim, space.discrete_sizes, cfg.model)
    recs = load_recordings(files)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_bc(policy, recs, epochs=args.epochs, seq_len=cfg.ppo.seq_len, batch=args.batch, lr=args.lr, device=dev, log=log)
    out = Path(args.out) if args.out else cfg.resolve_path(cfg.paths.checkpoints_dir) / "bc.pt"
    save_checkpoint(out, policy.cpu(), meta={"kind": "bc", "files": len(files)})
    log(f"saved {out}. Start RL from it with:  hsai train --init \"{out}\"")
    return 0


def _make_env_factory(args, device: str):
    if args.sim:
        from .tools.sim import make_sim_env
        return lambda c, s, f, l: make_sim_env(c, s, f, l, realtime=args.realtime, device=device)
    from .tools.game_env import make_game_env
    return lambda c, s, f, l: make_game_env(c, s, f, l, device=device)


def cmd_train(args) -> int:
    cfg = _cfg(args)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = cfg.resolve_path(cfg.paths.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    log = _log_factory(runs / "train.log")
    from .rl.train import train
    if not args.sim:
        if sys.platform != "win32":
            log("the real game can only be driven on Windows; use --sim here")
            return 1
        log(f"Training for {args.hours} h. Make sure Half Sword is running and in the foreground. F11 pauses, F12 stops.")
    train(cfg, _make_env_factory(args, device), hours=args.hours, init=args.init, resume=args.resume, log=log,
          max_episodes=args.episodes, dashboard=not args.no_dashboard, device=device)
    return 0


def cmd_play(args) -> int:
    cfg = _cfg(args)
    import torch
    from .rl.actor import ActorLoop, build_dims
    from .rl.checkpoint import load_checkpoint, policy_from_checkpoint
    log = _log_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck_path = Path(args.ckpt) if args.ckpt else None
    if ck_path is None:
        runs = cfg.resolve_path(cfg.paths.runs_dir)
        cands = sorted(runs.glob("*/elite.pt"), key=lambda p: p.stat().st_mtime) if runs.exists() else []
        if not cands:
            cands = sorted(runs.glob("*/weights/latest.pt"), key=lambda p: p.stat().st_mtime) if runs.exists() else []
        if not cands:
            log("no checkpoint found; pass --ckpt")
            return 1
        ck_path = cands[-1]
    log(f"loading {ck_path}")
    policy = policy_from_checkpoint(load_checkpoint(ck_path))
    space, feat, _ = build_dims(cfg)
    env = _make_env_factory(args, device)(cfg, space, feat, log)
    status: Dict[str, Any] = {}
    loop = ActorLoop(cfg, env, policy, space, None, status, log=log, learn=False, device=device)
    loop.run(hours=args.hours, max_episodes=args.episodes)
    return 0


def cmd_bench(args) -> int:
    from .tools.doctor import run_doctor
    run_doctor(_cfg(args), _log_factory(), quick=True, bench=True)
    return 0


def cmd_config(args) -> int:
    log = _log_factory()
    if args.action == "set":
        values: Dict[str, Any] = {}
        for kv in args.pairs:
            d = parse_override(kv)
            for dotted, v in d.items():
                node = values
                parts = dotted.split(".")
                for p in parts[:-1]:
                    node = node.setdefault(p, {})
                node[parts[-1]] = v
        save_user_config(values)
        log(f"saved to configs/user.yaml: {values}")
    else:
        from .config import to_dict
        import yaml
        print(yaml.safe_dump(to_dict(_cfg(args)), sort_keys=False))
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hsai", description="HS-AI: real-time reinforcement learning agent for Half Sword")
    p.add_argument("--config", help="path to a user config yaml (default configs/user.yaml)")
    p.add_argument("-o", "--override", action="append", help="override, e.g. -o ppo.lr=1e-4 (repeatable)")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("setup", help="install the game mod and check the system (run this first)")
    s.add_argument("--console", action="store_true", help="keep the UE4SS console windows visible")
    s.add_argument("--stable", action="store_true", help="use the stable UE4SS release instead of experimental-latest")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("doctor", help="check GPU, game, mod, telemetry, capture and inference speed")
    s.add_argument("--quick", action="store_true")
    s.add_argument("--no-bench", action="store_true")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("mod", help="install / uninstall / status of the UE4SS telemetry mod")
    s.add_argument("action", choices=["install", "uninstall", "status"])
    s.add_argument("--dir", help="Half Sword install folder (if Steam auto-detection fails)")
    s.add_argument("--zip", help="use a local UE4SS zip instead of downloading")
    s.add_argument("--all", action="store_true", help="uninstall: also remove UE4SS")
    s.add_argument("--console", action="store_true", help="keep the UE4SS console windows visible")
    s.add_argument("--stable", action="store_true", help="use the stable UE4SS release instead of experimental-latest")
    s.set_defaults(fn=cmd_mod)

    s = sub.add_parser("watch", help="show what the agent sees (frames + telemetry)")
    s.add_argument("--seconds", type=float, default=0.0)
    s.set_defaults(fn=cmd_watch)

    s = sub.add_parser("macro", help="record/play input macros for menu navigation")
    s.add_argument("action", choices=["record", "play", "list"])
    s.add_argument("name", nargs="?", default="after_fight")
    s.add_argument("--countdown", type=int, default=5)
    s.add_argument("--stop-key", default="F8")
    s.add_argument("--max-seconds", type=float, default=120.0)
    s.add_argument("--speed", type=float, default=1.0)
    s.set_defaults(fn=cmd_macro)

    s = sub.add_parser("record", help="record your own fights for behaviour cloning")
    s.add_argument("--minutes", type=float, default=0.0)
    s.set_defaults(fn=cmd_record)

    s = sub.add_parser("pretrain", help="behaviour-clone the policy from your recordings")
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_pretrain)

    s = sub.add_parser("train", help="train with PPO against the live game")
    s.add_argument("--hours", type=float, default=2.0)
    s.add_argument("--init", help="start from a checkpoint (e.g. checkpoints/bc.pt)")
    s.add_argument("--resume", help="continue a run directory")
    s.add_argument("--episodes", type=int, help="stop after N episodes")
    s.add_argument("--sim", action="store_true", help="offline simulator instead of the game (pipeline test)")
    s.add_argument("--realtime", action="store_true", help="sim: run at real time")
    s.add_argument("--no-dashboard", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("play", help="run a trained policy without learning")
    s.add_argument("--ckpt")
    s.add_argument("--hours", type=float)
    s.add_argument("--episodes", type=int)
    s.add_argument("--sim", action="store_true")
    s.add_argument("--realtime", action="store_true")
    s.set_defaults(fn=cmd_play)

    s = sub.add_parser("bench", help="inference backend benchmark")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("config", help="show or set configuration")
    s.add_argument("action", choices=["show", "set"])
    s.add_argument("pairs", nargs="*", help="key=value pairs for set, e.g. game.time_dilation=0.7")
    s.set_defaults(fn=cmd_config)
    return p


def main(argv: List[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        p.print_help()
        sys.exit(0)
    try:
        sys.exit(args.fn(args))
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
