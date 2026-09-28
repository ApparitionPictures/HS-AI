"""Record your own play (frames, state, and your inputs) for behaviour cloning.

Files land in recordings/ with the same layout as training rollouts.  Only
frames during a live fight are recorded.  Stop with the kill-switch key."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..capture import FramePreprocessor, ScreenCapture
from ..config import Config
from ..env.episode import EpisodeTracker, is_fight_live
from ..env.reward import Outcome, RewardShaper
from ..input.rawinput import RawInputListener
from ..rl.actor import build_dims
from ..rl.spool import RolloutBuffer, SpoolWriter
from ..util.win32 import VK, key_is_down, vk_from_name
from .game_env import capture_region, make_telemetry, preproc_crop


def run_record(cfg: Config, log: Callable[[str], None] = print, minutes: float = 0.0, device: str = "cuda") -> None:
    import torch
    dev = device if torch.cuda.is_available() else "cpu"
    space, feat, img_ch = build_dims(cfg)
    tel = make_telemetry(cfg)
    cap = ScreenCapture(cfg.capture.backend, cfg.capture.monitor_index, cfg.capture.target_fps, capture_region(cfg)).start()
    pre = FramePreprocessor(cfg.obs.size, cfg.obs.stack, cfg.obs.gray, preproc_crop(cfg), device=dev)
    raw = RawInputListener().start()
    out_dir = cfg.resolve_path(cfg.paths.recordings_dir)
    writer = SpoolWriter(out_dir)
    reward = RewardShaper(cfg.reward, 1.0 / cfg.game.target_fps)
    tracker = EpisodeTracker(cfg.episode, 1.0 / cfg.game.target_fps)
    kill_vk = vk_from_name(cfg.input.kill_switch_key)
    keys = space.human_keys_of_interest()
    T = cfg.record.chunk_steps
    buf = RolloutBuffer(T, (img_ch, cfg.obs.size, cfg.obs.size), feat.dim, space.n_discrete, 1)
    t_end = time.perf_counter() + minutes * 60 if minutes else None
    dt = 1.0 / cfg.game.target_fps
    live = False
    prev_action = None
    steps = 0
    t_last = time.perf_counter()
    log(f"recording to {out_dir}. Play normally; press {cfg.input.kill_switch_key} to stop.")
    try:
        while True:
            if key_is_down(kill_vk) or (t_end and time.perf_counter() > t_end):
                break
            frame = cap.get_frame()
            if frame is None:
                continue
            now = time.perf_counter()
            real_dt = max(1e-3, now - t_last)
            t_last = now
            st = tel.latest()
            obs = pre.push(frame)
            fight = is_fight_live(st) and tel.is_fresh()
            if fight and not live:
                live = True
                feat.reset(); reward.reset(st); tracker.reset(st); prev_action = None
                first = True
                log("fight started, recording")
            elif not fight and live:
                live = False
                log(f"fight ended ({steps} steps so far)")
            if not live:
                continue
            dx, dy, _ = raw.consume()
            held = {k for k in keys if key_is_down(VK.get(k.upper(), vk_from_name(k)))}
            buttons = {b for b in ("L", "R") if key_is_down(VK[b + "BUTTON"])}
            mouse, disc = space.from_human(held, buttons, dx / real_dt, dy / real_dt)
            state = feat(st, prev_action, tracker.t / cfg.episode.max_seconds, reward.t_since_dealt, reward.t_since_taken)
            r = reward.step(st)
            outcome = tracker.update(st)
            done = outcome != Outcome.NONE
            buf.add(obs.cpu().numpy(), state, mouse, disc, 0.0, r, done, first, 0.0, np.zeros(1, np.float16))
            first = False
            prev_action = space.prev_action_vector(mouse, disc)
            steps += 1
            if done:
                live = False
                log(f"episode {outcome.value}: dealt {reward.stats.damage_dealt:.0f} taken {reward.stats.damage_taken:.0f}")
            if buf.full:
                writer.submit(buf)
                buf = RolloutBuffer(T, (img_ch, cfg.obs.size, cfg.obs.size), feat.dim, space.n_discrete, 1)
    except KeyboardInterrupt:
        pass
    finally:
        if buf.n > 32:
            writer.submit(buf)
        writer.close()
        time.sleep(1.0)
        raw.stop()
        cap.stop()
        tel.stop()
        log(f"recorded {steps} steps ({steps / cfg.game.target_fps / 60:.1f} min of fighting)")
