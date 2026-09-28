"""Factory for the real game environment (Windows)."""
from __future__ import annotations

from typing import Callable

from ..capture import FramePreprocessor, ScreenCapture
from ..config import Config
from ..env.actions import ActionSpace
from ..env.features import StateFeaturizer
from ..env.half_sword_env import HalfSwordEnv
from ..input.controller import InputController
from ..telemetry import CommandWriter, TelemetryReader
from ..util.win32 import make_dpi_aware, screen_size, set_process_priority


def capture_region(cfg: Config):
    if not cfg.capture.region:
        return None
    w, h = screen_size()
    l, t, r, b = cfg.capture.region
    return (int(l * w), int(t * h), int(r * w), int(b * h))


def make_telemetry(cfg: Config) -> TelemetryReader:
    return TelemetryReader(cfg.telemetry.transport, cfg.telemetry.pipe_name, cfg.telemetry.file_path,
                           bone_names=cfg.obs.bones, stale_ms=cfg.telemetry.stale_ms).start()


def make_controller(cfg: Config) -> InputController:
    return InputController(cfg.input.mouse_subticks, cfg.game.target_fps, cfg.input.mouse_smoothing,
                           cfg.input.kill_switch_key, cfg.input.pause_key, cfg.input.require_foreground,
                           cfg.game.window_title_contains, cfg.game.process_name).start()


def make_game_env(cfg: Config, space: ActionSpace, feat: StateFeaturizer, log: Callable[[str], None] = print,
                  device: str = "cuda") -> HalfSwordEnv:
    make_dpi_aware()
    set_process_priority(high=True)
    capture = ScreenCapture(cfg.capture.backend, cfg.capture.monitor_index, cfg.capture.target_fps, capture_region(cfg)).start()
    log(f"[capture] {capture.backend} {capture.width}x{capture.height} @ {cfg.capture.target_fps} fps")
    preproc = FramePreprocessor(cfg.obs.size, cfg.obs.stack, cfg.obs.gray, cfg.obs.crop, device=device)
    telemetry = make_telemetry(cfg)
    commands = CommandWriter(cfg.telemetry.cmd_path)
    controller = make_controller(cfg)
    return HalfSwordEnv(cfg, capture, preproc, telemetry, controller, commands, space, feat, log=log)
