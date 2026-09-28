"""The real-time Half Sword environment.

One ``step`` = apply the action, block until the next captured frame (which
paces the loop at the capture rate), read the newest telemetry, compute the
reward and terminal state, and return the observation pair
(stacked frames on the GPU, state vector on the CPU).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from ..capture import FramePreprocessor, ScreenCapture
from ..config import Config
from ..input.controller import ActionCommand, InputController
from ..macro import MacroPlayer, load_macro, simple_macro
from ..telemetry import CommandWriter, GameState, TelemetryReader
from .actions import ActionSpace
from .episode import EpisodeTracker, is_fight_live
from .features import StateFeaturizer
from .reward import Outcome, RewardShaper


class KillSwitch(Exception):
    pass


class HalfSwordEnv:
    def __init__(self, cfg: Config, capture: ScreenCapture, preproc: FramePreprocessor, telemetry: TelemetryReader,
                 controller: InputController, commands: CommandWriter, action_space: ActionSpace,
                 featurizer: StateFeaturizer, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.capture = capture
        self.preproc = preproc
        self.telemetry = telemetry
        self.controller = controller
        self.commands = commands
        self.space = action_space
        self.featurizer = featurizer
        self.log = log
        self.dt = 1.0 / cfg.game.target_fps
        self.reward = RewardShaper(cfg.reward, self.dt)
        self.tracker = EpisodeTracker(cfg.episode, self.dt)
        self.macros_dir = cfg.resolve_path(cfg.paths.macros_dir)
        self.player = MacroPlayer(controller.sender, abort=lambda: controller.killed.is_set())
        self.episode_index = 0
        self.last_outcome = Outcome.NONE
        self.last_state: Optional[GameState] = None
        self.prev_action_vec: Optional[np.ndarray] = None
        self.t = 0.0
        self.step_ms = 0.0
        self.stale_steps = 0
        self.total_steps = 0

    # ------------------------------------------------------------------
    def _check_kill(self) -> None:
        if self.controller.killed.is_set():
            raise KillSwitch("kill switch pressed")

    def wait_if_paused(self) -> None:
        while self.controller.paused and not self.controller.killed.is_set():
            time.sleep(0.05)
        self._check_kill()

    def _fresh_state(self, timeout: float = 0.0) -> Optional[GameState]:
        if timeout > 0:
            self.telemetry.wait_new(timeout)
        if self.telemetry.is_fresh():
            return self.telemetry.latest()
        return None

    def _require_telemetry(self, timeout_s: float = 15.0) -> None:
        t0 = time.perf_counter()
        while not self.telemetry.is_fresh():
            self._check_kill()
            if time.perf_counter() - t0 > timeout_s:
                raise RuntimeError(
                    "No telemetry from the game. Is Half Sword running with the HSAI mod installed? "
                    "Run `hsai doctor` to check.")
            time.sleep(0.1)

    # ------------------------------------------------------------------
    def _run_reset_procedure(self) -> bool:
        c = self.cfg.episode
        self.controller.set_enabled(False)
        self.controller.release_all()
        if self.last_outcome == Outcome.SURRENDER:
            self.log("holding G to surrender")
            self.controller.hold(self.cfg.input.keys.surrender, 3.0)
            time.sleep(1.0)
        if self.episode_index > 0:
            time.sleep(c.post_terminal_wait_s)
        st = self._fresh_state()
        if is_fight_live(st) and self.last_outcome == Outcome.NONE:
            return True  # first episode, the user already started a fight
        for attempt in range(3):
            self._check_kill()
            if c.reset_mode == "mod":
                self.log("mod reset: heal + respawn enemy")
                self.commands.heal()
                self.commands.kill_enemies()
                if c.mod_reset_enemy_class:
                    self.commands.despawn_spawned()
                    time.sleep(0.5)
                    self.commands.spawn(c.mod_reset_enemy_class, c.mod_reset_spawn_distance_m)
                else:
                    self.commands.heal_enemies()
            else:
                macro = load_macro(self.macros_dir, c.macro_after_fight)
                if macro is None:
                    self.log(f"macro '{c.macro_after_fight}' not found, using built-in: hold G 3s")
                    macro = simple_macro([{"wait": 0.5}, {"hold": self.cfg.input.keys.surrender, "seconds": 3.0}])
                self.log(f"playing macro '{c.macro_after_fight}' (attempt {attempt + 1})")
                if not self.player.play(macro):
                    raise KillSwitch("aborted during macro")
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < c.fight_start_timeout_s:
                self._check_kill()
                st = self._fresh_state(timeout=0.1)
                if is_fight_live(st):
                    time.sleep(0.3)
                    st = self._fresh_state()
                    if is_fight_live(st):
                        return True
                time.sleep(0.05)
            self.log("fight did not start, retrying the reset procedure")
        return False

    def reset(self) -> Tuple[torch.Tensor, np.ndarray, Dict]:
        self.wait_if_paused()
        self._require_telemetry()
        ok = self._run_reset_procedure()
        if not ok:
            raise RuntimeError("could not start a new fight (check your 'after_fight' macro with `hsai macro play after_fight`)")
        if self.cfg.episode.lock_on_at_start:
            time.sleep(0.2)
            self.controller.tap(self.cfg.input.keys.lock_on)
        if self.cfg.game.time_dilation != 1.0:
            self.commands.time_dilation(self.cfg.game.time_dilation)
        st = self._fresh_state(timeout=0.2) or self.telemetry.latest()
        self.featurizer.reset()
        self.reward.reset(st)
        self.tracker.reset(st)
        self.preproc.reset()
        for _ in range(self.preproc.stack):
            frame = self.capture.get_frame()
            if frame is not None:
                img = self.preproc.push(frame)
        self.t = 0.0
        self.prev_action_vec = None
        self.last_state = st
        self.last_outcome = Outcome.NONE
        self.episode_index += 1
        state_vec = self.featurizer(st, None, 0.0, 99.0, 99.0) if st else np.zeros(self.featurizer.dim, np.float32)
        self.controller.set_enabled(True)
        return self.preproc.obs, state_vec, {"episode": self.episode_index, "map": st.map_name if st else ""}

    # ------------------------------------------------------------------
    def step(self, mouse: np.ndarray, discrete: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, float, bool, Dict]:
        self._check_kill()
        t0 = time.perf_counter()
        cmd: ActionCommand = self.space.to_command(mouse, discrete)
        self.controller.set_target(cmd)
        frame = self.capture.get_frame()
        st = self.telemetry.latest()
        stale = not self.telemetry.is_fresh()
        if stale:
            self.stale_steps += 1
            st = self.last_state if st is None else st
        img = self.preproc.push(frame) if frame is not None else self.preproc.obs
        self.t += self.dt
        r = self.reward.step(st) if st is not None else 0.0
        surrender = self.space.is_surrender(discrete)
        outcome = self.tracker.update(st, surrender_pressed=surrender)
        done = outcome != Outcome.NONE
        info: Dict = {"outcome": outcome.value, "stale": stale}
        if done:
            r += self.reward.terminal(outcome, st)
            self.last_outcome = outcome
            self.controller.release_all()
            self.controller.set_enabled(False)
            s = self.reward.stats
            info.update({"damage_dealt": s.damage_dealt, "damage_taken": s.damage_taken, "ko_dealt": s.ko_dealt,
                         "ko_taken": s.ko_taken, "steps": s.steps, "seconds": s.seconds, "end_health": s.end_health,
                         "min_health": s.min_health, "reward_sum": s.reward_sum, "reward_terminal": s.reward_terminal,
                         "kite_steps": s.kite_steps, "components": dict(s.components)})
        self.prev_action_vec = self.space.prev_action_vector(mouse, discrete)
        t_frac = self.t / max(1e-3, self.cfg.episode.max_seconds)
        state_vec = (self.featurizer(st, self.prev_action_vec, t_frac, self.reward.t_since_dealt, self.reward.t_since_taken)
                     if st is not None else np.zeros(self.featurizer.dim, np.float32))
        self.last_state = st
        self.total_steps += 1
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        info["step_ms"] = self.step_ms
        return img, state_vec, float(r), done, info

    def close(self) -> None:
        self.controller.release_all()
        self.controller.set_enabled(False)
