"""Stateful, safe input controller.

The agent sets a target (mouse velocity + key/button hold set) once per tick;
a high-priority thread converts that into smooth sub-tick mouse deltas and
diff-applies key states.  Safety features:

* kill switch key (default F12): releases everything and raises the stop flag
* pause key (default F11): toggles pause (everything released while paused)
* foreground check: nothing is sent unless the game window is in front
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from ..util.timing import enable_high_res_timer, precise_sleep_until
from ..util.win32 import (find_game_window, foreground_window, key_is_down, set_thread_priority,
                          vk_from_name, IS_WINDOWS)
from .sendinput import SendInput


@dataclass
class ActionCommand:
    mouse_vx: float = 0.0            # pixels / second
    mouse_vy: float = 0.0
    keys: Set[str] = field(default_factory=set)      # key names held (e.g. {"W", "SHIFT"})
    buttons: Set[str] = field(default_factory=set)   # {"L", "R"}


class InputController:
    def __init__(self, subticks_per_tick: int = 4, tick_hz: float = 60.0, smoothing: float = 0.35,
                 kill_switch: str = "F12", pause_key: str = "F11", require_foreground: bool = True,
                 window_title_contains: str = "Half Sword", process_name: str = ""):
        self.sender = SendInput()
        self.rate_hz = subticks_per_tick * tick_hz
        self.smoothing = float(smoothing)
        self.kill_vk = vk_from_name(kill_switch)
        self.pause_vk = vk_from_name(pause_key)
        self.require_foreground = require_foreground and IS_WINDOWS
        self.window_title_contains = window_title_contains
        self.process_name = process_name
        self._target = ActionCommand()
        self._lock = threading.Lock()
        self._held_keys: Set[str] = set()
        self._held_buttons: Set[str] = set()
        self._vx = 0.0
        self._vy = 0.0
        self._accx = 0.0
        self._accy = 0.0
        self._stop = threading.Event()
        self.killed = threading.Event()
        self.paused = False
        self._pause_edge = False
        self.game_hwnd: Optional[int] = None
        self._hwnd_checked = 0.0
        self.game_in_front = False
        self.enabled = False
        self._thread: Optional[threading.Thread] = None
        self.stats_moves = 0

    # ---- lifecycle ------------------------------------------------------
    def start(self) -> "InputController":
        enable_high_res_timer()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="hsai-input", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.release_all()

    # ---- agent-facing API ----------------------------------------------
    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        if not on:
            self.release_all()

    def set_target(self, cmd: ActionCommand) -> None:
        with self._lock:
            self._target = cmd

    def release_all(self) -> None:
        with self._lock:
            self._target = ActionCommand()
            for k in list(self._held_keys):
                self.sender.key(k, False)
            self._held_keys.clear()
            for b in list(self._held_buttons):
                self.sender.mouse_button(b, False)
            self._held_buttons.clear()
            self._vx = self._vy = self._accx = self._accy = 0.0

    def tap(self, key: str, seconds: float = 0.08) -> None:
        """Blocking key tap outside the agent loop (macros / resets)."""
        self.sender.key(key, True)
        time.sleep(seconds)
        self.sender.key(key, False)

    def hold(self, key: str, seconds: float) -> None:
        self.sender.key(key, True)
        time.sleep(seconds)
        self.sender.key(key, False)

    # ---- safety ---------------------------------------------------------
    def _check_window(self) -> bool:
        if not self.require_foreground:
            self.game_in_front = True
            return True
        now = time.perf_counter()
        if self.game_hwnd is None or now - self._hwnd_checked > 2.0:
            self._hwnd_checked = now
            self.game_hwnd = find_game_window(self.window_title_contains, self.process_name)
        self.game_in_front = self.game_hwnd is not None and foreground_window() == self.game_hwnd
        return self.game_in_front

    def _check_hotkeys(self) -> None:
        if key_is_down(self.kill_vk):
            self.killed.set()
            self.enabled = False
        p = key_is_down(self.pause_vk)
        if p and not self._pause_edge:
            self.paused = not self.paused
        self._pause_edge = p

    # ---- worker thread --------------------------------------------------
    def _loop(self) -> None:
        set_thread_priority(critical=True)
        period = 1.0 / self.rate_hz
        next_t = time.perf_counter()
        n = 0
        while not self._stop.is_set():
            next_t += period
            precise_sleep_until(next_t)
            if time.perf_counter() - next_t > 0.05:
                next_t = time.perf_counter()
            n += 1
            if n % 8 == 0:
                self._check_hotkeys()
                self._check_window()
            active = self.enabled and not self.paused and not self.killed.is_set() and self.game_in_front
            if not active:
                if self._held_keys or self._held_buttons:
                    self.release_all()
                continue
            with self._lock:
                tgt = self._target
                # smooth commanded velocity
                a = 1.0 - self.smoothing
                self._vx += a * (tgt.mouse_vx - self._vx)
                self._vy += a * (tgt.mouse_vy - self._vy)
                self._accx += self._vx * period
                self._accy += self._vy * period
                dx = int(math.trunc(self._accx))
                dy = int(math.trunc(self._accy))
                self._accx -= dx
                self._accy -= dy
                if dx or dy:
                    self.sender.mouse_move(dx, dy)
                    self.stats_moves += 1
                # keys
                for k in tgt.keys - self._held_keys:
                    self.sender.key(k, True)
                for k in self._held_keys - tgt.keys:
                    self.sender.key(k, False)
                self._held_keys = set(tgt.keys)
                for b in tgt.buttons - self._held_buttons:
                    self.sender.mouse_button(b, True)
                for b in self._held_buttons - tgt.buttons:
                    self.sender.mouse_button(b, False)
                self._held_buttons = set(tgt.buttons)
