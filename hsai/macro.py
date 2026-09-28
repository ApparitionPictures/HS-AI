"""Record and replay input macros (menu navigation such as "after a fight,
hold G, click Fight again").  Recording samples real keys/mouse at 100 Hz;
replay reproduces the same timing through SendInput.

Macro JSON format: {"version": 1, "duration": s, "events": [ {t, type, ...}, ... ]}
  type=key     key=NAME down=bool
  type=button  btn=L|R|M down=bool
  type=abs     x=0..1 y=0..1              (cursor visible: absolute position)
  type=rel     dx=int dy=int               (cursor hidden: raw delta)
  type=wheel   n=int
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .input.sendinput import SendInput
from .util.win32 import IS_WINDOWS, VK, key_is_down, screen_size, vk_from_name

RECORD_KEYS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(d) for d in range(10)] + \
    ["SPACE", "SHIFT", "CTRL", "ALT", "TAB", "ENTER", "ESC", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
     "UP", "DOWN", "LEFT", "RIGHT", "BACKSPACE"]


def _cursor_visible() -> bool:
    if not IS_WINDOWS:
        return True
    import ctypes
    from ctypes import wintypes

    class CURSORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hCursor", wintypes.HANDLE),
                    ("ptScreenPos", wintypes.POINT)]

    ci = CURSORINFO()
    ci.cbSize = ctypes.sizeof(CURSORINFO)
    if ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci)):
        return bool(ci.flags & 0x00000001)
    return True


def _cursor_pos():
    if not IS_WINDOWS:
        return (0, 0)
    import ctypes
    from ctypes import wintypes
    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)


class MacroRecorder:
    def __init__(self, stop_key: str = "F8", rate_hz: float = 100.0, max_seconds: float = 120.0,
                 log: Callable[[str], None] = print):
        self.stop_vk = vk_from_name(stop_key)
        self.rate = rate_hz
        self.max_seconds = max_seconds
        self.log = log

    def record(self) -> Dict:
        from .input.rawinput import RawInputListener
        raw = RawInputListener().start()
        w, h = screen_size()
        events: List[Dict] = []
        held = {k: False for k in RECORD_KEYS}
        btn = {"L": False, "R": False, "M": False}
        last_abs = None
        t0 = time.perf_counter()
        period = 1.0 / self.rate
        next_t = t0
        while True:
            now = time.perf_counter()
            t = now - t0
            if key_is_down(self.stop_vk) or t > self.max_seconds:
                break
            # keys
            for k in RECORD_KEYS:
                d = key_is_down(VK[k])
                if d != held[k]:
                    held[k] = d
                    events.append({"t": round(t, 4), "type": "key", "key": k, "down": d})
            # buttons
            for b in ("L", "R", "M"):
                d = key_is_down(VK[b + "BUTTON"])
                if d != btn[b]:
                    btn[b] = d
                    events.append({"t": round(t, 4), "type": "button", "btn": b, "down": d})
            dx, dy, wheel = raw.consume()
            if _cursor_visible():
                pos = _cursor_pos()
                if pos != last_abs:
                    last_abs = pos
                    events.append({"t": round(t, 4), "type": "abs", "x": round(pos[0] / max(1, w - 1), 5),
                                   "y": round(pos[1] / max(1, h - 1), 5)})
            elif dx or dy:
                events.append({"t": round(t, 4), "type": "rel", "dx": int(dx), "dy": int(dy)})
            if wheel:
                events.append({"t": round(t, 4), "type": "wheel", "n": int(wheel)})
            next_t += period
            sl = next_t - time.perf_counter()
            if sl > 0:
                time.sleep(sl)
        raw.stop()
        # release anything still held at the end of the recording
        t_end = time.perf_counter() - t0
        for k, d in held.items():
            if d and k != "F8":
                events.append({"t": round(t_end, 4), "type": "key", "key": k, "down": False})
        for b, d in btn.items():
            if d:
                events.append({"t": round(t_end, 4), "type": "button", "btn": b, "down": False})
        return {"version": 1, "duration": round(t_end, 3), "screen": [w, h], "events": events}


class MacroPlayer:
    def __init__(self, sender: Optional[SendInput] = None, abort: Optional[Callable[[], bool]] = None):
        self.sender = sender or SendInput()
        self.abort = abort or (lambda: False)

    def play(self, macro: Dict, speed: float = 1.0) -> bool:
        t0 = time.perf_counter()
        held_keys = set()
        held_btns = set()
        try:
            for ev in macro.get("events", []):
                target = t0 + ev["t"] / max(1e-3, speed)
                while True:
                    if self.abort():
                        return False
                    now = time.perf_counter()
                    if now >= target:
                        break
                    time.sleep(min(0.01, target - now))
                typ = ev["type"]
                if typ == "key":
                    self.sender.key(ev["key"], ev["down"])
                    (held_keys.add if ev["down"] else held_keys.discard)(ev["key"])
                elif typ == "button":
                    self.sender.mouse_button(ev["btn"], ev["down"])
                    (held_btns.add if ev["down"] else held_btns.discard)(ev["btn"])
                elif typ == "abs":
                    self.sender.mouse_move_abs(ev["x"], ev["y"])
                elif typ == "rel":
                    self.sender.mouse_move(ev["dx"], ev["dy"])
                elif typ == "wheel":
                    self.sender.mouse_wheel(ev["n"])
            return True
        finally:
            for k in held_keys:
                self.sender.key(k, False)
            for b in held_btns:
                self.sender.mouse_button(b, False)


def macro_path(macros_dir: Path, name: str) -> Path:
    return Path(macros_dir) / f"{name}.json"


def save_macro(macros_dir: Path, name: str, macro: Dict) -> Path:
    p = macro_path(macros_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(macro, f, indent=1)
    return p


def load_macro(macros_dir: Path, name: str) -> Optional[Dict]:
    p = macro_path(macros_dir, name)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_macro(steps: List[Dict]) -> Dict:
    """Build a macro from a compact description, e.g.
    [{"hold": "G", "seconds": 3.0}, {"wait": 2.0}, {"tap": "TAB"}]"""
    t = 0.0
    events: List[Dict] = []
    for s in steps:
        if "wait" in s:
            t += float(s["wait"])
        elif "hold" in s:
            events.append({"t": round(t, 4), "type": "key", "key": s["hold"].upper(), "down": True})
            t += float(s.get("seconds", 1.0))
            events.append({"t": round(t, 4), "type": "key", "key": s["hold"].upper(), "down": False})
        elif "tap" in s:
            events.append({"t": round(t, 4), "type": "key", "key": s["tap"].upper(), "down": True})
            t += float(s.get("seconds", 0.08))
            events.append({"t": round(t, 4), "type": "key", "key": s["tap"].upper(), "down": False})
        elif "click" in s:
            x, y = s["click"]
            events.append({"t": round(t, 4), "type": "abs", "x": x, "y": y})
            t += 0.05
            events.append({"t": round(t, 4), "type": "button", "btn": s.get("btn", "L"), "down": True})
            t += 0.08
            events.append({"t": round(t, 4), "type": "button", "btn": s.get("btn", "L"), "down": False})
    return {"version": 1, "duration": round(t, 3), "events": events}
