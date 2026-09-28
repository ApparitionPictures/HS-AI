"""Win32 SendInput wrapper: scancode keyboard events + relative/absolute mouse.

Unreal reads raw input; injected relative mouse moves and scancode key events
generated through SendInput are delivered to the game exactly like a real
device.  On non-Windows platforms every call is a no-op (for tests).
"""
from __future__ import annotations

import sys
from typing import Optional

from ..util.win32 import EXTENDED_VKS, vk_from_name

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    ULONG_PTR = ctypes.c_size_t

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    user32 = ctypes.windll.user32
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]

    INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x0001, 0x0002, 0x0008
    MOUSEEVENTF_MOVE, MOUSEEVENTF_ABSOLUTE = 0x0001, 0x8000
    MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
    MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
    MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP = 0x0020, 0x0040
    MOUSEEVENTF_WHEEL = 0x0800
    MAPVK_VK_TO_VSC = 0
    HSAI_TAG = 0x48534149  # 'HSAI' marker in dwExtraInfo so our own injected input can be recognised


class SendInput:
    """Stateless low-level sender.  Use InputController for stateful, safe control."""

    def __init__(self):
        self.enabled = IS_WINDOWS
        self.sent = 0

    # --- keyboard --------------------------------------------------------
    def key(self, name_or_vk, down: bool) -> None:
        if not self.enabled:
            return
        vk = vk_from_name(name_or_vk) if isinstance(name_or_vk, str) else int(name_or_vk)
        scan = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
        flags = KEYEVENTF_SCANCODE
        if vk in EXTENDED_VKS:
            flags |= KEYEVENTF_EXTENDEDKEY
        if not down:
            flags |= KEYEVENTF_KEYUP
        inp = INPUT(type=INPUT_KEYBOARD)
        inp.ki = KEYBDINPUT(0, scan, flags, 0, HSAI_TAG)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        self.sent += 1

    # --- mouse ------------------------------------------------------------
    def mouse_move(self, dx: int, dy: int) -> None:
        if not self.enabled or (dx == 0 and dy == 0):
            return
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, HSAI_TAG)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        self.sent += 1

    def mouse_move_abs(self, nx: float, ny: float) -> None:
        """Move to a normalized [0,1] screen position (for menus)."""
        if not self.enabled:
            return
        x = int(max(0.0, min(1.0, nx)) * 65535)
        y = int(max(0.0, min(1.0, ny)) * 65535)
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(x, y, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, HSAI_TAG)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        self.sent += 1

    def mouse_button(self, button: str, down: bool) -> None:
        if not self.enabled:
            return
        b = button.upper()[0]
        flag = {"L": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP), "R": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
                "M": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP)}[b][0 if down else 1]
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(0, 0, 0, flag, 0, HSAI_TAG)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        self.sent += 1

    def mouse_wheel(self, clicks: int) -> None:
        if not self.enabled or clicks == 0:
            return
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(0, 0, ctypes.c_uint32(int(clicks * 120) & 0xFFFFFFFF).value, MOUSEEVENTF_WHEEL, 0, HSAI_TAG)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        self.sent += 1
