"""Thin ctypes wrappers around the handful of Win32 calls we need.

All functions degrade gracefully on non-Windows platforms so the training
core and unit tests can run on Linux.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    HIGH_PRIORITY_CLASS = 0x00000080
    ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
    THREAD_PRIORITY_HIGHEST = 2
    THREAD_PRIORITY_TIME_CRITICAL = 15


def window_title(hwnd) -> str:
    if not IS_WINDOWS:
        return ""
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def window_process_path(hwnd) -> str:
    if not IS_WINDOWS:
        return ""
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not h:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value
        return ""
    finally:
        kernel32.CloseHandle(h)


def enum_windows() -> List[Tuple[int, str]]:
    """[(hwnd, title)] for visible top-level windows."""
    if not IS_WINDOWS:
        return []
    out: List[Tuple[int, str]] = []

    @WNDENUMPROC
    def cb(hwnd, lparam):
        if user32.IsWindowVisible(hwnd):
            t = window_title(hwnd)
            if t:
                out.append((int(hwnd), t))
        return True

    user32.EnumWindows(cb, 0)
    return out


def find_game_window(title_contains: str, process_name: str = "") -> Optional[int]:
    """Return the hwnd of the game window (by title substring, else by process exe name)."""
    if not IS_WINDOWS:
        return None
    tl = title_contains.lower().replace(" ", "")
    for hwnd, title in enum_windows():
        if tl and tl in title.lower().replace(" ", ""):
            return hwnd
    if process_name:
        pn = process_name.lower()
        for hwnd, _title in enum_windows():
            if os.path.basename(window_process_path(hwnd)).lower() == pn:
                return hwnd
    return None


def foreground_window() -> int:
    if not IS_WINDOWS:
        return 0
    return int(user32.GetForegroundWindow() or 0)


def window_rect(hwnd) -> Optional[Tuple[int, int, int, int]]:
    if not IS_WINDOWS:
        return None
    r = wintypes.RECT()
    if user32.GetWindowRect(hwnd, ctypes.byref(r)):
        return (r.left, r.top, r.right, r.bottom)
    return None


def key_is_down(vk: int) -> bool:
    if not IS_WINDOWS:
        return False
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def set_process_priority(high: bool = True) -> None:
    if not IS_WINDOWS:
        return
    try:
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), HIGH_PRIORITY_CLASS if high else ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def set_thread_priority(critical: bool = False) -> None:
    if not IS_WINDOWS:
        return
    try:
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL if critical else THREAD_PRIORITY_HIGHEST)
    except Exception:
        pass


def screen_size() -> Tuple[int, int]:
    if not IS_WINDOWS:
        return (1920, 1080)
    try:
        # per-monitor DPI awareness so we see physical pixels
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            user32.SetProcessDPIAware()
        return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
    except Exception:
        return (1920, 1080)


def make_dpi_aware() -> None:
    if not IS_WINDOWS:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


# ---- virtual key codes ----------------------------------------------------
VK = {
    "LBUTTON": 0x01, "RBUTTON": 0x02, "MBUTTON": 0x04,
    "BACKSPACE": 0x08, "TAB": 0x09, "ENTER": 0x0D, "SHIFT": 0x10, "CTRL": 0x11, "ALT": 0x12,
    "PAUSE": 0x13, "CAPSLOCK": 0x14, "ESC": 0x1B, "ESCAPE": 0x1B, "SPACE": 0x20,
    "PAGEUP": 0x21, "PAGEDOWN": 0x22, "END": 0x23, "HOME": 0x24,
    "LEFT": 0x25, "UP": 0x26, "RIGHT": 0x27, "DOWN": 0x28,
    "INSERT": 0x2D, "DELETE": 0x2E,
    "LSHIFT": 0xA0, "RSHIFT": 0xA1, "LCTRL": 0xA2, "RCTRL": 0xA3, "LALT": 0xA4, "RALT": 0xA5,
    "OEM_3": 0xC0, "GRAVE": 0xC0, "TILDE": 0xC0,
}
for i in range(10):
    VK[str(i)] = 0x30 + i
for i in range(26):
    VK[chr(ord("A") + i)] = 0x41 + i
for i in range(1, 25):
    VK[f"F{i}"] = 0x70 + i - 1

EXTENDED_VKS = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0xA3, 0xA5, 0x5B, 0x5C, 0x6F, 0x0D}


def vk_from_name(name: str) -> int:
    key = name.strip().upper()
    if key in VK:
        return VK[key]
    if len(key) == 1:
        return ord(key)
    raise KeyError(f"unknown key name {name!r}")
