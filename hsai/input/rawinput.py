"""Raw Input listener: captures real mouse deltas/buttons even while the game
has the cursor locked.  Used to record human play for behaviour cloning and to
record macros.  Windows only; a no-op stub elsewhere."""
from __future__ import annotations

import sys
import threading
from typing import Tuple

IS_WINDOWS = sys.platform == "win32"


class RawInputListener:
    def __init__(self):
        self._lock = threading.Lock()
        self._dx = 0
        self._dy = 0
        self._wheel = 0
        self.buttons = {"L": False, "R": False, "M": False}
        self._thread = None
        self._tid = None
        self.ready = threading.Event()
        self.events = 0

    def start(self) -> "RawInputListener":
        if not IS_WINDOWS:
            self.ready.set()
            return self
        self._thread = threading.Thread(target=self._run, name="hsai-rawinput", daemon=True)
        self._thread.start()
        self.ready.wait(2.0)
        return self

    def stop(self) -> None:
        if IS_WINDOWS and self._tid:
            import ctypes
            ctypes.windll.user32.PostThreadMessageW(self._tid, 0x0012, 0, 0)  # WM_QUIT

    def consume(self) -> Tuple[int, int, int]:
        """Return and reset accumulated (dx, dy, wheel) since the last call."""
        with self._lock:
            out = (self._dx, self._dy, self._wheel)
            self._dx = self._dy = self._wheel = 0
            return out

    # -----------------------------------------------------------------
    def _run(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._tid = kernel32.GetCurrentThreadId()

        WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        class WNDCLASSW(ctypes.Structure):
            _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                        ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                        ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                        ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

        class RAWINPUTDEVICE(ctypes.Structure):
            _fields_ = [("usUsagePage", wintypes.USHORT), ("usUsage", wintypes.USHORT),
                        ("dwFlags", wintypes.DWORD), ("hwndTarget", wintypes.HWND)]

        class RAWINPUTHEADER(ctypes.Structure):
            _fields_ = [("dwType", wintypes.DWORD), ("dwSize", wintypes.DWORD),
                        ("hDevice", wintypes.HANDLE), ("wParam", wintypes.WPARAM)]

        class _BTN(ctypes.Structure):
            _fields_ = [("usButtonFlags", wintypes.USHORT), ("usButtonData", wintypes.USHORT)]

        class _BTNU(ctypes.Union):
            _fields_ = [("ulButtons", wintypes.ULONG), ("b", _BTN)]

        class RAWMOUSE(ctypes.Structure):
            _anonymous_ = ("u",)
            _fields_ = [("usFlags", wintypes.USHORT), ("u", _BTNU), ("ulRawButtons", wintypes.ULONG),
                        ("lLastX", wintypes.LONG), ("lLastY", wintypes.LONG), ("ulExtraInformation", wintypes.ULONG)]

        class RAWINPUT(ctypes.Structure):
            _fields_ = [("header", RAWINPUTHEADER), ("mouse", RAWMOUSE), ("_pad", ctypes.c_byte * 16)]

        user32.GetRawInputData.argtypes = [wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p,
                                           ctypes.POINTER(wintypes.UINT), wintypes.UINT]
        user32.DefWindowProcW.restype = ctypes.c_ssize_t
        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                           wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]

        WM_INPUT = 0x00FF
        RID_INPUT = 0x10000003
        RIM_TYPEMOUSE = 0
        RIDEV_INPUTSINK = 0x00000100
        HWND_MESSAGE = wintypes.HWND(-3)
        MOUSE_MOVE_ABSOLUTE = 0x01
        RI_LEFT_DOWN, RI_LEFT_UP = 0x0001, 0x0002
        RI_RIGHT_DOWN, RI_RIGHT_UP = 0x0004, 0x0008
        RI_MID_DOWN, RI_MID_UP = 0x0010, 0x0020
        RI_WHEEL = 0x0400

        raw = RAWINPUT()
        size = wintypes.UINT(ctypes.sizeof(RAWINPUT))
        hdr_size = ctypes.sizeof(RAWINPUTHEADER)

        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                sz = wintypes.UINT(ctypes.sizeof(RAWINPUT))
                got = user32.GetRawInputData(wintypes.HANDLE(lparam), RID_INPUT, ctypes.byref(raw), ctypes.byref(sz), hdr_size)
                if got != 0xFFFFFFFF and raw.header.dwType == RIM_TYPEMOUSE:
                    m = raw.mouse
                    with self._lock:
                        if not (m.usFlags & MOUSE_MOVE_ABSOLUTE):
                            self._dx += m.lLastX
                            self._dy += m.lLastY
                        fl = m.b.usButtonFlags
                        if fl & RI_LEFT_DOWN: self.buttons["L"] = True
                        if fl & RI_LEFT_UP: self.buttons["L"] = False
                        if fl & RI_RIGHT_DOWN: self.buttons["R"] = True
                        if fl & RI_RIGHT_UP: self.buttons["R"] = False
                        if fl & RI_MID_DOWN: self.buttons["M"] = True
                        if fl & RI_MID_UP: self.buttons["M"] = False
                        if fl & RI_WHEEL:
                            d = m.b.usButtonData
                            if d >= 32768:
                                d -= 65536
                            self._wheel += d // 120
                        self.events += 1
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc = WNDPROC(wndproc)
        hinst = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSW()
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = hinst
        wc.lpszClassName = "HSAIRawInputWindow"
        user32.RegisterClassW(ctypes.byref(wc))
        hwnd = user32.CreateWindowExW(0, wc.lpszClassName, "hsai_rawinput", 0, 0, 0, 0, 0, HWND_MESSAGE, None, hinst, None)
        rid = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, hwnd)
        user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(RAWINPUTDEVICE))
        self.ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
