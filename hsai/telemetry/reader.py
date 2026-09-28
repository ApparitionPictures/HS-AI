"""Receives telemetry lines from the Lua mod.

Transport 'pipe': this process hosts a Windows named pipe server; the mod
connects as a client and streams lines.  Transport 'file': the mod rewrites a
small file every frame and we poll it.  Both feed the same ``latest()`` API.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, List, Optional

from .state import GameState, parse_line

IS_WINDOWS = sys.platform == "win32"


class TelemetryReader:
    def __init__(self, transport: str = "pipe", pipe_name: str = r"\\.\pipe\hsai_telemetry",
                 file_path: str = "", bone_names: Optional[List[str]] = None, stale_ms: int = 250):
        self.transport = transport
        self.pipe_name = pipe_name
        self.file_path = os.path.expandvars(file_path) if file_path else ""
        self.bone_names = bone_names
        self.stale_ms = stale_ms
        self._lock = threading.Lock()
        self._latest: Optional[GameState] = None
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.lines_received = 0
        self.connected = False
        self.on_state: Optional[Callable[[GameState], None]] = None
        self._new_event = threading.Event()

    # ---- public API -----------------------------------------------------
    def start(self) -> "TelemetryReader":
        if self.transport == "pipe" and IS_WINDOWS:
            t = threading.Thread(target=self._pipe_loop, name="hsai-telemetry-pipe", daemon=True)
            t.start(); self._threads.append(t)
        if self.file_path and (self.transport == "file" or not IS_WINDOWS or self.transport == "pipe"):
            # file poller also runs in pipe mode as a fallback (the mod writes the file only when the pipe is down)
            t = threading.Thread(target=self._file_loop, name="hsai-telemetry-file", daemon=True)
            t.start(); self._threads.append(t)
        return self

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> Optional[GameState]:
        with self._lock:
            return self._latest

    def wait_new(self, timeout: float) -> Optional[GameState]:
        """Block until a state newer than the last returned one arrives (or timeout)."""
        if self._new_event.wait(timeout):
            self._new_event.clear()
        return self.latest()

    def age_ms(self) -> float:
        st = self.latest()
        if st is None:
            return 1e9
        return (time.perf_counter() - st.recv_time) * 1000.0

    def is_fresh(self) -> bool:
        return self.age_ms() <= self.stale_ms

    def push_line(self, line: str) -> None:
        st = parse_line(line, self.bone_names)
        if st is None:
            return
        with self._lock:
            if self._latest is not None and st.seq < self._latest.seq and (self._latest.seq - st.seq) < 1000:
                return  # out-of-order duplicate
            self._latest = st
        self.lines_received += 1
        self._new_event.set()
        if self.on_state:
            try:
                self.on_state(st)
            except Exception:
                pass

    # ---- named pipe server ---------------------------------------------
    def _pipe_loop(self) -> None:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        PIPE_ACCESS_INBOUND = 0x00000001
        PIPE_TYPE_BYTE = 0x00000000
        PIPE_READMODE_BYTE = 0x00000000
        PIPE_WAIT = 0x00000000
        PIPE_UNLIMITED_INSTANCES = 255
        INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
        ERROR_PIPE_CONNECTED = 535
        k32.CreateNamedPipeW.restype = wintypes.HANDLE
        k32.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                         wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
        k32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        k32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]

        buf = ctypes.create_string_buffer(65536)
        nread = wintypes.DWORD(0)
        while not self._stop.is_set():
            h = k32.CreateNamedPipeW(self.pipe_name, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                                     PIPE_UNLIMITED_INSTANCES, 65536, 65536, 0, None)
            if h == INVALID_HANDLE_VALUE or h is None:
                time.sleep(1.0)
                continue
            ok = k32.ConnectNamedPipe(h, None)
            if not ok and k32.GetLastError() != ERROR_PIPE_CONNECTED:
                k32.CloseHandle(h)
                time.sleep(0.2)
                continue
            self.connected = True
            pending = b""
            while not self._stop.is_set():
                if not k32.ReadFile(h, buf, 65536, ctypes.byref(nread), None) or nread.value == 0:
                    break
                pending += buf.raw[: nread.value]
                *lines, pending = pending.split(b"\n")
                if lines:
                    # only the newest complete line matters
                    try:
                        self.push_line(lines[-1].decode("utf-8", "replace"))
                    except Exception:
                        pass
                if len(pending) > 1 << 20:
                    pending = b""
            self.connected = False
            k32.DisconnectNamedPipe(h)
            k32.CloseHandle(h)

    # ---- file poller ----------------------------------------------------
    def _file_loop(self) -> None:
        last_mtime = -1.0
        while not self._stop.is_set():
            try:
                m = os.path.getmtime(self.file_path)
                if m != last_mtime:
                    last_mtime = m
                    with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                        line = f.read()
                    if self.transport == "file" or not self.connected:
                        self.push_line(line)
            except FileNotFoundError:
                time.sleep(0.25)
                continue
            except Exception:
                pass
            time.sleep(0.002)
