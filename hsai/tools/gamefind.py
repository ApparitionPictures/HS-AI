"""Locate the Half Sword installation through Steam (Windows)."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

IS_WINDOWS = sys.platform == "win32"


def steam_root() -> Optional[Path]:
    if not IS_WINDOWS:
        return None
    try:
        import winreg
        for hive, key in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
                          (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")):
            try:
                with winreg.OpenKey(hive, key) as k:
                    for name in ("SteamPath", "InstallPath"):
                        try:
                            v, _ = winreg.QueryValueEx(k, name)
                            p = Path(v)
                            if p.exists():
                                return p
                        except OSError:
                            pass
            except OSError:
                pass
    except ImportError:
        pass
    for guess in (r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam", r"D:\Steam", r"D:\SteamLibrary"):
        if Path(guess).exists():
            return Path(guess)
    return None


def steam_libraries() -> List[Path]:
    root = steam_root()
    libs: List[Path] = []
    if root is None:
        return libs
    libs.append(root)
    vdf = root / "steamapps" / "libraryfolders.vdf"
    if vdf.exists():
        txt = vdf.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r'"path"\s+"([^"]+)"', txt):
            p = Path(m.group(1).replace("\\\\", "\\"))
            if p.exists() and p not in libs:
                libs.append(p)
    return libs


def find_game_dir(app_id: int = 2397300, override: str = "") -> Optional[Path]:
    if override:
        p = Path(os.path.expandvars(override))
        return p if p.exists() else None
    for lib in steam_libraries():
        acf = lib / "steamapps" / f"appmanifest_{app_id}.acf"
        if acf.exists():
            m = re.search(r'"installdir"\s+"([^"]+)"', acf.read_text(encoding="utf-8", errors="replace"))
            if m:
                p = lib / "steamapps" / "common" / m.group(1)
                if p.exists():
                    return p
        cand = lib / "steamapps" / "common" / "Half Sword"
        if cand.exists():
            return cand
    return None


def find_win64_dir(game_dir: Path) -> Optional[Path]:
    direct = game_dir / "HalfSwordUE5" / "Binaries" / "Win64"
    if direct.exists():
        return direct
    for p in game_dir.glob("*/Binaries/Win64"):
        if p.is_dir():
            return p
    return None


def find_shipping_exe(win64: Path) -> Optional[Path]:
    for p in win64.glob("*Shipping.exe"):
        return p
    for p in win64.glob("*.exe"):
        return p
    return None
