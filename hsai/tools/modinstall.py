"""Install UE4SS + the HSAI telemetry mod into the game's Win64 folder."""
from __future__ import annotations

import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from ..config import REPO_ROOT
from .gamefind import find_game_dir, find_win64_dir

UE4SS_VERSION = "v3.0.1"
UE4SS_STABLE_URL = f"https://github.com/UE4SS-RE/RE-UE4SS/releases/download/{UE4SS_VERSION}/UE4SS_{UE4SS_VERSION}.zip"
UE4SS_EXPERIMENTAL_API = "https://api.github.com/repos/UE4SS-RE/RE-UE4SS/releases/tags/experimental-latest"
MOD_SRC = REPO_ROOT / "mod" / "HSAI"


def ue4ss_root(win64: Path) -> Path:
    """UE4SS >= 3.0.1-experimental keeps everything in Win64/ue4ss/, the stable
    3.0.1 zip is flat in Win64/.  Return whichever holds UE4SS.dll (or the
    experimental layout when neither exists yet)."""
    if (win64 / "ue4ss" / "UE4SS.dll").exists():
        return win64 / "ue4ss"
    if (win64 / "UE4SS.dll").exists():
        return win64
    return win64 / "ue4ss"


def local_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "HSAI"


def ue4ss_installed(win64: Path) -> bool:
    return (win64 / "dwmapi.dll").exists() and (ue4ss_root(win64) / "UE4SS.dll").exists()


def mod_installed(win64: Path) -> bool:
    return (ue4ss_root(win64) / "Mods" / "HSAI" / "scripts" / "main.lua").exists()


def mod_enabled(win64: Path) -> bool:
    mods_txt = ue4ss_root(win64) / "Mods" / "mods.txt"
    if (ue4ss_root(win64) / "Mods" / "HSAI" / "enabled.txt").exists():
        return True
    if mods_txt.exists():
        for line in mods_txt.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().replace(" ", "").lower().startswith("hsai:1"):
                return True
    return False


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "hsai-installer", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def resolve_ue4ss_url(log: Callable[[str], None], prefer_stable: bool = False) -> str:
    """The Half Sword modding community runs the UE4SS *experimental* builds
    (UE 5.4+ support, ue4ss/ sub-folder layout).  Ask GitHub for the newest
    experimental asset and fall back to the stable release."""
    if not prefer_stable:
        try:
            import json
            d = json.loads(_http_get(UE4SS_EXPERIMENTAL_API, timeout=30).decode("utf-8"))
            assets = [a for a in d.get("assets", []) if a["name"].lower().startswith("ue4ss_") and a["name"].lower().endswith(".zip")]
            if assets:
                a = sorted(assets, key=lambda x: x.get("updated_at", ""))[-1]
                log(f"UE4SS experimental build: {a['name']}")
                return a["browser_download_url"]
        except Exception as e:
            log(f"could not query experimental UE4SS builds ({e}); using stable {UE4SS_VERSION}")
    return UE4SS_STABLE_URL


def download_ue4ss(log: Callable[[str], None], prefer_stable: bool = False) -> bytes:
    url = resolve_ue4ss_url(log, prefer_stable)
    log(f"downloading {url} ...")
    data = _http_get(url)
    log(f"downloaded {len(data) / 1e6:.1f} MB")
    return data


def install_ue4ss(win64: Path, log: Callable[[str], None], zip_path: Optional[Path] = None, hide_console: bool = True,
                  prefer_stable: bool = False) -> None:
    if ue4ss_installed(win64):
        log(f"UE4SS already installed ({ue4ss_root(win64)}), keeping it")
    else:
        data = Path(zip_path).read_bytes() if zip_path else download_ue4ss(log, prefer_stable)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            if not any(n.lower().endswith("dwmapi.dll") for n in names):
                raise RuntimeError("unexpected UE4SS zip layout (no dwmapi.dll)")
            z.extractall(win64)
        log(f"UE4SS extracted into {win64} (layout root: {ue4ss_root(win64)})")
    ini = ue4ss_root(win64) / "UE4SS-settings.ini"
    if ini.exists() and hide_console:
        txt = ini.read_text(encoding="utf-8", errors="replace")
        for key in ("ConsoleEnabled", "GuiConsoleEnabled", "GuiConsoleVisible"):
            txt = _set_ini(txt, key, "0")
        ini.write_text(txt, encoding="utf-8")
        log("UE4SS console windows disabled (edit ue4ss/UE4SS-settings.ini to re-enable)")


def _set_ini(txt: str, key: str, value: str) -> str:
    out = []
    done = False
    for line in txt.splitlines():
        if line.strip().lower().startswith(key.lower() + " ") or line.strip().lower().startswith(key.lower() + "="):
            out.append(f"{key} = {value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key} = {value}")
    return "\n".join(out) + "\n"


def install_mod(win64: Path, log: Callable[[str], None]) -> None:
    mods = ue4ss_root(win64) / "Mods"
    mods.mkdir(parents=True, exist_ok=True)
    dst = mods / "HSAI"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(MOD_SRC, dst)
    (dst / "enabled.txt").write_text("", encoding="utf-8")
    mods_txt = mods / "mods.txt"
    lines = mods_txt.read_text(encoding="utf-8", errors="replace").splitlines() if mods_txt.exists() else []
    lines = [l for l in lines if not l.strip().replace(" ", "").lower().startswith("hsai:")]
    # first entry so it loads early; the built-in 'Keybinds' entry must stay last
    lines.insert(0, "HSAI : 1")
    mods_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    local_dir().mkdir(parents=True, exist_ok=True)
    log(f"HSAI mod installed to {dst}")


def uninstall_mod(win64: Path, log: Callable[[str], None], remove_ue4ss: bool = False) -> None:
    root = ue4ss_root(win64)
    mods = root / "Mods"
    dst = mods / "HSAI"
    if dst.exists():
        shutil.rmtree(dst)
        log("HSAI mod removed")
    mods_txt = mods / "mods.txt"
    if mods_txt.exists():
        lines = [l for l in mods_txt.read_text(encoding="utf-8", errors="replace").splitlines()
                 if not l.strip().replace(" ", "").lower().startswith("hsai:")]
        mods_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if remove_ue4ss:
        targets = [win64 / "dwmapi.dll", win64 / "ue4ss"] if root != win64 else \
            [win64 / "dwmapi.dll", win64 / "UE4SS.dll", win64 / "UE4SS-settings.ini", win64 / "Mods"]
        for p in targets:
            if p.exists():
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        log("UE4SS removed")


def install_all(log: Callable[[str], None], install_dir: str = "", app_id: int = 2397300,
                zip_path: Optional[Path] = None, hide_console: bool = True, prefer_stable: bool = False) -> Path:
    game = find_game_dir(app_id, install_dir)
    if game is None:
        raise FileNotFoundError("Half Sword installation not found. In Steam: right-click Half Sword > Manage > Browse local files, "
                                "then run: hsai mod install --dir \"<that folder>\"")
    win64 = find_win64_dir(game)
    if win64 is None:
        raise FileNotFoundError(f"could not find the Binaries\\Win64 folder under {game}")
    log(f"game folder: {game}")
    install_ue4ss(win64, log, zip_path=zip_path, hide_console=hide_console, prefer_stable=prefer_stable)
    install_mod(win64, log)
    return win64
