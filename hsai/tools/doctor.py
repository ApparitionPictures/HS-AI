"""System check: GPU, capture, game, mod, telemetry, inference speed."""
from __future__ import annotations

import platform
import time
from typing import Callable, List, Tuple

from ..config import Config
from .gamefind import find_game_dir, find_shipping_exe, find_win64_dir
from .modinstall import local_dir, mod_enabled, mod_installed, ue4ss_installed, ue4ss_root

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self, log: Callable[[str], None]):
        self.rows: List[Tuple[str, str, str]] = []
        self.log = log

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        self.log(f"[{status:4s}] {name}: {detail}")

    @property
    def failures(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == FAIL)


def _static_checks(cfg: Config, rep: Report) -> None:
    rep.add(OK, "python", f"{platform.python_version()} on {platform.system()} {platform.release()}")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            rep.add(OK, "gpu", f"{name} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}, CUDA {torch.version.cuda}, "
                               f"bf16={torch.cuda.is_bf16_supported()}")
        else:
            rep.add(FAIL, "gpu", f"torch {torch.__version__} has no CUDA. Re-run scripts\\install.bat (it installs the CUDA build).")
    except Exception as e:
        rep.add(FAIL, "torch", f"not importable: {e}")
    try:
        import tensorrt as trt
        rep.add(OK, "tensorrt", f"{trt.__version__} available (fastest backend)")
    except Exception:
        rep.add(WARN, "tensorrt", "not installed; CUDA-graph backend will be used (~1 ms). Optional: scripts\\install.bat --trt")
    got_capture = False
    for name in ("bettercam", "dxcam", "mss"):
        try:
            __import__(name)
            rep.add(OK, "capture lib", f"{name} importable")
            got_capture = True
            break
        except Exception:
            pass
    if not got_capture:
        rep.add(FAIL, "capture lib", "neither bettercam, dxcam nor mss importable -> pip install bettercam")
    try:
        import cv2  # noqa: F401
        rep.add(OK, "opencv", "importable (needed for `hsai watch` window)")
    except Exception:
        rep.add(WARN, "opencv", "not importable; `hsai watch` prints text instead of a window")
    # ---- game -------------------------------------------------------------
    game = find_game_dir(cfg.game.steam_app_id, cfg.game.install_dir)
    if game is None:
        rep.add(FAIL, "game install", "Half Sword not found via Steam. Use: hsai mod install --dir \"<game folder>\"")
        win64 = None
    else:
        win64 = find_win64_dir(game)
        exe = find_shipping_exe(win64) if win64 else None
        rep.add(OK if exe else FAIL, "game install", f"{game}  exe={exe.name if exe else 'not found'}")
    if win64 is not None:
        u = ue4ss_installed(win64)
        rep.add(OK if u else FAIL, "ue4ss", f"installed at {ue4ss_root(win64)}" if u else "missing -> run: hsai mod install")
        m = mod_installed(win64)
        rep.add(OK if m else FAIL, "hsai mod", "installed" if m else "missing -> run: hsai mod install")
        if m:
            rep.add(OK if mod_enabled(win64) else FAIL, "hsai mod enabled", "yes" if mod_enabled(win64) else "no (mods.txt)")
    status = local_dir() / "mod_status.txt"
    if status.exists():
        age_h = (time.time() - status.stat().st_mtime) / 3600.0
        rep.add(OK, "mod loaded by game", f"last load {age_h:.1f} h ago ({status})")
    else:
        rep.add(WARN, "mod loaded by game", "never (start the game once after installing the mod)")
    from ..util.win32 import IS_WINDOWS, find_game_window, foreground_window, window_title
    if IS_WINDOWS:
        hwnd = find_game_window(cfg.game.window_title_contains, cfg.game.process_name)
        if hwnd:
            rep.add(OK, "game window", f"'{window_title(hwnd)}' {'(in front)' if foreground_window() == hwnd else '(not in front)'}")
        else:
            rep.add(WARN, "game window", "not running")
    else:
        rep.add(WARN, "platform", "not Windows: game I/O unavailable, only offline training/tests work here")


def _live_checks(cfg: Config, rep: Report) -> None:
    from ..util.win32 import IS_WINDOWS
    try:
        from .game_env import make_telemetry
        tel = make_telemetry(cfg)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0 and tel.lines_received == 0:
            time.sleep(0.05)
        st = tel.latest()
        if st is None:
            rep.add(FAIL if IS_WINDOWS else WARN, "telemetry", "no data in 3 s (game not running, not in a level, or mod not loaded)")
        else:
            rate = tel.lines_received / max(1e-3, time.perf_counter() - t0)
            p = st.player
            e = st.enemies[0] if st.enemies else None
            bones = sorted(p.bones.keys()) if p else []
            rep.add(OK, "telemetry", f"{rate:.0f} lines/s, map='{st.map_name}', fighters={st.n_fighters}, "
                                    f"player={'hp %.0f' % p.health if p else 'none'}, enemies={len(st.enemies)}"
                                    + (f", nearest hp {e.health:.0f} at {st.distance_to(e):.1f} m" if e else ""))
            rep.add(OK if bones else WARN, "bones", f"{bones or 'none found (check obs.bones names in configs and mod/HSAI/config.txt)'}")
        tel.stop()
    except Exception as ex:
        rep.add(FAIL, "telemetry", repr(ex))
    if IS_WINDOWS:
        try:
            from ..capture import ScreenCapture
            from .game_env import capture_region
            cap = ScreenCapture(cfg.capture.backend, cfg.capture.monitor_index, cfg.capture.target_fps, capture_region(cfg)).start()
            for _ in range(15):  # warm-up (first frames include device init)
                cap.get_frame()
            n = 120
            dark = 0
            t0 = time.perf_counter()
            for _ in range(n):
                f = cap.get_frame()
                if f is not None and f.mean() < 2:
                    dark += 1
            fps = n / (time.perf_counter() - t0)
            cap.stop()
            hint = ""
            if fps < cfg.game.target_fps * 0.9:
                hint = " -> below target: set the Windows display to 2560x1440 (or 1920x1080) while training, and close overlays"
            rep.add(OK if fps >= cfg.game.target_fps * 0.9 else WARN, "capture",
                    f"{cap.backend} region {cap.width}x{cap.height} at {fps:.0f} fps" + (" (frames are black!)" if dark > n // 2 else "") + hint)
        except Exception as ex:
            rep.add(FAIL, "capture", repr(ex))


def _bench(cfg: Config, rep: Report) -> None:
    try:
        import numpy as np
        import torch
        from ..model.backends import make_backend
        from ..model.policy import build_policy
        from ..rl.actor import build_dims
        space, feat, img_ch = build_dims(cfg)
        pol = build_policy(img_ch, cfg.obs.size, feat.dim, space.discrete_sizes, cfg.model)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for name in (["tensorrt", "cuda_graph", "eager"] if dev == "cuda" else ["eager"]):
            try:
                b = make_backend(pol, name, dev, cfg.model.precision, cfg.model.trt_workspace_mb,
                                 cache_dir=str(cfg.resolve_path(cfg.paths.checkpoints_dir) / "trt_cache"), log=lambda s: None)
                if b.name != name:
                    continue
                img = torch.zeros((img_ch, cfg.obs.size, cfg.obs.size), dtype=torch.uint8, device=dev)
                stv = np.zeros(feat.dim, np.float32)
                for _ in range(20):
                    b.step(img, stv)
                t0 = time.perf_counter()
                for _ in range(200):
                    b.step(img, stv)
                ms = (time.perf_counter() - t0) / 200 * 1000
                rep.add(OK if ms < 8 else WARN, f"inference {name}", f"{ms:.2f} ms/step ({pol.count_params() / 1e6:.1f} M params)")
            except Exception as ex:
                rep.add(WARN, f"inference {name}", f"unavailable: {type(ex).__name__}: {str(ex)[:120]}")
    except Exception as ex:
        rep.add(FAIL, "inference", repr(ex))


def run_doctor(cfg: Config, log: Callable[[str], None] = print, quick: bool = False, bench: bool = True) -> Report:
    rep = Report(log)
    _static_checks(cfg, rep)
    if not quick:
        _live_checks(cfg, rep)
    if bench:
        _bench(cfg, rep)
    macro = cfg.resolve_path(cfg.paths.macros_dir) / f"{cfg.episode.macro_after_fight}.json"
    rep.add(OK if macro.exists() else WARN, "after_fight macro",
            str(macro) if macro.exists() else f"not recorded yet -> hsai macro record {cfg.episode.macro_after_fight}")
    log("")
    log(f"summary: {rep.failures} problem(s)" if rep.failures else "summary: all critical checks passed")
    return rep
