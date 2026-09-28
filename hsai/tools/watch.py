"""Show what the agent perceives: the stacked observation and the telemetry."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..capture import FramePreprocessor, ScreenCapture
from ..config import Config
from ..env.episode import is_fight_live
from .game_env import capture_region, make_telemetry


def run_watch(cfg: Config, log: Callable[[str], None] = print, seconds: float = 0.0, device: str = "cuda") -> None:
    import torch
    dev = device if torch.cuda.is_available() else "cpu"
    tel = make_telemetry(cfg)
    cap = ScreenCapture(cfg.capture.backend, cfg.capture.monitor_index, cfg.capture.target_fps, capture_region(cfg)).start()
    pre = FramePreprocessor(cfg.obs.size, cfg.obs.stack, cfg.obs.gray, cfg.obs.crop, device=dev)
    try:
        import cv2
    except Exception:
        cv2 = None
    t0 = time.perf_counter()
    last_print = 0.0
    n = 0
    log("watching... press Q in the window (or Ctrl+C) to stop")
    try:
        while True:
            frame = cap.get_frame()
            if frame is None:
                continue
            obs = pre.push(frame)
            n += 1
            st = tel.latest()
            now = time.perf_counter()
            lines = [f"fps {n / max(1e-3, now - t0):.0f}  telemetry age {tel.age_ms():.0f} ms  lines {tel.lines_received}"]
            if st is not None:
                p = st.player
                lines.append(f"map {st.map_name}  fighters {st.n_fighters}  fight_live {is_fight_live(st)}")
                if p:
                    lines.append(f"player hp {p.health:.0f} consc {p.consciousness:.0f} stam {p.stamina:.0f} bones {sorted(p.bones)}")
                for i, e in enumerate(st.enemies[:3]):
                    lines.append(f"enemy{i} hp {e.health:.0f} consc {e.consciousness:.0f} dist {st.distance_to(e):.1f} m team {e.team} dead {e.dead}")
            else:
                lines.append("no telemetry yet (game running with the HSAI mod?)")
            if cv2 is not None:
                c = 1 if cfg.obs.gray else 3
                last = obs[-c:].permute(1, 2, 0).cpu().numpy()
                if c == 1:
                    last = np.repeat(last, 3, axis=2)
                img = cv2.resize(last, (480, 480), interpolation=cv2.INTER_NEAREST)
                canvas = np.zeros((480 + 20 * (len(lines) + 1), 900, 3), dtype=np.uint8)
                canvas[:480, :480] = img
                for i, l in enumerate(lines):
                    cv2.putText(canvas, l, (8, 500 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("HS-AI watch", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            elif now - last_print > 0.5:
                last_print = now
                log(" | ".join(lines))
            if seconds and now - t0 > seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        tel.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()
