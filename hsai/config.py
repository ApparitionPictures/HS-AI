"""Typed configuration loaded from YAML.

Every tunable lives in ``configs/default.yaml``.  A user config (``configs/user.yaml``)
is layered on top of it so updates to defaults never clobber personal settings.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
USER_CONFIG = REPO_ROOT / "configs" / "user.yaml"


@dataclass
class GameCfg:
    window_title_contains: str = "Half Sword"
    process_name: str = "HalfSwordUE5-Win64-Shipping.exe"
    steam_app_id: int = 2397300
    install_dir: str = ""           # auto-detected from Steam when empty
    time_dilation: float = 1.0      # 1.0 = real time. <1 slows the game (curriculum), >1 speeds it up
    target_fps: int = 60            # decision rate of the agent (must be <= game fps)


@dataclass
class CaptureCfg:
    backend: str = "auto"           # auto | bettercam | dxcam | mss
    monitor_index: int = 0
    region: Optional[List[float]] = None   # normalized [left, top, right, bottom] of the monitor, None = full
    target_fps: int = 60
    gpu_upload: bool = True


@dataclass
class ObsCfg:
    size: int = 96                  # square resize of the (cropped) frame fed to the CNN
    stack: int = 4                  # number of consecutive frames stacked
    gray: bool = True
    crop: List[float] = field(default_factory=lambda: [0.10, 0.05, 0.90, 0.95])  # normalized crop of the frame
    max_enemies: int = 2            # nearest enemies encoded in the state vector
    bones: List[str] = field(default_factory=lambda: ["head", "hand_r", "hand_l", "pelvis"])
    include_prev_action: bool = True


@dataclass
class TelemetryCfg:
    transport: str = "pipe"         # pipe | file
    pipe_name: str = r"\\.\pipe\hsai_telemetry"
    file_path: str = "%LOCALAPPDATA%/HSAI/telemetry.txt"
    cmd_path: str = "%LOCALAPPDATA%/HSAI/cmd.txt"
    stale_ms: int = 250             # telemetry older than this = game not running / mod not loaded


@dataclass
class KeysCfg:
    forward: str = "W"
    back: str = "S"
    left: str = "A"
    right: str = "D"
    sprint: str = "SHIFT"
    crouch: str = "CTRL"
    kick: str = "SPACE"
    thrust: str = "ALT"
    lock_on: str = "TAB"
    interact: str = "E"
    swap: str = "X"
    surrender: str = "G"


@dataclass
class InputCfg:
    keys: KeysCfg = field(default_factory=KeysCfg)
    mouse_max_speed_px_s: float = 3000.0   # |action|=1 -> this many pixels per second
    mouse_subticks: int = 4                # mouse deltas emitted per agent tick (240 Hz at 60 Hz ticks)
    mouse_smoothing: float = 0.35          # EMA on the commanded velocity, 0 = none, 0.9 = very smooth
    kill_switch_key: str = "F12"           # release everything and stop the actor
    pause_key: str = "F11"                 # toggle pause (keys released while paused)
    require_foreground: bool = True        # only send input while the game window is in front


@dataclass
class ActionsCfg:
    move: bool = True
    lmb: bool = True
    rmb: bool = True
    thrust: bool = True
    sprint: bool = True
    kick: bool = True
    crouch: bool = True
    surrender: bool = False   # learned give-up action (off by default; env auto-surrenders when hopeless)


@dataclass
class RewardCfg:
    # dense
    damage_dealt: float = 0.04         # per point of enemy health lost (100 hp -> +4)
    ko_dealt: float = 0.02             # per point of enemy consciousness lost
    damage_taken: float = 0.05         # per point of own health lost (negative)
    ko_taken: float = 0.02             # per point of own consciousness lost (negative)
    time_penalty: float = 0.0005       # per step (60 Hz): -1.8 per minute
    low_stamina_penalty: float = 0.0005  # per step while stamina < low_stamina_frac
    low_stamina_frac: float = 0.15
    # engagement shaping (potential-based, policy-invariant)
    approach_potential: float = 0.02   # reward per metre of approach toward engage distance
    engage_distance_m: float = 2.0
    kite_penalty: float = 0.002        # per step while disengaged too long (see below)
    kite_grace_s: float = 4.0          # seconds of no damage exchange + far away before kite_penalty applies
    kite_far_m: float = 3.5
    # terminal
    win: float = 10.0
    win_health_bonus: float = 6.0      # extra * (own health fraction at the end): flawless win = win + bonus
    lose: float = -10.0
    timeout: float = -3.0
    surrender: float = -6.0
    clip_step: float = 3.0             # per-step reward clip (safety)


@dataclass
class EpisodeCfg:
    max_seconds: float = 120.0
    start_grace_s: float = 1.0         # ignore terminals right after a reset
    ko_grace_s: float = 3.0            # consciousness <= 0 for this long counts as loss/win
    enemy_dead_confirm_s: float = 1.0
    reset_mode: str = "macro"          # macro | mod
    macro_after_fight: str = "after_fight"
    lock_on_at_start: bool = True
    post_terminal_wait_s: float = 2.5
    fight_start_timeout_s: float = 90.0
    auto_surrender_after_s: float = 20.0   # player down (health<=hopeless) for this long -> hold G
    hopeless_health: float = 5.0
    mod_reset_spawn_distance_m: float = 3.0
    mod_reset_enemy_class: str = ""    # e.g. /Game/Assets/Characters/Willie_BP.Willie_BP_C ('' = keep game's enemies)


@dataclass
class ModelCfg:
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    conv_out: int = 256
    state_hidden: int = 128
    gru_hidden: int = 512
    logstd_init: float = -0.5
    logstd_min: float = -3.0
    logstd_max: float = 0.5
    backend: str = "auto"              # auto | tensorrt | cuda_graph | eager
    precision: str = "fp16"            # inference precision for backends: fp16 | bf16 | fp32
    trt_workspace_mb: int = 1024


@dataclass
class PPOCfg:
    gamma: float = 0.997
    lam: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    value_coef: float = 0.5
    entropy_discrete: float = 0.004
    entropy_continuous: float = 0.0015
    lr: float = 3.0e-4
    lr_min: float = 5.0e-5
    lr_decay_updates: int = 4000
    epochs: int = 3
    minibatches: int = 4
    seq_len: int = 32                  # truncated BPTT length
    rollout_steps: int = 1024          # steps per spool file (17 s at 60 Hz)
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    adv_norm: bool = True
    value_norm: bool = True
    reward_norm: bool = True
    weight_decay: float = 0.0
    bc_kl_coef: float = 0.0            # >0 keeps the policy close to a behaviour-cloned prior early on
    bc_kl_decay_updates: int = 500
    amp: str = "bf16"                  # bf16 | fp16 | off


@dataclass
class ExploreCfg:
    noise: str = "pink"                # pink | white
    beta: float = 1.0                  # colored-noise exponent (0 = white, 1 = pink, 2 = brown)
    scale: float = 1.0                 # multiplier on policy std at rollout time
    eval_every_episodes: int = 10      # every N episodes run one deterministic (no-noise) episode


@dataclass
class GenerationsCfg:
    episodes_per_generation: int = 20
    min_episodes_for_elite: int = 10
    patience: int = 2                  # consecutive worse generations before reverting to elite
    margin: float = 0.05               # fitness must drop by more than this to count as worse
    keep_top_k: int = 5
    lr_factors: List[float] = field(default_factory=lambda: [0.7, 1.4])
    entropy_factors: List[float] = field(default_factory=lambda: [0.7, 1.4])
    beta_delta: float = 0.25
    fitness_win: float = 2.0
    fitness_clean: float = 1.0
    fitness_dealt: float = 1.0
    fitness_taken: float = 1.0


@dataclass
class LearnerCfg:
    device: str = "cuda"
    minibatch_sleep_ms: int = 0        # sleep between minibatches to leave GPU time for the game
    weight_sync_every_updates: int = 1
    compile: bool = False
    channels_last: bool = True
    cudnn_benchmark: bool = True
    max_spool_backlog: int = 6


@dataclass
class PathsCfg:
    runs_dir: str = "runs"
    spool_dir: str = "spool"
    checkpoints_dir: str = "checkpoints"
    macros_dir: str = "macros"
    recordings_dir: str = "recordings"


@dataclass
class RecordCfg:
    chunk_steps: int = 1800            # 30 s chunks at 60 Hz
    keep_full_frames: bool = False


@dataclass
class Config:
    game: GameCfg = field(default_factory=GameCfg)
    capture: CaptureCfg = field(default_factory=CaptureCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    telemetry: TelemetryCfg = field(default_factory=TelemetryCfg)
    input: InputCfg = field(default_factory=InputCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    episode: EpisodeCfg = field(default_factory=EpisodeCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    explore: ExploreCfg = field(default_factory=ExploreCfg)
    generations: GenerationsCfg = field(default_factory=GenerationsCfg)
    learner: LearnerCfg = field(default_factory=LearnerCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    record: RecordCfg = field(default_factory=RecordCfg)
    seed: int = 1

    # ---- derived helpers -------------------------------------------------
    def resolve_path(self, p: str) -> Path:
        p = os.path.expandvars(os.path.expanduser(p))
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path


# ---------------------------------------------------------------------------
# Loading / merging
# ---------------------------------------------------------------------------

def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _from_dict(cls, data: Dict[str, Any], path: str = "") -> Any:
    if not dataclasses.is_dataclass(cls):
        return data
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in (data or {}).items():
        if key not in fields:
            print(f"[config] warning: unknown key '{path}{key}' ignored")
            continue
        f = fields[key]
        ftype = f.type
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default  # type: ignore[misc]
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _from_dict(type(default), value, f"{path}{key}.")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def to_dict(cfg: Any) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def load_config(user_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load default.yaml, layer user.yaml (or ``user_path``) and dotted overrides on top."""
    data: Dict[str, Any] = {}
    if DEFAULT_CONFIG.exists():
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    upath = Path(user_path) if user_path else USER_CONFIG
    if upath.exists():
        with open(upath, "r", encoding="utf-8") as f:
            data = _merge(data, yaml.safe_load(f) or {})
    for dotted, value in (overrides or {}).items():
        node = data
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return _from_dict(Config, data)


def save_user_config(values: Dict[str, Any]) -> None:
    """Merge ``values`` (nested dict) into configs/user.yaml."""
    current: Dict[str, Any] = {}
    if USER_CONFIG.exists():
        with open(USER_CONFIG, "r", encoding="utf-8") as f:
            current = yaml.safe_load(f) or {}
    merged = _merge(current, values)
    USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_CONFIG, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)


def parse_override(s: str) -> Dict[str, Any]:
    """'ppo.lr=1e-4' -> {'ppo.lr': 0.0001} with YAML typing."""
    if "=" not in s:
        raise ValueError(f"override must look like section.key=value, got {s!r}")
    k, v = s.split("=", 1)
    return {k.strip(): yaml.safe_load(v)}
