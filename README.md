# HS-AI — a real-time AI that learns to fight in Half Sword

HS-AI watches the screen and the game's own combat state 60 times per second,
decides a mouse motion and key presses every frame, and learns from the result
with reinforcement learning. It runs on your PC next to the game and uses your
NVIDIA GPU (RTX 4090 recommended) for both the fast reaction loop and training.

Everything is driven from a few double-click scripts and `hsai` commands.
You never need to write code.

---

## What it does, in one picture

```
 Half Sword (Steam)                       HS-AI (this program)
 ┌───────────────────────┐   60 fps      ┌───────────────────────────────────┐
 │ game renders a frame  │──screen──────▶│ crop/resize on GPU  ─┐            │
 │                       │               │                      ├▶ policy net│──▶ mouse + keys
 │ tiny UE4SS mod reads  │──telemetry───▶│ health, positions,   │  (~1 ms)   │    (SendInput,
 │ Health/Consciousness/ │ (named pipe)  │ hand speeds, ...    ─┘            │     240 Hz smooth
 │ Stamina/positions     │               │                                   │     mouse)
 └───────────────────────┘               │ rollouts ──▶ PPO learner (2nd process, bf16 tensor cores)
                                         │ generations: keep the best ("elite"), revert + perturb when worse
                                         └───────────────────────────────────┘
```

* **Perception at the game's frame rate.** Screen capture uses the Windows
  DXGI desktop-duplication API (240 Hz capable). The game state (exact health,
  consciousness, stamina, positions, head/hand bone positions of you and the
  enemies) comes from a small Lua mod running inside the game through UE4SS,
  so nothing is guessed from pixels.
* **Reaction every frame.** The policy network runs in about 1 ms with CUDA
  graphs, or ~0.5 ms with TensorRT (optional). Human reaction time is ~200 ms.
* **Physics-aware control.** The mouse is controlled as a *velocity* that a
  240 Hz thread turns into smooth deltas, so a swing is a continuous motion,
  not 60 separate jerks. Exploration noise is "pink" (temporally correlated),
  so the agent tries whole swings instead of jitter. Hand positions and hand
  velocities of both fighters are part of what the network sees.
* **Rewards that match what you asked for.** Highest reward: win without
  taking damage. Damage dealt is rewarded, damage taken is punished, a small
  time penalty pushes for decisive fights, approaching to engagement range is
  encouraged, and running away is only penalised after several seconds with
  no exchange while far away, so a short retreat to reset is free.
* **Generations / elite loop.** Every 20 fights the policy gets a fitness score
  (win rate, cleanliness of wins, damage dealt vs. taken). The best generation
  is kept as the *elite*. If results get worse for two generations in a row,
  the weights revert to the elite and the learning settings are perturbed, so
  learning continues from the strongest point instead of drifting.

---

## Requirements

* Windows 10/11, Half Sword (Steam), NVIDIA GPU with a recent driver.
* Python 3.11 or 3.12 (the installer installs it with `winget` if missing).
* Internet for the one-time install (PyTorch, UE4SS).

---

## Install (once)

1. Download or clone this repository, e.g. to `C:\HS-AI`.
2. Double-click **`scripts\install.bat`**. It creates a private Python
   environment, installs PyTorch with CUDA, installs UE4SS and the HS-AI mod
   into the game folder, and runs a system check. Takes 5-15 minutes.
   * For the fastest inference backend run `scripts\install.bat --trt`
     (installs TensorRT; optional).
3. Game settings (Half Sword → Options):
   * Display: **borderless window**, 1920×1080 or 2560×1440 (not 4K: cheaper
     capture and more GPU headroom for the game to hold 60+ fps).
   * Mouse sensitivity: low (the AI does not care, but low sensitivity gives
     it finer control). Keep the default key bindings, or mirror your changes in
     `configs/user.yaml` under `input.keys`.
   * Uncapped or 120 fps limit, so the game never drops under 60.

> The mod only *reads* game state unless HS-AI explicitly sends it a command.
> `hsai mod uninstall` removes it again.

## First run: 3 steps

**Step 1 – check everything.** Start Half Sword, get into a fight (any mode,
Free Mode is the easiest), tab back to a terminal and run
`scripts\doctor.bat`. Every line should be PASS or WARN. It also prints what
the mod sees (map name, your health, the enemy's distance, which bones were
found) and how fast inference runs. `scripts\watch.bat` shows a live view of
what the AI perceives.

**Step 2 – teach it how to get to the next fight.** The AI cannot know how
your menus work, so you show it once. Run `scripts\record_macro.bat`, switch
to the game, and, when the countdown ends, do exactly what you normally do
after a fight to start the next one (hold G, click through the Inn, pick your
loadout, ...). Press **F8** when the new fight has started. Test it with
`hsai macro play after_fight`. If your flow changes (e.g. randomised gear),
re-record it. This is also where *you* randomise your gear if you want
different loadouts every fight: record the clicks that do it.

**Step 3 – train.** Double-click **`scripts\train.bat`** (default 2 hours;
`scripts\train.bat 4` for four hours). Keep the game window in front. A
dashboard shows fps, inference time, win rate, learner losses and the elite.
Keyboard: **F11** pauses the AI, **F12** stops it immediately and releases
every key. Training resumes later with `hsai train --resume <run folder>`.

Watch the best policy with `scripts\play.bat` (uses the latest `elite.pt`).

### Strongly recommended: show it how to fight first

Learning from a single real-time game is slow: the AI gets ~200k decisions per
hour and starts from random flailing. Twenty minutes of *your* play give it a
sensible prior (how a swing works, that you face the enemy, when to block):

```
scripts\record_play.bat      # fight normally for 15-30 min, F12 to stop
scripts\pretrain.bat         # imitation learning -> checkpoints\bc.pt
hsai train --hours 2 --init checkpoints\bc.pt
```

Set `ppo.bc_kl_coef: 0.05` in `configs/user.yaml` to keep the early RL close to
your style while it explores.

---

## Commands (`hsai.bat` at the repo root)

| command | what it does |
|---|---|
| `hsai setup` | install mod + system check (install.bat runs this) |
| `hsai doctor` | full check incl. telemetry, capture fps and inference speed |
| `hsai watch` | live window with the observation and telemetry |
| `hsai macro record after_fight` | record menu navigation; `play`/`list` too |
| `hsai record` | record your own fights (for imitation) |
| `hsai pretrain` | behaviour-clone from the recordings |
| `hsai train --hours 2` | reinforcement learning; `--init`, `--resume`, `--sim` |
| `hsai play` | run the best policy without learning |
| `hsai bench` | inference backend benchmark |
| `hsai config set key=value` | change a setting (written to configs/user.yaml) |
| `hsai mod install/uninstall/status` | manage the UE4SS telemetry mod |

Any setting can also be overridden for one run: `hsai -o game.time_dilation=0.6 train`.

## Useful settings (configs/user.yaml)

* `game.time_dilation` – 0.6 slows the game to 60 % during training (an
  easier curriculum; the AI still decides at 60 Hz). Raise back to 1.0 later.
* `episode.reset_mode` – `macro` (default, replays your recording) or `mod`
  (asks the mod to heal you and respawn an enemy in place: no menus at all,
  fastest training; needs `episode.mod_reset_enemy_class` set to an enemy
  class path, see `hsai mod status` output and the modding notes below).
* `reward.*` – all the weights described above.
* `actions.surrender: true` – lets the AI learn to hold G itself (off by
  default; the environment already auto-surrenders when it lies helpless).
* `model.backend` – `auto` (TensorRT → CUDA graph → eager), `cuda_graph`, `tensorrt`.
* `obs.size`, `obs.stack`, `obs.crop` – what the CNN sees.

## Files produced

```
runs/<date>/weights/latest.pt   newest weights (what the actor runs)
runs/<date>/elite.pt            best generation so far  <- use this to play
runs/<date>/checkpoints/        top-k generation checkpoints
runs/<date>/episodes.csv        one row per fight
runs/<date>/updates.csv         one row per PPO update
runs/<date>/generations.json    fitness history, elite, reverts
recordings/                     your recorded fights
macros/                         recorded menu macros
```

## Controls the AI uses (defaults, match your in-game bindings)

Mouse = arms. LMB = right hand grip/swing, RMB = left hand / half-sword,
Alt = thrust, WASD = move, Shift = sprint/dodge, Space = kick, Ctrl = crouch,
Tab = lock-on (pressed once when a fight starts), G = surrender/advance
(used by the reset flow).

## Honest expectations

* One game instance = one environment. It cannot be sped up or copied 100
  times; "100 sessions" here means 100 *fights* over time, scored per
  generation, with the best kept. Expect visible improvement (approaching,
  facing, swinging at the right moment) within a couple of hours, and
  competent fighting after many hours across several sessions, much sooner
  with imitation pretraining.
* The game must render at ≥ 60 fps for 60 Hz perception. Lower the game's
  resolution before anything else if it does not.
* The telemetry mod was written against the Early-Access game's known
  character class (`Willie_BP_C`) and property names (`Health`,
  `Consciousness`, `Stamina`, `All Body Tonus`, `Team Int`, `DED`). If a game
  update renames them, `hsai doctor` will show no player/enemy data; the names
  are in `mod/HSAI/config.txt` and can be updated without touching code.

## Troubleshooting

* **doctor says `ue4ss: missing` / `hsai mod: missing`.** The mod install step
  failed. Run `scripts\install_mod.bat` and read its message. If it says
  permission denied, run `scripts\install_mod_admin.bat`. If your antivirus
  quarantined `UE4SS.dll`/`dwmapi.dll` (a known false positive for UE4SS),
  add the Half Sword folder to its exclusions and run the installer again. No
  internet: download `UE4SS_v3.0.1.zip` from the UE4SS GitHub releases and run
  `hsai mod install --zip "<path>"`.
* **`telemetry: no data`.** The game must be running *inside a level* (a fight
  or the Inn) with the mod loaded. `hsai mod status` shows whether the mod
  ever loaded. If it never did, UE4SS did not start: check that `dwmapi.dll`
  sits next to `HalfSwordUE5-Win64-Shipping.exe`.
* **`capture` below 60 fps.** You are probably on a 4K desktop. Set the Windows
  display resolution to 2560x1440 (or 1920x1080) while training; the capture
  cost scales with desktop pixels, not with the game's render resolution.
* **The game freezes or crashes at start after installing the mod.** Open
  `ue4ss/UE4SS-settings.ini` and set `bUseUObjectArrayCache = false`, or
  uninstall with `hsai mod uninstall --all`.

## Safety

* Inputs are only sent while the Half Sword window is in the foreground.
* **F12** releases every key and stops; **F11** pauses.
* Nothing is written to the game folder except UE4SS and the `HSAI` mod folder.

## Developers

`pip install -r requirements-dev.txt && pytest` runs 22 CPU tests including an
end-to-end training run on a toy simulator (`hsai train --sim`). Layout:

```
hsai/telemetry   Lua-mod protocol, named-pipe reader, command writer
hsai/capture     DXGI capture, GPU preprocessing
hsai/input       SendInput, 240 Hz controller, RawInput, macros
hsai/env         action space, state features, reward, terminal logic, env loop
hsai/model       policy (CNN+MLP+GRU), colored noise, CUDA-graph / TensorRT backends
hsai/rl          spool, GAE, PPO, generations, BC, actor, learner, orchestrator
hsai/tools       doctor, watch, record, game finder, mod installer, simulator
mod/HSAI         the UE4SS Lua telemetry mod
```
