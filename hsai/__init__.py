"""HS-AI: a real-time reinforcement-learning agent for Half Sword.

Design summary
--------------
* Perception at the game's frame rate (60 Hz target) from DXGI desktop duplication
  plus exact game state exported by a tiny UE4SS Lua mod (health, consciousness,
  stamina, positions, bone locations).
* A small CNN + state-MLP + GRU policy with a hybrid action space: continuous mouse
  velocity (the arms) and discrete keys (WASD, LMB, RMB, Alt, Shift, Space, Ctrl).
* Asynchronous PPO: the actor process talks to the game at 60 Hz; the learner
  process trains on the same GPU (bf16 tensor cores) and publishes new weights.
* Generation / elite loop: every N episodes the policy is scored, the best is
  kept as the "elite", regressions revert to it with perturbed hyper-parameters.
* Inference backends: CUDA-graph captured PyTorch (default) or TensorRT with
  in-place weight refit (optional, lowest latency).
"""

__version__ = "0.1.0"
