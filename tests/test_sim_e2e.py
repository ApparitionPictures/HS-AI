"""End-to-end: actor + learner process + spool + generations on the toy simulator (CPU)."""
import os
import sys
from pathlib import Path

import pytest

from hsai.config import load_config


@pytest.mark.timeout(600)
def test_train_on_sim(tmp_path):
    from hsai.rl.train import train
    from hsai.tools.sim import make_sim_env
    cfg = load_config(overrides={
        "obs.size": 48, "model.conv_channels": [16, 32, 32], "model.conv_out": 64, "model.state_hidden": 32,
        "model.gru_hidden": 64, "ppo.rollout_steps": 128, "ppo.seq_len": 16, "ppo.minibatches": 2, "ppo.epochs": 1,
        "episode.max_seconds": 3.0, "generations.episodes_per_generation": 2, "generations.min_episodes_for_elite": 1,
        "paths.runs_dir": str(tmp_path / "runs"), "paths.spool_dir": str(tmp_path / "spool"),
        "paths.checkpoints_dir": str(tmp_path / "ckpt"), "learner.device": "cpu", "explore.eval_every_episodes": 0})
    run = train(cfg, lambda c, s, f, l: make_sim_env(c, s, f, l), log=lambda s: None, max_episodes=6, dashboard=False, device="cpu")
    assert (run / "episodes.csv").exists()
    assert (run / "weights" / "latest.pt").exists()
    lines = (run / "episodes.csv").read_text().strip().splitlines()
    assert len(lines) == 7  # header + 6 episodes
    # the learner must have consumed rollouts and published at least one update
    assert (run / "updates.csv").exists() and len((run / "updates.csv").read_text().strip().splitlines()) >= 2
    assert (run / "generations.json").exists()
