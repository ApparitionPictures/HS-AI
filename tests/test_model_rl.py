import time
from pathlib import Path

import numpy as np
import pytest
import torch

from hsai.config import load_config
from hsai.model.noise import ActionSampler, powerlaw_psd_gaussian
from hsai.model.policy import Policy, PolicySpec
from hsai.model.backends import EagerBackend
from hsai.rl.gae import compute_gae
from hsai.rl.generations import GenerationManager
from hsai.rl.ppo import PPOLearner
from hsai.rl.spool import RolloutBuffer, SpoolWriter, list_spool, load_rollout


def _spec():
    return PolicySpec(img_channels=4, img_size=48, state_dim=40, discrete_sizes=[9, 2, 2], gru_hidden=64, conv_out=64,
                      state_hidden=32, conv_channels=[16, 32, 32])


def test_policy_shapes_and_gru_equivalence():
    p = Policy(_spec())
    img = torch.rand(1, 4, 48, 48); st = torch.rand(1, 40); h = p.initial_state(1)
    mean, logstd, logits, v, h2 = p.act_step(img, st, h)
    assert mean.shape == (1, 2) and logits.shape == (1, 13) and v.shape == (1, 1) and h2.shape == (1, 64)
    out = p.forward_seq(torch.rand(5, 3, 4, 48, 48), torch.rand(5, 3, 40), p.initial_state(3), torch.zeros(5, 3))
    assert out[3].shape == (5, 3)
    cell = torch.nn.GRUCell(64, 64)
    p.gru.ih.weight.data.copy_(cell.weight_ih); p.gru.ih.bias.data.copy_(cell.bias_ih)
    p.gru.hh.weight.data.copy_(cell.weight_hh); p.gru.hh.bias.data.copy_(cell.bias_hh)
    x = torch.randn(2, 64); hh = torch.randn(2, 64)
    assert torch.allclose(p.gru(x, hh), cell(x, hh), atol=1e-5)


def test_seq_matches_step():
    """forward_seq must reproduce a chain of act_step calls (with resets)."""
    p = Policy(_spec()).eval()
    T = 6
    img = torch.rand(T, 1, 4, 48, 48); st = torch.rand(T, 1, 40)
    first = torch.zeros(T, 1); first[0, 0] = 1; first[3, 0] = 1
    h = p.initial_state(1)
    means = []
    with torch.no_grad():
        for t in range(T):
            if first[t, 0] == 1:
                h = p.initial_state(1)
            m, _, _, _, h = p.act_step(img[t], st[t], h)
            means.append(m)
        ms, _, _, _ = p.forward_seq(img, st, p.initial_state(1), first)
    assert torch.allclose(torch.cat(means, 0), ms[:, 0], atol=1e-5)


def test_colored_noise_stats():
    rng = np.random.default_rng(0)
    white = powerlaw_psd_gaussian(0.0, 4, 4096, rng)
    pink = powerlaw_psd_gaussian(1.0, 4, 4096, rng)
    assert abs(white.std() - 1) < 0.1 and abs(pink.std() - 1) < 0.15
    ac = lambda y: np.mean([np.corrcoef(y[i, :-1], y[i, 1:])[0, 1] for i in range(4)])
    assert ac(white) < 0.1 and ac(pink) > 0.6


def test_gumbel_marginals_match_softmax():
    logits = np.array([2.0, 0.0, -2.0])
    s = ActionSampler(2, [3], beta=1.0, seed=1)
    cnt = np.zeros(3)
    for _ in range(6000):
        _, d, _ = s.sample(np.zeros(2), np.zeros(2), logits)
        cnt[d[0]] += 1
    p = np.exp(logits); p /= p.sum()
    assert np.abs(cnt / cnt.sum() - p).max() < 0.03


def test_sampler_logprob_matches_torch():
    spec = _spec()
    p = Policy(spec)
    s = ActionSampler(2, spec.discrete_sizes, beta=0.0, seed=0)
    mean = np.array([0.2, -0.1]); logstd = np.array([-0.5, -0.5]); logits = np.random.randn(13)
    mouse, disc, lp = s.sample(mean, logstd, logits)
    lp_t, _, _ = p.log_prob(torch.tensor(mean).float()[None], torch.tensor(logstd).float()[None], torch.tensor(logits).float()[None],
                            torch.tensor(mouse)[None], torch.tensor(disc)[None])
    assert abs(float(lp_t[0]) - lp) < 1e-4


def test_gae():
    r = np.array([1, 0, 0, 1, 0], dtype=np.float32); v = np.full(5, 0.5, np.float32); d = np.array([0, 0, 1, 0, 0], dtype=bool)
    adv, ret = compute_gae(r, v, d, 0.7, 0.9, 1.0)
    assert abs(ret[0] - 1.0) < 1e-5 and abs(ret[3] - (1 + 0.81 * 0.7)) < 1e-5


def test_ppo_update_and_spool(tmp_path):
    cfg = load_config(overrides={"ppo.seq_len": 8, "ppo.minibatches": 2, "ppo.epochs": 1})
    spec = _spec()
    pol = Policy(spec)
    learner = PPOLearner(pol, cfg.ppo, device="cpu")
    w = SpoolWriter(tmp_path)
    buf = RolloutBuffer(64, (4, 48, 48), 40, 3, 64)
    rng = np.random.default_rng(0)
    for t in range(64):
        buf.add(rng.integers(0, 255, (4, 48, 48), dtype=np.uint8), rng.standard_normal(40).astype(np.float32),
                rng.standard_normal(2).astype(np.float32) * 0.3, rng.integers(0, 2, 3), -4.0, rng.standard_normal() * 0.1,
                t % 30 == 29, t % 30 == 0, 0.0, np.zeros(64, np.float16))
    buf.episodes = [{"outcome": "win", "end_health": 80, "damage_dealt": 100, "damage_taken": 20, "reward_sum": 12}]
    w.submit(buf); w.close()
    for _ in range(50):
        if list_spool(tmp_path):
            break
        time.sleep(0.05)
    files = list_spool(tmp_path)
    assert len(files) == 1
    roll = load_rollout(files[0])
    assert roll["img"].shape == (64, 4, 48, 48) and roll["episodes"][0]["outcome"] == "win"
    before = {k: v.clone() for k, v in pol.state_dict().items()}
    stats = learner.update([roll])
    assert stats["steps"] == 64 and np.isfinite(stats["loss_pi"])
    assert any(not torch.equal(before[k], v) for k, v in pol.state_dict().items())


def test_generations_revert(tmp_path):
    cfg = load_config(overrides={"generations.episodes_per_generation": 4, "generations.min_episodes_for_elite": 2,
                                 "generations.patience": 2})
    gm = GenerationManager(cfg.generations, tmp_path)
    good = [{"outcome": "win", "end_health": 90, "damage_dealt": 100, "damage_taken": 10, "reward_sum": 14}] * 4
    bad = [{"outcome": "loss", "end_health": 0, "damage_dealt": 10, "damage_taken": 100, "reward_sum": -9}] * 4
    hyper = {"lr": 3e-4, "entropy_discrete": 0.004, "entropy_continuous": 0.0015, "beta": 1.0, "scale": 1.0}
    gm.add_episodes(good); d1 = gm.close_generation(hyper)
    assert d1["is_elite"] and not d1["revert"]
    gm.add_episodes(bad); d2 = gm.close_generation(hyper)
    assert not d2["revert"]
    gm.add_episodes(bad); d3 = gm.close_generation(hyper)
    assert d3["revert"] and d3["new_hyper"]["lr"] != hyper["lr"]
    gm2 = GenerationManager(cfg.generations, tmp_path); gm2.load()
    assert gm2.elite.index == 1 and len(gm2.history) == 3


def test_eager_backend_matches_policy():
    p = Policy(_spec()).eval()
    b = EagerBackend(p, device="cpu", precision="fp32")
    img = (torch.rand(4, 48, 48) * 255).to(torch.uint8)
    st = np.random.randn(40).astype(np.float32)
    mean, logstd, logits, v = b.step(img, st)
    with torch.no_grad():
        m2, _, lg2, v2, _ = p.act_step(img[None].float() / 255.0, torch.from_numpy(st)[None], p.initial_state(1))
    assert np.allclose(mean, m2[0].numpy(), atol=1e-5) and abs(v - float(v2)) < 1e-5
    assert b.hidden().shape == (64,)
