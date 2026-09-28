import numpy as np

from hsai.config import load_config
from hsai.env import ActionSpace, EpisodeTracker, Outcome, RewardShaper, StateFeaturizer, is_fight_live
from hsai.telemetry.state import Fighter, GameState


def _cfg():
    return load_config()


def _st(seq, ehp=100.0, php=100.0, edist=3.0, econsc=100.0, pconsc=100.0, pdead=False, edead=False, t=None):
    p = Fighter(health=php, consciousness=pconsc, stamina=90, tonus=100, team=1, pos=(0, 0, 0), dead=pdead,
                bones={"head": (0, 0, 1.7), "hand_r": (0.5, 0.3, 1.2)})
    e = Fighter(health=ehp, consciousness=econsc, stamina=100, tonus=100, team=2, pos=(edist, 0, 0), yaw=180, dead=edead,
                bones={"head": (edist, 0, 1.7)})
    return GameState(seq=seq, game_time=(seq / 60.0 if t is None else t), map_name="Alley", n_fighters=2, player=p, enemies=[e])


def test_action_space_roundtrip():
    cfg = _cfg()
    sp = ActionSpace(cfg.actions, cfg.input.keys, cfg.input.mouse_max_speed_px_s)
    cmd = sp.to_command([0.5, -0.2], [5, 1, 0, 1, 0, 0, 0])
    assert cmd.mouse_vx == 0.5 * cfg.input.mouse_max_speed_px_s
    assert {"W", "A", "ALT"} <= cmd.keys and cmd.buttons == {"L"}
    m, d = sp.from_human({"W", "A", "ALT"}, {"L"}, 1500, -600)
    assert list(d) == [5, 1, 0, 1, 0, 0, 0]
    assert sp.prev_action_vector(m, d).shape[0] == sp.prev_action_dim


def test_featurizer_dim_and_frame():
    cfg = _cfg()
    sp = ActionSpace(cfg.actions, cfg.input.keys, cfg.input.mouse_max_speed_px_s)
    f = StateFeaturizer(cfg.obs.bones, cfg.obs.max_enemies, sp.prev_action_dim)
    v = f(_st(1), None, 0.0, 99, 99)
    assert v.shape == (f.dim,) and np.isfinite(v).all()
    # enemy straight ahead at 3 m: forward feature positive, right ~0
    off = f.player_dim
    assert v[off] == 1.0 and v[off + 5] > 0 and abs(v[off + 6]) < 1e-6
    # bone velocity appears on the second frame
    st2 = _st(2)
    st2.player.bones["hand_r"] = (0.5, 0.6, 1.2)
    v2 = f(st2, np.zeros(sp.prev_action_dim, np.float32), 0.01, 99, 99)
    assert abs(v2[11 + 7 + 5]) > 0  # hand_r right-velocity component


def test_reward_prefers_clean_win():
    cfg = _cfg()
    rs = RewardShaper(cfg.reward)
    rs.reset(_st(0))
    r_hit = rs.step(_st(1, ehp=80.0))
    assert r_hit > 0
    r_hurt = rs.step(_st(2, ehp=80.0, php=70.0))
    assert r_hurt < 0
    clean = RewardShaper(cfg.reward); clean.reset(_st(0))
    dirty = RewardShaper(cfg.reward); dirty.reset(_st(0))
    assert clean.terminal(Outcome.WIN, _st(3, php=100.0)) > dirty.terminal(Outcome.WIN, _st(3, php=20.0)) > 0
    assert RewardShaper(cfg.reward).terminal(Outcome.LOSS, _st(3)) < RewardShaper(cfg.reward).terminal(Outcome.SURRENDER, _st(3)) < 0


def test_approach_is_potential_based():
    cfg = _cfg()
    rs = RewardShaper(cfg.reward)
    rs.reset(_st(0, edist=6.0))
    total = 0.0
    for i in range(1, 60):
        total += rs.step(_st(i, edist=6.0 - i * 0.05))
    back = RewardShaper(cfg.reward)
    back.reset(_st(0, edist=6.0))
    total_back = sum(back.step(_st(i, edist=6.0 + i * 0.05)) for i in range(1, 60))
    assert total > total_back


def test_kite_penalty_after_grace():
    cfg = _cfg()
    rs = RewardShaper(cfg.reward)
    rs.reset(_st(0, edist=8.0))
    for i in range(1, int(cfg.reward.kite_grace_s * 60) + 120):
        rs.step(_st(i, edist=8.0))
    assert rs.stats.kite_steps > 0
    assert rs.stats.kite_steps < int(cfg.reward.kite_grace_s * 60) + 120


def test_episode_outcomes():
    cfg = _cfg()
    tr = EpisodeTracker(cfg.episode)
    tr.reset(_st(0))
    for i in range(1, 70):
        assert tr.update(_st(i)) == Outcome.NONE
    # enemy dead -> win after confirm window
    o = Outcome.NONE
    for i in range(70, 200):
        o = tr.update(_st(i, ehp=0.0, edead=True))
        if o != Outcome.NONE:
            break
    assert o == Outcome.WIN
    tr.reset(_st(0))
    for i in range(1, 70):
        tr.update(_st(i))
    assert tr.update(_st(70, php=0.0, pdead=True)) == Outcome.LOSS
    tr.reset(_st(0))
    for i in range(1, 70):
        tr.update(_st(i))
    o = Outcome.NONE
    for i in range(70, 400):
        o = tr.update(_st(i, pconsc=0.0))
        if o != Outcome.NONE:
            break
    assert o == Outcome.LOSS
    tr.reset(_st(0))
    o = Outcome.NONE
    for i in range(1, int(cfg.episode.max_seconds * 60) + 10):
        o = tr.update(_st(i))
        if o != Outcome.NONE:
            break
    assert o == Outcome.TIMEOUT


def test_fight_live():
    assert is_fight_live(_st(1))
    assert not is_fight_live(_st(1, edead=True, ehp=0))
    assert not is_fight_live(None)
