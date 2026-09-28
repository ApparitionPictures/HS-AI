from hsai.config import load_config, parse_override


def test_defaults_load():
    cfg = load_config()
    assert cfg.game.target_fps == 60
    assert cfg.ppo.gamma > 0.99
    assert cfg.input.keys.surrender == "G"


def test_override():
    cfg = load_config(overrides={"ppo.lr": 1e-4, "obs.size": 64})
    assert cfg.ppo.lr == 1e-4 and cfg.obs.size == 64
    assert parse_override("game.time_dilation=0.5") == {"game.time_dilation": 0.5}
