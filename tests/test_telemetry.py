import math

from hsai.telemetry.state import Fighter, GameState, encode_line, parse_line


def _state():
    p = Fighter(health=88.5, consciousness=100, stamina=70, tonus=100, team=1, pos=(1.0, 2.0, 0.5), yaw=90,
                vel=(0.1, 0, 0), bones={"head": (1, 2, 2.2), "hand_r": (1.2, 2.1, 1.5)})
    e = Fighter(health=40, consciousness=60, stamina=50, tonus=90, team=2, pos=(3.0, 2.5, 0.5), yaw=-90,
                bones={"head": (3, 2.5, 2.2)})
    return GameState(seq=5, game_time=12.34, map_name="Alley", cam_yaw=45, cam_pitch=-5, n_fighters=2, player=p, enemies=[e])


def test_roundtrip():
    st = _state()
    back = parse_line(encode_line(st))
    assert back is not None
    assert back.player.health == 88.5 and back.enemies[0].team == 2
    assert math.isclose(back.player.bones["head"][2], 2.2, abs_tol=1e-6)
    assert "hand_l" not in back.player.bones
    assert math.isclose(back.distance_to(back.enemies[0]), math.sqrt(4 + 0.25), rel_tol=1e-3)


def test_partial_line_rejected():
    line = encode_line(_state())
    assert parse_line(line[:-5]) is None
    assert parse_line("garbage") is None
    assert parse_line(line.replace("|E1|", "|E2|")) is None


def test_no_player():
    st = _state()
    st.player = None
    back = parse_line(encode_line(st))
    assert back.player is None and len(back.enemies) == 1
