from backend.competitions.registry import get_competition


def test_world_cup_registered_with_knockout_format():
    comp = get_competition("world_cup")
    assert comp.format == "knockout"
    assert comp.elo_k_overrides.get("FIFA World Cup") == 60
