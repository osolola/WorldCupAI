from backend.competitions.knockout import run_bracket


def _fake_predict(a, b):
    # Deterministic stand-in: alphabetically-first team always wins.
    if a < b:
        return {"team_a_win": 0.9, "team_b_win": 0.05, "draw": 0.05}
    return {"team_a_win": 0.05, "team_b_win": 0.9, "draw": 0.05}


def test_run_bracket_reduces_to_single_champion():
    teams = ["B", "A", "D", "C"]

    rounds, champion = run_bracket(teams, _fake_predict)

    assert champion == "A"
    assert len(rounds) == 2
    assert len(rounds[0]) == 2
    assert len(rounds[1]) == 1
