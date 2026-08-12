from backend.core.neutral import symmetrize


def test_symmetrize_cancels_home_bias():
    def raw(home, away):
        # Simulate a model with a strong home-field bias: whoever is "home" wins,
        # regardless of which team that is.
        return (0.9, 0.05, 0.05)

    result = symmetrize(raw, "A", "B")
    assert abs(result["team_a_win"] - result["team_b_win"]) < 1e-9
    assert abs(result["team_a_win"] - 0.475) < 1e-9


def test_symmetrize_preserves_genuine_strength_difference():
    def raw(home, away):
        # A always beats B regardless of home/away assignment.
        return (0.8, 0.1, 0.1) if home == "A" else (0.1, 0.8, 0.1)

    result = symmetrize(raw, "A", "B")
    assert result["team_a_win"] > result["team_b_win"]
