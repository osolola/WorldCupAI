from backend.evaluation.baselines import (
    fit_always_favorite_baseline,
    fit_elo_only_baseline,
    uniform_baseline,
)


def test_uniform_baseline_is_one_third_each():
    result = uniform_baseline([("A", "B"), ("C", "D")])
    assert len(result) == 2
    for probs in result:
        assert abs(probs["team_a_win"] - 1 / 3) < 1e-9
        assert abs(probs["team_b_win"] - 1 / 3) < 1e-9
        assert abs(probs["draw"] - 1 / 3) < 1e-9


def test_always_favorite_baseline_recovers_empirical_rates():
    outcomes = ["favorite_win"] * 6 + ["underdog_win"] * 2 + ["draw"] * 2
    predict = fit_always_favorite_baseline(outcomes)

    favored = predict(team_a_is_favorite=True)
    assert abs(favored["team_a_win"] - 0.6) < 1e-9
    assert abs(favored["team_b_win"] - 0.2) < 1e-9
    assert abs(favored["draw"] - 0.2) < 1e-9

    underdog = predict(team_a_is_favorite=False)
    assert abs(underdog["team_a_win"] - 0.2) < 1e-9
    assert abs(underdog["team_b_win"] - 0.6) < 1e-9


def test_elo_only_baseline_favors_higher_elo_team():
    # Clearly separable synthetic data: large positive elo_diff -> home win,
    # large negative -> away win, near-zero -> draw.
    elo_diffs = [400] * 20 + [-400] * 20 + [0] * 20
    targets = [0] * 20 + [1] * 20 + [2] * 20

    raw_proba = fit_elo_only_baseline(elo_diffs, targets)

    strong_home = raw_proba(home_elo=1900, away_elo=1500)
    strong_away = raw_proba(home_elo=1500, away_elo=1900)

    assert strong_home[0] > strong_home[1]
    assert strong_away[1] > strong_away[0]
