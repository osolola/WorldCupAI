from backend.simulation.monte_carlo import simulate_group_stage, simulate_knockout_bracket


def _lopsided_predict_fn(strength):
    """team with higher strength[...] wins ~98% of the time, no draws."""
    def predict(team_a, team_b, neutral):
        if strength[team_a] > strength[team_b]:
            return {"team_a_win": 0.98, "team_b_win": 0.01, "draw": 0.01}
        return {"team_a_win": 0.01, "team_b_win": 0.98, "draw": 0.01}
    return predict


def test_dominant_team_wins_the_bracket_almost_always():
    teams = ["Best", "Weak1", "Weak2", "Weak3"]
    strength = {"Best": 100, "Weak1": 1, "Weak2": 1, "Weak3": 1}
    predict_fn = _lopsided_predict_fn(strength)

    result = simulate_knockout_bracket(teams, predict_fn, n_sims=500, seed=1)

    assert result["teams"]["Best"]["champion"]["probability"] > 0.9
    lo, hi = result["teams"]["Best"]["champion"]["ci_low"], result["teams"]["Best"]["champion"]["ci_high"]
    assert lo <= result["teams"]["Best"]["champion"]["probability"] <= hi


def test_champion_probabilities_sum_to_one():
    teams = ["A", "B", "C", "D"]
    strength = {"A": 3, "B": 2, "C": 1, "D": 1}
    predict_fn = _lopsided_predict_fn(strength)

    result = simulate_knockout_bracket(teams, predict_fn, n_sims=300, seed=2)

    total = sum(t["champion"]["probability"] for t in result["teams"].values())
    assert abs(total - 1.0) < 1e-9


def test_round_reached_totals_are_conserved():
    # In a 4-team bracket exactly 2 teams reach the final every simulation,
    # so probabilities of "reached round_of_2" must sum to exactly 2.0.
    teams = ["A", "B", "C", "D"]
    strength = {"A": 3, "B": 2, "C": 1, "D": 1}
    predict_fn = _lopsided_predict_fn(strength)

    result = simulate_knockout_bracket(teams, predict_fn, n_sims=300, seed=3)

    total_reached_final = sum(t["rounds_reached"]["round_of_2"]["probability"] for t in result["teams"].values())
    assert abs(total_reached_final - 2.0) < 1e-9


def _deterministic_score_sampler(strength):
    def sampler(team_a, team_b, neutral, rng):
        return strength[team_a], strength[team_b]
    return sampler


def test_group_stage_top_two_by_strength_always_advance():
    groups = {"A": ["Best", "Good", "Bad", "Worst"]}
    strength = {"Best": 5, "Good": 3, "Bad": 1, "Worst": 0}
    sampler = _deterministic_score_sampler(strength)

    result = simulate_group_stage(groups, sampler, advance_per_group=2, n_sims=50, seed=4)

    assert result["teams"]["Best"]["advance_probability"] == 1.0
    assert result["teams"]["Good"]["advance_probability"] == 1.0
    assert result["teams"]["Bad"]["advance_probability"] == 0.0
    assert result["teams"]["Worst"]["advance_probability"] == 0.0


def test_group_stage_probabilities_are_valid():
    groups = {"A": ["W", "X", "Y", "Z"]}
    strength = {"W": 2, "X": 2, "Y": 1, "Z": 1}

    def noisy_sampler(team_a, team_b, neutral, rng):
        return int(rng.poisson(strength[team_a])), int(rng.poisson(strength[team_b]))

    result = simulate_group_stage(groups, noisy_sampler, advance_per_group=2, n_sims=200, seed=5)

    for team, stats in result["teams"].items():
        assert 0.0 <= stats["advance_probability"] <= 1.0
        assert stats["ci_low"] <= stats["advance_probability"] <= stats["ci_high"]
