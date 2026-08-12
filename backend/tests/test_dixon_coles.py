import numpy as np
import pandas as pd

from backend.core.dixon_coles import DixonColesModel, fit_dixon_coles


def test_match_probabilities_sum_to_one_and_are_nonnegative():
    model = DixonColesModel(mu=0.3, home_adv=0.2, rho=-0.05, attack={"A": 0.4, "B": -0.2}, defense={"A": 0.1, "B": -0.1})
    probs = model.match_probabilities("A", "B", neutral=True)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_two_unknown_teams_on_neutral_ground_are_symmetric():
    model = DixonColesModel(mu=0.3, home_adv=0.25, rho=-0.05, attack={}, defense={})
    probs = model.match_probabilities("Unknown A", "Unknown B", neutral=True)
    assert abs(probs["team_a_win"] - probs["team_b_win"]) < 1e-9


def test_home_advantage_only_applies_when_not_neutral():
    model = DixonColesModel(mu=0.3, home_adv=0.25, rho=-0.05, attack={}, defense={})
    neutral_probs = model.match_probabilities("A", "B", neutral=True)
    home_probs = model.match_probabilities("A", "B", neutral=False)
    assert home_probs["team_a_win"] > neutral_probs["team_a_win"]


def test_scoreline_matrix_sums_to_one():
    model = DixonColesModel(mu=0.3, home_adv=0.2, rho=-0.05, attack={"A": 0.5}, defense={"B": -0.3})
    grid = model.scoreline_matrix("A", "B", neutral=True)
    assert abs(grid.sum() - 1.0) < 1e-9


def test_sample_score_is_reproducible_with_seeded_rng():
    model = DixonColesModel(mu=0.3, home_adv=0.2, rho=-0.05, attack={"A": 0.5}, defense={"B": -0.3})
    draws_1 = [model.sample_score("A", "B", True, np.random.default_rng(42)) for _ in range(20)]
    draws_2 = [model.sample_score("A", "B", True, np.random.default_rng(42)) for _ in range(20)]
    assert draws_1 == draws_2


def _synthetic_league(n_rounds=6):
    """Strong beats Weak consistently; Weak beats Weaker consistently."""
    rows = []
    date = pd.Timestamp("2023-01-01")
    for r in range(n_rounds):
        rows.append({"date": date, "home_team": "Strong", "away_team": "Weak", "home_score": 3, "away_score": 0, "tournament": "Friendly", "neutral": True})
        rows.append({"date": date, "home_team": "Weak", "away_team": "Weaker", "home_score": 2, "away_score": 0, "tournament": "Friendly", "neutral": True})
        rows.append({"date": date, "home_team": "Strong", "away_team": "Weaker", "home_score": 4, "away_score": 0, "tournament": "Friendly", "neutral": True})
        date += pd.Timedelta(days=14)
    return pd.DataFrame(rows)


def test_fit_dixon_coles_recovers_sensible_team_ordering():
    df = _synthetic_league()
    model, result = fit_dixon_coles(df, xi=0.0001, reg_lambda=0.01, min_effective_weight=0.5)

    assert result.success or result.status in (0, 1, 2)
    assert model.attack["Strong"] > model.attack["Weak"] > model.attack["Weaker"]

    strong_vs_weaker = model.match_probabilities("Strong", "Weaker", neutral=True)
    assert strong_vs_weaker["team_a_win"] > strong_vs_weaker["team_b_win"]


def test_fit_dixon_coles_excludes_sparse_teams_below_threshold():
    df = _synthetic_league()
    # Add a single one-off match for a team that should not accumulate enough weight.
    df = pd.concat([df, pd.DataFrame([{
        "date": pd.Timestamp("2023-01-01"), "home_team": "OneOff", "away_team": "Strong",
        "home_score": 0, "away_score": 1, "tournament": "Friendly", "neutral": True,
    }])], ignore_index=True)

    model, _ = fit_dixon_coles(df, xi=0.0001, min_effective_weight=5.0)

    assert "OneOff" not in model.attack
    assert model.attack.get("OneOff", 0.0) == 0.0
