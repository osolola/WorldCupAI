import pandas as pd

from backend.evaluation.backtest import run_backtest

RATINGS_CFG = {
    "base_rating": 1500.0,
    "elo_k_default": 30,
    "attack_defense_k": 8.0,
    "elo_divisor": 600,
    "goals_divisor": 400,
    "rating_diff_clip": 600,
    "expected_goals_clip_min": 0.15,
    "expected_goals_clip_max": 6.0,
}
MODEL_CFG = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1, "random_seed": 42}
DIXON_COLES_CFG = {
    "xi": 0.0005,
    "reg_lambda": 0.01,
    "min_effective_weight": 0.5,
    "max_goals": 8,
    "maxiter": 200,
    "maxfun": 2000,
}


def _synthetic_history():
    rows = []
    teams = ["Strong", "Mid", "Weak"]
    date = pd.Timestamp("2010-01-01")
    for _ in range(300):
        for i, home in enumerate(teams):
            away = teams[(i + 1) % len(teams)]
            strength = {"Strong": 3, "Mid": 1, "Weak": 0}
            rows.append({
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_score": strength[home],
                "away_score": strength[away],
                "tournament": "Friendly",
                "neutral": date.day % 2 == 0,
            })
        date += pd.Timedelta(days=9)
    return pd.DataFrame(rows)


def test_run_backtest_produces_metrics_for_all_models():
    df = _synthetic_history()
    holdout_start = df["date"].quantile(0.7, interpolation="nearest")

    report = run_backtest(
        df,
        RATINGS_CFG,
        MODEL_CFG,
        DIXON_COLES_CFG,
        elo_k_overrides={},
        holdout_start_date=holdout_start,
        calibration_bins=5,
    )

    assert report["periods_evaluated"]
    for name in ["dixon_coles", "xgboost_elo", "elo_only", "always_favorite", "uniform"]:
        assert name in report["models"]
        m = report["models"][name]
        assert m["n_matches"] > 0
        assert 0 <= m["brier_score"] <= 2
        assert m["log_loss"] >= 0
        assert 0 <= m["expected_calibration_error"] <= 1
        assert len(m["calibration_curve"]) == 5

    for name in ["dixon_coles", "xgboost_elo"]:
        misses = report["models"][name]["biggest_misses"]
        assert len(misses) > 0
        assert misses[0]["probability_of_actual"] <= misses[-1]["probability_of_actual"]
        assert {"date", "team_a", "team_b", "outcome"} <= misses[0].keys()


def test_run_backtest_strong_models_beat_uniform_on_predictable_data():
    df = _synthetic_history()
    holdout_start = df["date"].quantile(0.7, interpolation="nearest")

    report = run_backtest(
        df,
        RATINGS_CFG,
        MODEL_CFG,
        DIXON_COLES_CFG,
        elo_k_overrides={},
        holdout_start_date=holdout_start,
        calibration_bins=5,
    )

    # Outcomes here are deterministic given the teams, so any model that's
    # learned team strength should have a strictly lower Brier score than
    # blind 1/3-1/3-1/3 guessing.
    assert report["models"]["dixon_coles"]["brier_score"] < report["models"]["uniform"]["brier_score"]
    assert report["models"]["xgboost_elo"]["brier_score"] < report["models"]["uniform"]["brier_score"]
