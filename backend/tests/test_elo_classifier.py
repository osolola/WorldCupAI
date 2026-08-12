import pandas as pd

from backend.core.elo_classifier import build_features, make_feature_row


def test_build_features_computes_diffs():
    df = pd.DataFrame([{
        "home_elo": 1600, "away_elo": 1500,
        "home_attack": 1550, "home_defense": 1520,
        "away_attack": 1480, "away_defense": 1510,
        "home_score": 2, "away_score": 1,
    }])

    out = build_features(df)

    assert out.loc[0, "elo_diff"] == 100
    assert out.loc[0, "attack_diff"] == 1550 - 1510
    assert out.loc[0, "defense_diff"] == 1480 - 1520


def test_make_feature_row_uses_base_rating_for_unknown_teams():
    row = make_feature_row({}, "Unknown A", "Unknown B")
    assert row == [[1500.0, 1500.0, 0.0, 1500.0, 1500.0, 1500.0, 1500.0, 0.0, 0.0]]
