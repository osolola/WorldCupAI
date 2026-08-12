import json

import pandas as pd
import pytest

from backend.ingestion import REQUIRED_COLUMNS, ingest, load_new_matches


def _write_existing_csv(path):
    pd.DataFrame([
        {"date": "2024-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0,
         "tournament": "Friendly", "city": "X", "country": "X", "neutral": False},
        {"date": "2024-02-01", "home_team": "B", "away_team": "A", "home_score": 2, "away_score": 2,
         "tournament": "Friendly", "city": "X", "country": "X", "neutral": False},
    ]).to_csv(path, index=False)


def test_load_new_matches_from_json(tmp_path):
    json_path = tmp_path / "new.json"
    json_path.write_text(json.dumps([
        {"date": "2024-03-01", "home_team": "C", "away_team": "D", "home_score": 3, "away_score": 1,
         "tournament": "Friendly", "city": "Y", "country": "Y", "neutral": True},
    ]))
    df = load_new_matches(json_path=json_path)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 1


def test_load_new_matches_missing_column_raises(tmp_path):
    json_path = tmp_path / "new.json"
    json_path.write_text(json.dumps([{"date": "2024-03-01", "home_team": "C", "away_team": "D"}]))
    with pytest.raises(ValueError):
        load_new_matches(json_path=json_path)


def test_load_new_matches_requires_a_source():
    with pytest.raises(ValueError):
        load_new_matches()


def test_ingest_appends_and_sorts_new_rows(tmp_path):
    results_path = tmp_path / "results.csv"
    _write_existing_csv(results_path)

    new_df = pd.DataFrame([
        {"date": "2023-06-01", "home_team": "E", "away_team": "F", "home_score": 0, "away_score": 0,
         "tournament": "Friendly", "city": "Z", "country": "Z", "neutral": True},
    ])[REQUIRED_COLUMNS]

    added, total = ingest(new_df, results_path)

    assert added == 1
    assert total == 3
    out = pd.read_csv(results_path)
    assert out["date"].is_monotonic_increasing
    assert "E" in out["home_team"].values


def test_ingest_is_idempotent_for_exact_duplicate_reingestion(tmp_path):
    results_path = tmp_path / "results.csv"
    _write_existing_csv(results_path)

    new_df = pd.DataFrame([
        {"date": "2023-06-01", "home_team": "E", "away_team": "F", "home_score": 0, "away_score": 0,
         "tournament": "Friendly", "city": "Z", "country": "Z", "neutral": True},
    ])[REQUIRED_COLUMNS]

    ingest(new_df, results_path)
    added_second_time, total = ingest(new_df, results_path)

    assert added_second_time == 0
    assert total == 3


def test_ingest_preserves_distinct_matches_sharing_date_teams_and_tournament(tmp_path):
    # Real-world case found in results.csv: doubleheaders/replays that share
    # (date, home_team, away_team, tournament) but have different scores are
    # genuinely distinct matches, not duplicates -- both must survive.
    results_path = tmp_path / "results.csv"
    pd.DataFrame([
        {"date": "1973-09-04", "home_team": "Singapore", "away_team": "Malaysia", "home_score": 0, "away_score": 0,
         "tournament": "SEAP Games", "city": "Singapore", "country": "Singapore", "neutral": False},
    ]).to_csv(results_path, index=False)

    same_key_different_score = pd.DataFrame([
        {"date": "1973-09-04", "home_team": "Singapore", "away_team": "Malaysia", "home_score": 0, "away_score": 3,
         "tournament": "SEAP Games", "city": "Singapore", "country": "Singapore", "neutral": False},
    ])[REQUIRED_COLUMNS]

    added, total = ingest(same_key_different_score, results_path)

    assert added == 1
    assert total == 2
    out = pd.read_csv(results_path)
    assert set(out["away_score"]) == {0, 3}
