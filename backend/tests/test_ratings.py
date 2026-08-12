import pandas as pd

from backend.core.ratings import compute_ratings, expected_result, update_elo


def _match(date, home, away, hs, away_score, tournament="Friendly"):
    return {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": away_score,
        "tournament": tournament,
    }


def _df(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_expected_result_symmetric_at_equal_ratings():
    assert expected_result(1500, 1500) == 0.5


def test_expected_result_favors_higher_rating():
    assert expected_result(1600, 1500) > 0.5


def test_update_elo_increases_on_win():
    assert update_elo(1500, 1500, actual_score=1, k=30) > 1500


def test_compute_ratings_new_teams_start_at_base_rating():
    df = _df([_match("2020-01-01", "A", "B", 1, 1)])
    processed, _ = compute_ratings(df, base_rating=1500.0)
    assert processed.loc[0, "home_elo"] == 1500.0
    assert processed.loc[0, "away_elo"] == 1500.0


def test_compute_ratings_winner_gains_elo_loser_loses_elo():
    df = _df([_match("2020-01-01", "A", "B", 3, 0)])
    _, ratings = compute_ratings(df, base_rating=1500.0, elo_k_default=30)
    assert ratings["A"]["elo"] > 1500.0
    assert ratings["B"]["elo"] < 1500.0


def test_compute_ratings_respects_tournament_k_override():
    df = _df([_match("2020-01-01", "A", "B", 1, 0, tournament="FIFA World Cup")])
    _, low_k = compute_ratings(df.copy(), elo_k_default=30, elo_k_overrides={})
    _, high_k = compute_ratings(df.copy(), elo_k_default=30, elo_k_overrides={"FIFA World Cup": 60})
    assert high_k["A"]["elo"] > low_k["A"]["elo"]


def test_attack_rises_and_conceding_defense_falls_when_outscoring_expectation():
    rows = [_match(f"2020-01-0{i + 1}", "A", "B", 5, 0) for i in range(3)]
    df = _df(rows)
    _, ratings = compute_ratings(df)
    assert ratings["A"]["attack"] > 1500.0
    assert ratings["B"]["defense"] < 1500.0
