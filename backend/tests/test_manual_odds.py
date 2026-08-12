import pytest

from backend.markets.manual_odds import (
    latest_snapshot,
    load_snapshots,
    record_snapshot,
    snapshots_for_match,
)


def test_load_snapshots_empty_when_file_missing(tmp_path):
    assert load_snapshots(tmp_path / "does_not_exist.jsonl") == []


def test_record_snapshot_requires_all_three_outcomes(tmp_path):
    log_path = tmp_path / "odds.jsonl"
    with pytest.raises(ValueError):
        record_snapshot(
            "m1", "France", "Spain", "bookmaker",
            decimal_odds={"team_a_win": 3.0, "team_b_win": 2.5},  # missing "draw"
            log_path=log_path,
        )


def test_record_and_load_round_trip(tmp_path):
    log_path = tmp_path / "odds.jsonl"
    record_snapshot(
        "m1", "France", "Spain", "bookmaker",
        decimal_odds={"team_a_win": 3.0, "team_b_win": 2.5, "draw": 3.2},
        is_closing=False,
        captured_at="2026-07-13T10:00:00Z",
        log_path=log_path,
    )
    record_snapshot(
        "m1", "France", "Spain", "bookmaker",
        decimal_odds={"team_a_win": 2.8, "team_b_win": 2.7, "draw": 3.3},
        is_closing=True,
        captured_at="2026-07-14T14:55:00Z",
        log_path=log_path,
    )

    snapshots = load_snapshots(log_path)
    assert len(snapshots) == 2
    assert snapshots[0]["is_closing"] is False
    assert snapshots[1]["is_closing"] is True


def test_snapshots_for_match_filters_by_match_id(tmp_path):
    log_path = tmp_path / "odds.jsonl"
    record_snapshot("m1", "France", "Spain", "bookmaker", {"team_a_win": 3.0, "team_b_win": 2.5, "draw": 3.2}, log_path=log_path)
    record_snapshot("m2", "England", "Argentina", "bookmaker", {"team_a_win": 2.0, "team_b_win": 3.5, "draw": 3.1}, log_path=log_path)

    m1_only = snapshots_for_match("m1", log_path=log_path)
    assert len(m1_only) == 1
    assert m1_only[0]["match_id"] == "m1"


def test_latest_snapshot_picks_most_recent_and_respects_filters(tmp_path):
    log_path = tmp_path / "odds.jsonl"
    record_snapshot("m1", "France", "Spain", "bookmaker", {"team_a_win": 3.0, "team_b_win": 2.5, "draw": 3.2}, captured_at="2026-07-13T10:00:00Z", log_path=log_path)
    record_snapshot("m1", "France", "Spain", "polymarket", {"team_a_win": 2.9, "team_b_win": 2.6, "draw": 3.1}, captured_at="2026-07-13T12:00:00Z", log_path=log_path)
    record_snapshot("m1", "France", "Spain", "bookmaker", {"team_a_win": 2.8, "team_b_win": 2.7, "draw": 3.3}, is_closing=True, captured_at="2026-07-14T14:55:00Z", log_path=log_path)

    latest_any = latest_snapshot("m1", log_path=log_path)
    assert latest_any["source"] == "bookmaker"
    assert latest_any["is_closing"] is True

    latest_polymarket = latest_snapshot("m1", source="polymarket", log_path=log_path)
    assert latest_polymarket["source"] == "polymarket"

    closing_only = latest_snapshot("m1", closing_only=True, log_path=log_path)
    assert closing_only["is_closing"] is True
    assert closing_only["decimal_odds"]["team_a_win"] == 2.8


def test_latest_snapshot_returns_none_when_no_match(tmp_path):
    log_path = tmp_path / "odds.jsonl"
    assert latest_snapshot("does-not-exist", log_path=log_path) is None
