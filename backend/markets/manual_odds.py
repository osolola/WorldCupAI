import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "odds_snapshots.jsonl"


def record_snapshot(
    match_id,
    team_a,
    team_b,
    source,
    decimal_odds,
    is_closing=False,
    captured_at=None,
    log_path=DEFAULT_LOG_PATH,
):
    """
    Appends one odds snapshot to the append-only log. decimal_odds must have
    keys "team_a_win", "team_b_win", "draw". captured_at defaults to now
    (UTC, ISO 8601) -- always record it explicitly rather than trusting
    file mtimes, since CLV is only legitimate if we know exactly when a
    price was observed relative to kickoff.
    """
    missing = {"team_a_win", "team_b_win", "draw"} - set(decimal_odds)
    if missing:
        raise ValueError(f"decimal_odds missing keys: {missing}")

    record = {
        "match_id": match_id,
        "team_a": team_a,
        "team_b": team_b,
        "source": source,
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
        "is_closing": bool(is_closing),
        "decimal_odds": {k: float(v) for k, v in decimal_odds.items()},
    }

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def load_snapshots(log_path=DEFAULT_LOG_PATH):
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def snapshots_for_match(match_id, log_path=DEFAULT_LOG_PATH):
    return [s for s in load_snapshots(log_path) if s["match_id"] == match_id]


def latest_snapshot(match_id, source=None, closing_only=False, log_path=DEFAULT_LOG_PATH):
    """Most recently captured snapshot for a match, optionally filtered by source and/or closing-only."""
    candidates = snapshots_for_match(match_id, log_path)
    if source:
        candidates = [s for s in candidates if s["source"] == source]
    if closing_only:
        candidates = [s for s in candidates if s["is_closing"]]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s["captured_at"])
