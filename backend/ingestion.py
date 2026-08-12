import json

import pandas as pd

REQUIRED_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


def load_new_matches(json_path=None, csv_path=None):
    """Loads and validates a batch of newly completed matches from either a JSON records file or a CSV."""
    if json_path:
        with open(json_path) as f:
            new_df = pd.DataFrame(json.load(f))
    elif csv_path:
        new_df = pd.read_csv(csv_path)
    else:
        raise ValueError("must provide json_path or csv_path")

    missing = set(REQUIRED_COLUMNS) - set(new_df.columns)
    if missing:
        raise ValueError(f"new matches missing required columns: {missing}")
    return new_df[REQUIRED_COLUMNS]


def ingest(new_df, results_path):
    """
    Appends newly completed matches to the historical results CSV that all
    ratings/models are computed from.

    Only *exact* full-row duplicates are collapsed (e.g. accidentally
    running ingestion twice with the same file) -- this dataset genuinely
    contains distinct historical matches that share date/home_team/
    away_team/tournament (doubleheaders, same-day replays, and other
    data-entry quirks with no true unique key), so deduplicating on any
    partial key was found to silently delete real rows. That means
    re-ingesting a *corrected* score for an already-ingested match will
    add a second row rather than overwrite the first -- fix those by
    editing results.csv directly.

    Returns (n_added, n_total_rows).
    """
    existing = pd.read_csv(results_path)
    existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
    before = len(existing)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(keep="first")
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    combined.to_csv(results_path, index=False)
    return len(combined) - before, len(combined)
