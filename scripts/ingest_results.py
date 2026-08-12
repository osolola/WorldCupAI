"""
Appends newly completed match results to results.csv, the historical
dataset every rating and model in this project is computed from. Run this
whenever new results are available, then restart the API -- main.py
recomputes ratings and retrains every model from results.csv fresh on
every startup, so there's no separate "recompute" step: ingest, then
restart.

This is the extension point Phase 1's MatchResultSource abstraction was
built for: to keep this system producing predictions for an ongoing
competition (e.g. a weekly league) after the World Cup ends, write a
source that pulls new results from that competition's data provider and
feed its output through this same path, or register it directly as a new
data source in config.toml -- no changes needed to core/ or
competitions/knockout.py either way.

Usage (from the project root):
    python3 scripts/ingest_results.py --json new_matches.json
    python3 scripts/ingest_results.py --csv new_matches.csv
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.ingestion import ingest, load_new_matches  # noqa: E402

RESULTS_PATH = REPO_ROOT / "results.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", help="Path to a JSON file: a list of match records")
    parser.add_argument("--csv", help="Path to a CSV file with the same columns as results.csv")
    args = parser.parse_args()

    if not args.json and not args.csv:
        parser.error("provide --json or --csv")

    new_df = load_new_matches(json_path=args.json, csv_path=args.csv)
    added, total = ingest(new_df, RESULTS_PATH)

    print(f"Added {added} new match(es). results.csv now has {total} rows.")
    print("Restart the API (uvicorn backend.main:app) to pick up the new ratings.")


if __name__ == "__main__":
    main()
