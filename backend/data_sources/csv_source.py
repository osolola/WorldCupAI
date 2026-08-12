from pathlib import Path

import pandas as pd


class CsvMatchResultSource:
    """Loads historical match results from a static CSV file."""

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
