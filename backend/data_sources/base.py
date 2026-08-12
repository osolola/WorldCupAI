from typing import Protocol

import pandas as pd


class MatchResultSource(Protocol):
    """Any source of historical match results for a competition.

    load() must return a DataFrame with columns: date, home_team, away_team,
    home_score, away_score, tournament — sorted by date ascending, since the
    rating engine processes matches walk-forward.
    """

    def load(self) -> pd.DataFrame: ...
