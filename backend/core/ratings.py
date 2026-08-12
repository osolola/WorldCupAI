import numpy as np
import pandas as pd


def expected_result(rating_a, rating_b, divisor=600):
    """Win probability for A given an Elo-style rating gap."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / divisor))


def update_elo(rating, opponent_rating, actual_score, k, divisor=600):
    expected = expected_result(rating, opponent_rating, divisor)
    return rating + k * (actual_score - expected)


def expected_goals(attack_rating, defense_rating, avg_goals, divisor=400, diff_clip=600, goals_clip=(0.15, 6.0)):
    """Goals a team is expected to score, based on its Attack rating vs. the opponent's Defense rating."""
    diff = np.clip(attack_rating - defense_rating, -diff_clip, diff_clip)
    xg = avg_goals * (10 ** (diff / divisor))
    return float(np.clip(xg, *goals_clip))


def compute_ratings(
    df,
    base_rating=1500.0,
    elo_k_default=30,
    elo_k_overrides=None,
    attack_defense_k=8.0,
    elo_divisor=600,
    goals_divisor=400,
    rating_diff_clip=600,
    expected_goals_clip=(0.15, 6.0),
):
    """
    Computes, for every match in chronological order:
      - Overall Elo (win/draw/loss based), with a per-tournament K-factor
        override (e.g. {"FIFA World Cup": 60}) so any competition can weight
        its own high-stakes matches without this engine knowing what a
        "World Cup" is.
      - A separate Attack and Defense rating per team, updated from goals
        scored vs. goals expected (Attack rises by outscoring the opponent's
        Defense; Defense rises by conceding fewer goals than expected).

    df must already be sorted by date ascending (a MatchResultSource
    guarantees this). Returns the input df with rating columns attached,
    plus a {team: {elo, attack, defense}} dict of final ratings.
    """
    elo_k_overrides = elo_k_overrides or {}

    avg_goals = float(pd.concat([df["home_score"], df["away_score"]]).mean())

    elo, attack, defense = {}, {}, {}
    home_elo, away_elo = [], []
    home_attack, home_defense = [], []
    away_attack, away_defense = [], []

    for row in df.itertuples():
        home, away = row.home_team, row.away_team
        h_elo, a_elo = elo.get(home, base_rating), elo.get(away, base_rating)
        h_atk, h_def = attack.get(home, base_rating), defense.get(home, base_rating)
        a_atk, a_def = attack.get(away, base_rating), defense.get(away, base_rating)

        home_elo.append(h_elo)
        away_elo.append(a_elo)
        home_attack.append(h_atk)
        home_defense.append(h_def)
        away_attack.append(a_atk)
        away_defense.append(a_def)

        h_score, a_score = row.home_score, row.away_score
        if h_score > a_score:
            h_actual, a_actual = 1, 0
        elif h_score < a_score:
            h_actual, a_actual = 0, 1
        else:
            h_actual, a_actual = 0.5, 0.5

        k = elo_k_overrides.get(row.tournament, elo_k_default)
        elo[home] = update_elo(h_elo, a_elo, h_actual, k, elo_divisor)
        elo[away] = update_elo(a_elo, h_elo, a_actual, k, elo_divisor)

        xg_home = expected_goals(h_atk, a_def, avg_goals, goals_divisor, rating_diff_clip, expected_goals_clip)
        xg_away = expected_goals(a_atk, h_def, avg_goals, goals_divisor, rating_diff_clip, expected_goals_clip)
        home_err = h_score - xg_home
        away_err = a_score - xg_away

        attack[home] = h_atk + attack_defense_k * home_err
        defense[away] = a_def - attack_defense_k * home_err
        attack[away] = a_atk + attack_defense_k * away_err
        defense[home] = h_def - attack_defense_k * away_err

    df = df.copy()
    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    df["home_attack"] = home_attack
    df["home_defense"] = home_defense
    df["away_attack"] = away_attack
    df["away_defense"] = away_defense

    ratings = {
        team: {
            "elo": elo[team],
            "attack": attack.get(team, base_rating),
            "defense": defense.get(team, base_rating),
        }
        for team in elo
    }

    return df, ratings
