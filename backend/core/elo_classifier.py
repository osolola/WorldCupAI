from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense",
    "attack_diff",
    "defense_diff",
]


def _winner(row):
    if row["home_score"] > row["away_score"]:
        return 0  # home win
    if row["away_score"] > row["home_score"]:
        return 1  # away win
    return 2  # draw


def build_features(df):
    df = df.copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["attack_diff"] = df["home_attack"] - df["away_defense"]
    df["defense_diff"] = df["away_attack"] - df["home_defense"]
    return df


def train_model(df, n_estimators=200, max_depth=4, learning_rate=0.05, random_seed=42):
    """Trains an XGBoost classifier on Elo and Attack/Defense features. Target: 0=Home win, 1=Away win, 2=Draw."""
    df = build_features(df)
    df["target"] = df.apply(_winner, axis=1)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_seed,
        eval_metric="mlogloss",
    )
    model.fit(df[FEATURE_COLUMNS], df["target"])
    return model


def make_feature_row(ratings, home, away):
    h = ratings.get(home, {"elo": 1500.0, "attack": 1500.0, "defense": 1500.0})
    a = ratings.get(away, {"elo": 1500.0, "attack": 1500.0, "defense": 1500.0})

    return [[
        h["elo"],
        a["elo"],
        h["elo"] - a["elo"],
        h["attack"],
        h["defense"],
        a["attack"],
        a["defense"],
        h["attack"] - a["defense"],
        a["attack"] - h["defense"],
    ]]
