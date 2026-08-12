import numpy as np
from sklearn.linear_model import LogisticRegression


def uniform_baseline(matches):
    """The most naive possible baseline: 1/3 to every outcome, every match."""
    return [{"team_a_win": 1 / 3, "team_b_win": 1 / 3, "draw": 1 / 3} for _ in matches]


def fit_always_favorite_baseline(train_outcomes_by_favorite):
    """
    Learns three fixed numbers from training data: the empirical rate at
    which the pre-match Elo favorite wins, the underdog wins, and the match
    draws -- then applies those same three numbers to every match regardless
    of the specific teams involved. This is "always pick the favorite," just
    calibrated well enough to be scoreable by Brier/log loss instead of
    handing out raw 100%/0% predictions.

    train_outcomes_by_favorite: list of "favorite_win" | "underdog_win" | "draw".
    Returns a callable outcome-independent predictor: (team_a_is_favorite) -> probs.
    """
    total = len(train_outcomes_by_favorite)
    p_favorite = train_outcomes_by_favorite.count("favorite_win") / total
    p_underdog = train_outcomes_by_favorite.count("underdog_win") / total
    p_draw = train_outcomes_by_favorite.count("draw") / total

    def predict(team_a_is_favorite):
        if team_a_is_favorite:
            return {"team_a_win": p_favorite, "team_b_win": p_underdog, "draw": p_draw}
        return {"team_a_win": p_underdog, "team_b_win": p_favorite, "draw": p_draw}

    return predict


def fit_elo_only_baseline(elo_diffs, targets):
    """
    A minimal, single-feature competitor to the full Elo+Attack/Defense
    model: multinomial logistic regression on Elo difference alone.
    targets: 0 = home win, 1 = away win, 2 = draw (same convention as
    core.elo_classifier, since this is fit on the raw home/away schema and
    symmetrized at prediction time the same way).
    """
    X = np.array(elo_diffs).reshape(-1, 1)
    y = np.array(targets)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    class_to_col = {c: idx for idx, c in enumerate(clf.classes_)}

    def raw_proba(home_elo, away_elo):
        # A training window can lack a class entirely (e.g. no draws yet) --
        # clf.classes_ then omits it, so we default any missing class to 0.0
        # rather than indexing into a probability vector that doesn't have it.
        probs = clf.predict_proba([[home_elo - away_elo]])[0]
        return [float(probs[class_to_col[c]]) if c in class_to_col else 0.0 for c in (0, 1, 2)]

    return raw_proba
