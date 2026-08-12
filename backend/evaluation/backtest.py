import numpy as np
import pandas as pd

from ..core.dixon_coles import fit_dixon_coles
from ..core.elo_classifier import FEATURE_COLUMNS, build_features, train_model
from ..core.ratings import compute_ratings
from .baselines import fit_always_favorite_baseline, fit_elo_only_baseline, uniform_baseline
from .metrics import brier_score, calibration_curve, expected_calibration_error, log_loss


def _outcome_labels(df):
    return np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["team_a_win", "team_b_win"],
        default="draw",
    )


def _target_codes(df):
    return np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        [0, 1],
        default=2,
    )


def _favorite_outcomes(df):
    home_win = (df["home_score"] > df["away_score"]).to_numpy()
    away_win = (df["home_score"] < df["away_score"]).to_numpy()
    draw = ~home_win & ~away_win
    is_favorite = (df["home_elo"] >= df["away_elo"]).to_numpy()
    return np.select(
        [draw, home_win == is_favorite],
        ["draw", "favorite_win"],
        default="underdog_win",
    )


def run_backtest(
    raw_df,
    ratings_cfg,
    model_cfg,
    dixon_coles_cfg,
    elo_k_overrides,
    holdout_start_date,
    calibration_bins=10,
    refit_period="Y",
):
    """
    Chronological holdout backtest. Elo/Attack-Defense ratings are computed
    once, walk-forward, over the entire match history -- they're inherently
    no-lookahead by construction, so this alone doesn't leak. The XGBoost
    classifier, Dixon-Coles model, and the fitted baselines ARE refit at
    every period boundary (default: yearly) using only data strictly before
    that boundary, then scored on the matches within that period only. This
    mirrors how a production system retrained periodically would actually
    perform, and keeps the comparison across models apples-to-apples: no
    model, anywhere in this backtest, ever sees a result before it's scored
    on predicting it.

    Models are evaluated in their *native* home/away frame using each
    match's actual home/away assignment and `neutral` flag -- not the
    neutral-symmetrized serving-time view used for hypothetical neutral-site
    matchups -- since these are real historical matches with a real venue.
    """
    df, _ = compute_ratings(
        raw_df,
        base_rating=ratings_cfg["base_rating"],
        elo_k_default=ratings_cfg["elo_k_default"],
        elo_k_overrides=elo_k_overrides,
        attack_defense_k=ratings_cfg["attack_defense_k"],
        elo_divisor=ratings_cfg["elo_divisor"],
        goals_divisor=ratings_cfg["goals_divisor"],
        rating_diff_clip=ratings_cfg["rating_diff_clip"],
        expected_goals_clip=(ratings_cfg["expected_goals_clip_min"], ratings_cfg["expected_goals_clip_max"]),
    )

    holdout_start = pd.Timestamp(holdout_start_date)
    holdout_mask = df["date"] >= holdout_start
    if not holdout_mask.any():
        raise ValueError("holdout_start_date leaves no matches in the holdout set")

    periods = sorted(df.loc[holdout_mask, "date"].dt.to_period(refit_period).unique())

    model_names = ["dixon_coles", "xgboost_elo", "elo_only", "always_favorite", "uniform"]
    y_true = {name: [] for name in model_names}
    y_prob = {name: [] for name in model_names}
    match_meta = []
    periods_evaluated = []

    for period in periods:
        period_start = period.start_time
        period_end = period.end_time

        train_df = df[df["date"] < period_start].reset_index(drop=True)
        eval_df = df[(df["date"] >= period_start) & (df["date"] <= period_end) & holdout_mask].reset_index(drop=True)
        if train_df.empty or eval_df.empty:
            continue
        periods_evaluated.append(str(period))

        xgb_model = train_model(
            train_df,
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            learning_rate=model_cfg["learning_rate"],
            random_seed=model_cfg["random_seed"],
        )

        dc_model, _ = fit_dixon_coles(
            train_df,
            xi=dixon_coles_cfg["xi"],
            reg_lambda=dixon_coles_cfg["reg_lambda"],
            min_effective_weight=dixon_coles_cfg["min_effective_weight"],
            max_goals=dixon_coles_cfg["max_goals"],
            maxiter=dixon_coles_cfg["maxiter"],
            maxfun=dixon_coles_cfg["maxfun"],
            as_of=train_df["date"].max(),
        )

        elo_only_raw = fit_elo_only_baseline(
            (train_df["home_elo"] - train_df["away_elo"]).tolist(),
            _target_codes(train_df).tolist(),
        )
        always_favorite_predict = fit_always_favorite_baseline(list(_favorite_outcomes(train_df)))

        eval_outcomes = _outcome_labels(eval_df)
        eval_features = build_features(eval_df)[FEATURE_COLUMNS]
        xgb_probs = xgb_model.predict_proba(eval_features)
        # A training window can lack a class entirely (e.g. no draws yet) --
        # xgb_model.classes_ then omits it, so map by class label rather than
        # assuming column 0/1/2 always means home/away/draw.
        xgb_class_to_col = {c: idx for idx, c in enumerate(xgb_model.classes_)}

        for i in range(len(eval_df)):
            row = eval_df.iloc[i]
            outcome = eval_outcomes[i]
            home, away = row["home_team"], row["away_team"]

            y_true["dixon_coles"].append(outcome)
            y_prob["dixon_coles"].append(dc_model.match_probabilities(home, away, neutral=bool(row["neutral"])))

            y_true["xgboost_elo"].append(outcome)
            p = xgb_probs[i]
            y_prob["xgboost_elo"].append({
                "team_a_win": float(p[xgb_class_to_col[0]]) if 0 in xgb_class_to_col else 0.0,
                "team_b_win": float(p[xgb_class_to_col[1]]) if 1 in xgb_class_to_col else 0.0,
                "draw": float(p[xgb_class_to_col[2]]) if 2 in xgb_class_to_col else 0.0,
            })

            y_true["elo_only"].append(outcome)
            raw = elo_only_raw(row["home_elo"], row["away_elo"])
            y_prob["elo_only"].append({"team_a_win": raw[0], "team_b_win": raw[1], "draw": raw[2]})

            y_true["always_favorite"].append(outcome)
            y_prob["always_favorite"].append(always_favorite_predict(row["home_elo"] >= row["away_elo"]))

            match_meta.append({
                "date": str(row["date"].date()),
                "team_a": home,
                "team_b": away,
                "outcome": outcome,
                "neutral": bool(row["neutral"]),
            })

        y_true["uniform"].extend(eval_outcomes)
        y_prob["uniform"].extend(uniform_baseline(range(len(eval_df))))

    report = {
        "holdout_start_date": str(holdout_start_date),
        "refit_period": refit_period,
        "periods_evaluated": periods_evaluated,
        "models": {},
    }
    for name in model_names:
        if not y_true[name]:
            continue
        bins = calibration_curve(y_true[name], y_prob[name], n_bins=calibration_bins)
        report["models"][name] = {
            "n_matches": len(y_true[name]),
            "brier_score": brier_score(y_true[name], y_prob[name]),
            "log_loss": log_loss(y_true[name], y_prob[name]),
            "expected_calibration_error": expected_calibration_error(bins),
            "calibration_curve": bins,
        }

    # Honesty over polish: for the two real models, surface the matches
    # where the model was most confidently wrong -- lowest probability
    # assigned to what actually happened.
    for name in ("dixon_coles", "xgboost_elo"):
        if name not in report["models"]:
            continue
        misses = []
        for i, meta in enumerate(match_meta):
            prob_of_actual = y_prob[name][i][y_true[name][i]]
            misses.append({**meta, "predicted_probabilities": y_prob[name][i], "probability_of_actual": prob_of_actual})
        misses.sort(key=lambda m: m["probability_of_actual"])
        report["models"][name]["biggest_misses"] = misses[:10]

    return report
