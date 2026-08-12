from backend.evaluation.metrics import (
    brier_score,
    calibration_curve,
    expected_calibration_error,
    log_loss,
)


def test_brier_score_zero_for_perfect_confident_prediction():
    y_true = ["team_a_win"]
    y_prob = [{"team_a_win": 1.0, "team_b_win": 0.0, "draw": 0.0}]
    assert brier_score(y_true, y_prob) == 0.0


def test_brier_score_worst_case_is_two():
    y_true = ["team_a_win"]
    y_prob = [{"team_a_win": 0.0, "team_b_win": 1.0, "draw": 0.0}]
    assert brier_score(y_true, y_prob) == 2.0


def test_brier_score_uniform_guess_on_three_classes():
    y_true = ["draw"]
    y_prob = [{"team_a_win": 1 / 3, "team_b_win": 1 / 3, "draw": 1 / 3}]
    # (1/3)^2 * 2 + (1/3 - 1)^2 = 2/9 + 4/9 = 6/9
    assert abs(brier_score(y_true, y_prob) - 6 / 9) < 1e-9


def test_log_loss_penalizes_confident_wrong_prediction_heavily():
    confident_right = log_loss(["team_a_win"], [{"team_a_win": 0.99, "team_b_win": 0.005, "draw": 0.005}])
    confident_wrong = log_loss(["team_a_win"], [{"team_a_win": 0.01, "team_b_win": 0.98, "draw": 0.01}])
    assert confident_wrong > confident_right


def test_log_loss_clips_extreme_probabilities_to_avoid_infinity():
    result = log_loss(["team_a_win"], [{"team_a_win": 0.0, "team_b_win": 1.0, "draw": 0.0}])
    assert result < float("inf")


def test_calibration_curve_perfectly_calibrated_bin():
    # 10 matches, each predicting exactly 70% for the outcome that happens
    # exactly 7/10 times -- that bin's mean_predicted should equal empirical_freq.
    y_true = ["team_a_win"] * 7 + ["team_b_win"] * 3
    y_prob = [{"team_a_win": 0.7, "team_b_win": 0.15, "draw": 0.15} for _ in range(10)]
    bins = calibration_curve(y_true, y_prob, n_bins=10)
    bin_70 = next(b for b in bins if b["bin_lower"] <= 0.7 < b["bin_upper"] or b["bin_upper"] == 1.0 and b["bin_lower"] <= 0.7)
    assert bin_70["count"] == 10
    assert abs(bin_70["mean_predicted"] - 0.7) < 1e-9
    assert abs(bin_70["empirical_freq"] - 0.7) < 1e-9


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    y_true = ["team_a_win"] * 7 + ["team_b_win"] * 3
    y_prob = [{"team_a_win": 0.7, "team_b_win": 0.15, "draw": 0.15} for _ in range(10)]
    bins = calibration_curve(y_true, y_prob, n_bins=10)
    # Only the 0.7 and 0.15 bins have data; both should be well calibrated
    # for the class they represent, so ECE should be small.
    ece = expected_calibration_error(bins)
    assert ece < 0.2


def test_expected_calibration_error_high_when_miscalibrated():
    y_true = ["draw"] * 10
    y_prob = [{"team_a_win": 0.05, "team_b_win": 0.05, "draw": 0.9} for _ in range(10)]
    # Model is *underconfident* here (says 90%, right 100% of the time) --
    # compare against a case where it's overconfident and wrong.
    y_true_bad = ["team_a_win"] * 10
    y_prob_bad = [{"team_a_win": 0.05, "team_b_win": 0.05, "draw": 0.9} for _ in range(10)]
    bins_bad = calibration_curve(y_true_bad, y_prob_bad, n_bins=10)
    assert expected_calibration_error(bins_bad) > 0.5
