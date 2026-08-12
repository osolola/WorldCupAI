from backend.markets.devig import devig_multiplicative, devig_shin, implied_probabilities


def test_implied_probabilities_is_inverse_of_decimal_odds():
    assert implied_probabilities([2.0, 4.0]) == [0.5, 0.25]


def test_devig_multiplicative_sums_to_one():
    fair = devig_multiplicative([1.30, 4.5, 12.0])
    assert abs(sum(fair) - 1.0) < 1e-9


def test_devig_multiplicative_preserves_relative_ordering():
    fair = devig_multiplicative([1.30, 4.5, 12.0])
    assert fair[0] > fair[1] > fair[2]


def test_devig_multiplicative_no_overround_is_unchanged():
    # Decimal odds of 2.0/2.0 imply exactly 0.5/0.5 with zero overround already.
    fair = devig_multiplicative([2.0, 2.0])
    assert abs(fair[0] - 0.5) < 1e-9
    assert abs(fair[1] - 0.5) < 1e-9


def test_devig_shin_sums_to_one():
    fair, z = devig_shin([1.30, 4.5, 12.0])
    assert abs(sum(fair) - 1.0) < 1e-6
    assert 0.0 <= z <= 0.5


def test_devig_shin_pulls_probability_from_longshot_to_favorite():
    # The documented favorite-longshot bias: Shin's method should shift
    # fair probability mass away from the longshot and toward the favorite,
    # relative to naive multiplicative normalization.
    mult = devig_multiplicative([1.30, 4.5, 12.0])
    shin, z = devig_shin([1.30, 4.5, 12.0])

    assert z > 0.0
    assert shin[0] > mult[0]   # favorite gets more probability under Shin
    assert shin[2] < mult[2]   # longshot gets less probability under Shin


def test_devig_shin_degenerates_to_multiplicative_for_two_outcomes():
    # With only two outcomes there's a single degree of freedom given they
    # must sum to 1, so Shin's correction has no room to differ.
    mult = devig_multiplicative([1.15, 15.0])
    shin, _ = devig_shin([1.15, 15.0])
    assert abs(mult[0] - shin[0]) < 1e-6
    assert abs(mult[1] - shin[1]) < 1e-6
