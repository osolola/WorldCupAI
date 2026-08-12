from backend.markets.kelly import kelly_fraction


def test_kelly_fraction_zero_at_break_even():
    # decimal_odds=2.0 -> break-even probability is exactly 0.5
    assert abs(kelly_fraction(model_prob=0.5, decimal_odds=2.0)) < 1e-9


def test_kelly_fraction_positive_edge():
    f = kelly_fraction(model_prob=0.6, decimal_odds=2.0)
    assert abs(f - 0.2) < 1e-9


def test_kelly_fraction_negative_edge_clips_to_zero():
    f = kelly_fraction(model_prob=0.4, decimal_odds=2.0)
    assert f == 0.0


def test_kelly_fraction_cap_scales_down_stake():
    full = kelly_fraction(model_prob=0.6, decimal_odds=2.0, fraction_cap=1.0)
    quarter = kelly_fraction(model_prob=0.6, decimal_odds=2.0, fraction_cap=0.25)
    assert abs(quarter - full * 0.25) < 1e-9


def test_kelly_fraction_zero_odds_below_evens_is_safe():
    assert kelly_fraction(model_prob=0.9, decimal_odds=1.0) == 0.0
