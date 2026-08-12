def kelly_fraction(model_prob, decimal_odds, fraction_cap=1.0):
    """
    Kelly criterion stake as a fraction of bankroll: f* = (b*p - q) / b,
    where b = decimal_odds - 1 (net odds), p = model's win probability,
    q = 1 - p. A non-positive edge clips to 0 (no bet). fraction_cap scales
    down to "fractional Kelly" for risk control (e.g. 0.25 = quarter Kelly)
    -- standard practice, since full Kelly is high-variance under any model
    uncertainty and this model's edges are estimates, not ground truth.

    Simulation-only sizing: this function does not place, size, or execute
    any real bet.
    """
    b = decimal_odds - 1
    if b <= 0:
        return 0.0

    p = model_prob
    q = 1 - p
    f = (b * p - q) / b
    return max(0.0, f) * fraction_cap
