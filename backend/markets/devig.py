import numpy as np
from scipy.optimize import brentq


def implied_probabilities(decimal_odds):
    return [1.0 / o for o in decimal_odds]


def devig_multiplicative(decimal_odds):
    """Simplest de-vig: normalize raw implied probabilities so they sum to 1."""
    implied = implied_probabilities(decimal_odds)
    total = sum(implied)
    return [p / total for p in implied]


def devig_shin(decimal_odds, z_bounds=(1e-6, 0.5)):
    """
    Shin's (1992) method: models the bookmaker's overround as coming partly
    from a fraction z of "insider"/informed money, which the book prices
    around by distorting longshots more than favorites (the well-documented
    favorite-longshot bias). Solves for z such that the resulting fair
    probabilities sum to 1, then returns (fair_probabilities, z).

    Falls back to the multiplicative method (and reports z=0.0) if no root
    is bracketed in z_bounds -- this happens for near-zero-overround inputs,
    where the multiplicative and Shin answers converge anyway.
    """
    p = np.array(implied_probabilities(decimal_odds))
    total_p = p.sum()

    def fair_at(z):
        if z <= 0:
            return p / total_p
        return (np.sqrt(z ** 2 + 4 * (1 - z) * (p ** 2) / total_p) - z) / (2 * (1 - z))

    def residual(z):
        return fair_at(z).sum() - 1.0

    lo, hi = z_bounds
    try:
        if residual(lo) * residual(hi) > 0:
            return devig_multiplicative(decimal_odds), 0.0
        z = brentq(residual, lo, hi)
    except (ValueError, RuntimeError):
        return devig_multiplicative(decimal_odds), 0.0

    return fair_at(z).tolist(), float(z)
