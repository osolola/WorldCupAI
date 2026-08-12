from .devig import devig_shin


def fair_probabilities(decimal_odds):
    """De-vigs a {outcome: decimal_odds} dict via Shin's method, returned keyed the same way."""
    outcomes = list(decimal_odds.keys())
    fair, z = devig_shin([decimal_odds[o] for o in outcomes])
    return {o: p for o, p in zip(outcomes, fair)}, z


def edge(model_probs, decimal_odds):
    """model probability minus de-vigged fair market probability, per outcome."""
    fair, _ = fair_probabilities(decimal_odds)
    return {o: model_probs[o] - fair[o] for o in model_probs}


def closing_line_value(entry_decimal_odds, closing_decimal_odds, side):
    """
    CLV in probability space for a position taken on `side`. Positive CLV
    means the market's fair probability for `side` rose between entry and
    close -- the market moved toward your position after you took it, which
    is the standard proxy for genuine predictive edge (it's assessable
    immediately, unlike a single match's win/loss outcome which is heavily
    influenced by variance).
    """
    entry_fair, _ = fair_probabilities(entry_decimal_odds)
    closing_fair, _ = fair_probabilities(closing_decimal_odds)
    return closing_fair[side] - entry_fair[side]
