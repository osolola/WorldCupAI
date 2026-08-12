from backend.markets.clv import closing_line_value, edge, fair_probabilities


def test_fair_probabilities_sums_to_one():
    fair, z = fair_probabilities({"team_a_win": 1.30, "team_b_win": 12.0, "draw": 4.5})
    assert abs(sum(fair.values()) - 1.0) < 1e-6
    assert set(fair.keys()) == {"team_a_win", "team_b_win", "draw"}


def test_edge_positive_when_model_more_bullish_than_market():
    decimal_odds = {"team_a_win": 3.0, "team_b_win": 2.5, "draw": 3.4}
    fair, _ = fair_probabilities(decimal_odds)
    model_probs = dict(fair)
    model_probs["team_a_win"] += 0.1
    model_probs["team_b_win"] -= 0.1

    result = edge(model_probs, decimal_odds)
    assert result["team_a_win"] > 0
    assert result["team_b_win"] < 0


def test_closing_line_value_positive_when_market_moves_toward_your_side():
    entry_odds = {"team_a_win": 3.0, "team_b_win": 2.5, "draw": 3.4}     # team_a_win fair ~0.32
    closing_odds = {"team_a_win": 2.0, "team_b_win": 3.8, "draw": 3.8}   # odds shortened -> fair ~0.49

    clv = closing_line_value(entry_odds, closing_odds, side="team_a_win")
    assert clv > 0


def test_closing_line_value_negative_when_market_moves_away_from_your_side():
    entry_odds = {"team_a_win": 2.0, "team_b_win": 3.8, "draw": 3.8}
    closing_odds = {"team_a_win": 3.0, "team_b_win": 2.5, "draw": 3.4}

    clv = closing_line_value(entry_odds, closing_odds, side="team_a_win")
    assert clv < 0
