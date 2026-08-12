from backend.evaluation.paper_trading import paper_trade


def test_paper_trade_skips_bets_with_no_edge():
    bets = [
        {"model_prob": 0.5, "market_prob": 0.5, "decimal_odds": 2.0, "won": True},
        {"model_prob": 0.4, "market_prob": 0.5, "decimal_odds": 2.0, "won": True},  # negative edge
    ]
    result = paper_trade(bets, initial_bankroll=1.0)
    assert result["n_bets"] == 0
    assert result["n_skipped"] == 2
    assert result["roi"] == 0.0
    assert result["equity_curve"] == [1.0]


def test_paper_trade_tracks_bankroll_roi_and_drawdown():
    # Full Kelly, decimal_odds=2.0, model_prob=0.75 -> stake 50% of bankroll each time.
    bets = [
        {"model_prob": 0.75, "market_prob": 0.5, "decimal_odds": 2.0, "won": True},   # 1.0 -> 1.5
        {"model_prob": 0.75, "market_prob": 0.5, "decimal_odds": 2.0, "won": False},  # 1.5 -> 0.75
        {"model_prob": 0.75, "market_prob": 0.5, "decimal_odds": 2.0, "won": True},   # 0.75 -> 1.125
    ]
    result = paper_trade(bets, initial_bankroll=1.0, kelly_fraction_cap=1.0)

    assert abs(result["final_bankroll"] - 1.125) < 1e-9
    assert abs(result["roi"] - 0.125) < 1e-9
    assert abs(result["max_drawdown"] - 0.5) < 1e-9
    assert result["n_bets"] == 3
    assert result["n_skipped"] == 0
    assert abs(result["win_rate"] - 2 / 3) < 1e-9


def test_paper_trade_fractional_kelly_reduces_drawdown_vs_full_kelly():
    bets = [
        {"model_prob": 0.75, "market_prob": 0.5, "decimal_odds": 2.0, "won": True},
        {"model_prob": 0.75, "market_prob": 0.5, "decimal_odds": 2.0, "won": False},
        {"model_prob": 0.75, "market_prob": 0.5, "decimal_odds": 2.0, "won": False},
    ]
    full = paper_trade(bets, initial_bankroll=1.0, kelly_fraction_cap=1.0)
    quarter = paper_trade(bets, initial_bankroll=1.0, kelly_fraction_cap=0.25)
    assert quarter["max_drawdown"] < full["max_drawdown"]


def test_paper_trade_bankroll_never_goes_nonpositive_under_kelly_sizing():
    # Kelly stakes are always a fraction < 1 of current bankroll, so a long
    # losing streak should shrink the bankroll toward (but never to) zero.
    bets = [{"model_prob": 0.75, "market_prob": 0.5, "decimal_odds": 2.0, "won": False} for _ in range(50)]
    result = paper_trade(bets, initial_bankroll=1.0, kelly_fraction_cap=1.0)
    assert result["final_bankroll"] > 0.0
    assert result["roi"] < 0.0
