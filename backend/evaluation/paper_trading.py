from ..markets.kelly import kelly_fraction


def paper_trade(bets, initial_bankroll=1.0, kelly_fraction_cap=0.25):
    """
    Simulates sequentially staking Kelly-sized bets against de-vigged market
    probabilities and tracks bankroll over time. This is NOT a real trading
    engine -- no execution, no slippage, no liquidity limits -- it scores
    "if I'd staked according to my model's edge on every logged prediction,
    what would have happened." Simulation only; never places a real bet.

    bets: chronologically ordered list of dicts, each:
      {"model_prob": float,     model's probability this side wins
       "market_prob": float,    de-vigged fair market probability, same side
       "decimal_odds": float,   decimal odds actually available for that side
       "won": bool}             whether that side actually won

    Returns roi, max_drawdown (peak-to-trough on the equity curve), the
    equity curve itself, and basic bet-count/win-rate bookkeeping.
    """
    bankroll = initial_bankroll
    equity_curve = [bankroll]
    n_bets = 0
    n_wins = 0

    for bet in bets:
        stake_fraction = kelly_fraction(bet["model_prob"], bet["decimal_odds"], kelly_fraction_cap)
        if stake_fraction <= 0:
            continue

        stake = bankroll * stake_fraction
        n_bets += 1
        if bet["won"]:
            bankroll += stake * (bet["decimal_odds"] - 1)
            n_wins += 1
        else:
            bankroll -= stake
        equity_curve.append(bankroll)

    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)

    return {
        "roi": (bankroll - initial_bankroll) / initial_bankroll,
        "max_drawdown": max_drawdown,
        "final_bankroll": bankroll,
        "equity_curve": equity_curve,
        "n_bets": n_bets,
        "n_skipped": len(bets) - n_bets,
        "win_rate": (n_wins / n_bets) if n_bets else None,
    }
