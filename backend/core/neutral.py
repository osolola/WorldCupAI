def symmetrize(raw_proba_fn, team_a, team_b):
    """
    Wraps a raw (home, away) -> (p_home_win, p_away_win, p_draw) predictor
    into a neutral-site, order-independent (team_a, team_b) predictor by
    averaging both team orderings. This cancels out whatever home-field
    bias the raw model learned from a training set dominated by
    non-neutral matches -- appropriate for competitions played at neutral
    venues. Models that explicitly parametrize home advantage (e.g.
    Dixon-Coles, which takes a `neutral` flag directly) don't need this.
    """
    p_ab = raw_proba_fn(team_a, team_b)
    p_ba = raw_proba_fn(team_b, team_a)

    return {
        "team_a_win": float((p_ab[0] + p_ba[1]) / 2),
        "team_b_win": float((p_ab[1] + p_ba[0]) / 2),
        "draw": float((p_ab[2] + p_ba[2]) / 2),
    }
