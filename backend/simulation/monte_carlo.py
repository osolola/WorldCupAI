import numpy as np


def _wilson_ci(p, n, z=1.96):
    """95% Wilson score interval -- better-behaved than the normal approximation near p=0 or p=1."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _resolve_knockout_match(team_a, team_b, probs, rng):
    outcome = rng.choice(["team_a_win", "team_b_win", "draw"], p=[probs["team_a_win"], probs["team_b_win"], probs["draw"]])
    if outcome == "draw":
        # Knockout ties go to a shootout, weighted by relative in-regulation
        # strength rather than an uninformed 50/50 coin flip.
        denom = probs["team_a_win"] + probs["team_b_win"]
        p_a_shootout = probs["team_a_win"] / denom if denom > 0 else 0.5
        outcome = "team_a_win" if rng.random() < p_a_shootout else "team_b_win"
    return team_a if outcome == "team_a_win" else team_b


def simulate_knockout_bracket(teams, predict_fn, n_sims=10000, seed=42, neutral=True):
    """
    Monte Carlo single-elimination bracket. Each simulated match draws a
    categorical outcome (team_a_win/team_b_win/draw) from
    predict_fn(team_a, team_b, neutral) -> {team_a_win, team_b_win, draw};
    a drawn match is resolved via a simulated penalty shootout (see
    _resolve_knockout_match) rather than sampling full scorelines, since
    only the eventual winner matters for bracket advancement.

    Returns, per team, the fraction of simulations in which it advanced
    past each round and the fraction in which it won the whole bracket,
    each with a 95% confidence interval from simulation variance.
    """
    rng = np.random.default_rng(seed)
    n_rounds = int(np.log2(len(teams)))
    # Label round r (0-indexed) by the bracket size a team advances INTO by
    # winning it, e.g. winning the first round of a 16-team bracket reaches
    # "round_of_8". The final round's "reached" size would be 1, which is
    # just the champion -- tracked separately instead of as a round label.
    round_labels = [f"round_of_{len(teams) // (2 ** (r + 1))}" for r in range(n_rounds - 1)]

    reached = {team: [0] * len(round_labels) for team in teams}
    champion_count = {team: 0 for team in teams}

    for _ in range(n_sims):
        current = list(teams)
        for r in range(n_rounds):
            next_round = []
            for i in range(0, len(current), 2):
                t1, t2 = current[i], current[i + 1]
                probs = predict_fn(t1, t2, neutral)
                winner = _resolve_knockout_match(t1, t2, probs, rng)
                if r < len(round_labels):
                    reached[winner][r] += 1
                next_round.append(winner)
            current = next_round
        champion_count[current[0]] += 1

    results = {}
    for team in teams:
        rounds = {}
        for r, label in enumerate(round_labels):
            p = reached[team][r] / n_sims
            lo, hi = _wilson_ci(p, n_sims)
            rounds[label] = {"probability": p, "ci_low": lo, "ci_high": hi}
        p_champ = champion_count[team] / n_sims
        lo, hi = _wilson_ci(p_champ, n_sims)
        results[team] = {
            "rounds_reached": rounds,
            "champion": {"probability": p_champ, "ci_low": lo, "ci_high": hi},
        }
    return {"n_simulations": n_sims, "seed": seed, "teams": results}


def _group_standings(group_teams, match_results):
    """match_results: list of (team_a, team_b, goals_a, goals_b). Ranks by points, then goal difference, then goals scored."""
    points = {t: 0 for t in group_teams}
    goal_diff = {t: 0 for t in group_teams}
    goals_for = {t: 0 for t in group_teams}

    for team_a, team_b, ga, gb in match_results:
        goal_diff[team_a] += ga - gb
        goal_diff[team_b] += gb - ga
        goals_for[team_a] += ga
        goals_for[team_b] += gb
        if ga > gb:
            points[team_a] += 3
        elif gb > ga:
            points[team_b] += 3
        else:
            points[team_a] += 1
            points[team_b] += 1

    return sorted(group_teams, key=lambda t: (points[t], goal_diff[t], goals_for[t]), reverse=True)


def simulate_group_stage(groups, score_sampler, advance_per_group=2, n_sims=10000, seed=42, neutral=True):
    """
    Round-robin group stage (each pair in a group plays once). Full
    scorelines are simulated via score_sampler(team_a, team_b, neutral, rng)
    -> (goals_a, goals_b) -- e.g. DixonColesModel.sample_score -- because
    standard football tiebreakers need goal difference and goals scored,
    not just points. Ties beyond points/GD/GF are broken by simulated
    match order (head-to-head and disciplinary tiebreakers aren't modeled).

    groups: {group_name: [team, ...]}. Returns, per team, the fraction of
    simulations in which it finished within the top `advance_per_group` of
    its group, with a 95% confidence interval.
    """
    rng = np.random.default_rng(seed)
    advance_count = {team: 0 for group in groups.values() for team in group}

    for _ in range(n_sims):
        for group_teams in groups.values():
            match_results = []
            for i in range(len(group_teams)):
                for j in range(i + 1, len(group_teams)):
                    team_a, team_b = group_teams[i], group_teams[j]
                    ga, gb = score_sampler(team_a, team_b, neutral, rng)
                    match_results.append((team_a, team_b, ga, gb))
            ranked = _group_standings(group_teams, match_results)
            for team in ranked[:advance_per_group]:
                advance_count[team] += 1

    results = {}
    for team, count in advance_count.items():
        p = count / n_sims
        lo, hi = _wilson_ci(p, n_sims)
        results[team] = {"advance_probability": p, "ci_low": lo, "ci_high": hi}
    return {"n_simulations": n_sims, "seed": seed, "advance_per_group": advance_per_group, "teams": results}
