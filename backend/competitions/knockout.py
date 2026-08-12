def _simulate_round(teams, predict_probs):
    matches = []
    next_round = []

    for i in range(0, len(teams), 2):
        t1, t2 = teams[i], teams[i + 1]
        probs = predict_probs(t1, t2)

        if probs["team_a_win"] >= probs["team_b_win"]:
            winner, win_prob = t1, probs["team_a_win"]
        else:
            winner, win_prob = t2, probs["team_b_win"]

        next_round.append(winner)
        matches.append({
            "team1": t1,
            "team2": t2,
            "winner": winner,
            "win_probability": win_prob,
            "probabilities": probs,
        })

    return matches, next_round


def run_bracket(teams, predict_probs):
    """
    Single-elimination bracket: teams are paired up in the order given
    (t[0] vs t[1], t[2] vs t[3], ...), the higher win-probability side
    advances each round, until one champion remains. predict_probs(a, b)
    must return {"team_a_win", "team_b_win", "draw"}.
    """
    rounds = []
    current = teams
    while len(current) > 1:
        matches, current = _simulate_round(current, predict_probs)
        rounds.append(matches)
    return rounds, current[0]
