import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

DEFAULT_MAX_GOALS = 10


def _tau_matrix(rho):
    """
    Dixon-Coles' low-score correlation correction. Vanilla Poisson assumes
    home and away goals are independent, which underweights 0-0/1-0/0-1/1-1
    relative to what actually happens; tau corrects those four cells only.
    Returns a function tau(x, y, lam_home, lam_away) -> multiplier.
    """
    def tau(x, y, lam_home, lam_away):
        if x == 0 and y == 0:
            return 1 - lam_home * lam_away * rho
        if x == 0 and y == 1:
            return 1 + lam_home * rho
        if x == 1 and y == 0:
            return 1 + lam_away * rho
        if x == 1 and y == 1:
            return 1 - rho
        return 1.0
    return tau


class DixonColesModel:
    """
    A fitted Dixon-Coles goals model: each team has an Attack and Defense
    strength (log-rate deviations from league average), there's a single
    home-advantage term applied only to non-neutral matches, and rho
    corrects the low-score independence assumption of plain Poisson.

    Teams not seen during fitting default to league-average strength
    (attack=defense=0) — a neutral prior rather than an error.
    """

    def __init__(self, mu, home_adv, rho, attack, defense, max_goals=DEFAULT_MAX_GOALS):
        self.mu = mu
        self.home_adv = home_adv
        self.rho = rho
        self.attack = attack
        self.defense = defense
        self.max_goals = max_goals
        self._tau = _tau_matrix(rho)

    def expected_goals(self, team_a, team_b, neutral=True):
        atk_a = self.attack.get(team_a, 0.0)
        def_a = self.defense.get(team_a, 0.0)
        atk_b = self.attack.get(team_b, 0.0)
        def_b = self.defense.get(team_b, 0.0)
        home_bonus = 0.0 if neutral else self.home_adv

        lam_a = float(np.exp(self.mu + atk_a - def_b + home_bonus))
        lam_b = float(np.exp(self.mu + atk_b - def_a))
        return lam_a, lam_b

    def scoreline_matrix(self, team_a, team_b, neutral=True):
        """grid[i, j] = P(team_a scores i, team_b scores j), i,j in [0, max_goals]."""
        lam_a, lam_b = self.expected_goals(team_a, team_b, neutral)
        goals = np.arange(0, self.max_goals + 1)
        pmf_a = poisson.pmf(goals, lam_a)
        pmf_b = poisson.pmf(goals, lam_b)
        grid = np.outer(pmf_a, pmf_b)

        for x in (0, 1):
            for y in (0, 1):
                grid[x, y] *= self._tau(x, y, lam_a, lam_b)

        grid = np.clip(grid, 0, None)
        grid /= grid.sum()
        return grid

    def match_probabilities(self, team_a, team_b, neutral=True):
        grid = self.scoreline_matrix(team_a, team_b, neutral)
        return {
            "team_a_win": float(np.tril(grid, -1).sum()),
            "team_b_win": float(np.triu(grid, 1).sum()),
            "draw": float(np.trace(grid)),
        }

    def top_scorelines(self, team_a, team_b, neutral=True, n=5):
        grid = self.scoreline_matrix(team_a, team_b, neutral)
        flat_idx = np.argsort(grid, axis=None)[::-1][:n]
        rows, cols = np.unravel_index(flat_idx, grid.shape)
        return [
            {"team_a_goals": int(i), "team_b_goals": int(j), "probability": float(grid[i, j])}
            for i, j in zip(rows, cols)
        ]

    def sample_score(self, team_a, team_b, neutral, rng):
        """Draws one (team_a_goals, team_b_goals) outcome from the fitted joint distribution."""
        grid = self.scoreline_matrix(team_a, team_b, neutral)
        flat = grid.flatten()
        idx = rng.choice(len(flat), p=flat)
        return divmod(int(idx), grid.shape[1])


def fit_dixon_coles(
    df,
    xi=0.00038,
    reg_lambda=0.01,
    min_effective_weight=5.0,
    max_goals=DEFAULT_MAX_GOALS,
    as_of=None,
    maxiter=2000,
    maxfun=20000,
):
    """
    Fits Attack/Defense strength per team, a global home-advantage term
    (applied only to non-neutral matches, using the `neutral` column), and
    the Dixon-Coles low-score correlation parameter rho, by maximizing a
    recency-weighted log-likelihood (exponential decay, half-life =
    ln(2)/xi days — default ~5 years, chosen because international teams
    play far fewer matches/year than club sides and need a longer window
    to accumulate signal).

    Teams whose total recency-weight falls below `min_effective_weight`
    are excluded from the fit (too little signal to estimate reliably) and
    default to league-average strength (0) at prediction time. Attack and
    defense are L2-regularized toward 0 (the league-average team) both for
    identifiability (Dixon-Coles parameters are only defined up to a
    shared shift without some anchor) and to shrink noisy, sparse-data
    teams toward the average rather than letting them run to extremes.
    """
    as_of = as_of or df["date"].max()
    days_ago = (as_of - df["date"]).dt.days.clip(lower=0)
    weight = np.exp(-xi * days_ago)

    team_weight = (
        pd.concat([
            pd.Series(weight.to_numpy(), index=df["home_team"].to_numpy()),
            pd.Series(weight.to_numpy(), index=df["away_team"].to_numpy()),
        ])
        .groupby(level=0)
        .sum()
    )
    fit_teams = sorted(team_weight[team_weight >= min_effective_weight].index)
    team_index = {t: i for i, t in enumerate(fit_teams)}
    n = len(fit_teams)

    in_fit_set = df["home_team"].isin(team_index) & df["away_team"].isin(team_index)
    fit_df = df[in_fit_set].reset_index(drop=True)
    fit_weight = weight[in_fit_set].to_numpy()

    home_idx = fit_df["home_team"].map(team_index).to_numpy()
    away_idx = fit_df["away_team"].map(team_index).to_numpy()
    home_goals = fit_df["home_score"].to_numpy(dtype=float)
    away_goals = fit_df["away_score"].to_numpy(dtype=float)
    neutral = fit_df["neutral"].to_numpy(dtype=bool) if "neutral" in fit_df else np.zeros(len(fit_df), dtype=bool)

    mask00 = (home_goals == 0) & (away_goals == 0)
    mask01 = (home_goals == 0) & (away_goals == 1)
    mask10 = (home_goals == 1) & (away_goals == 0)
    mask11 = (home_goals == 1) & (away_goals == 1)

    def unpack(params):
        mu, home_adv, rho = params[0], params[1], params[2]
        attack = params[3:3 + n]
        defense = params[3 + n:3 + 2 * n]
        return mu, home_adv, rho, attack, defense

    not_neutral = ~neutral

    def neg_log_likelihood_and_grad(params):
        mu, home_adv, rho, attack, defense = unpack(params)
        home_bonus = np.where(neutral, 0.0, home_adv)

        lam_home = np.clip(np.exp(mu + attack[home_idx] - defense[away_idx] + home_bonus), 1e-6, 50)
        lam_away = np.clip(np.exp(mu + attack[away_idx] - defense[home_idx]), 1e-6, 50)

        log_pmf = poisson.logpmf(home_goals, lam_home) + poisson.logpmf(away_goals, lam_away)

        tau = np.ones(len(fit_df))
        tau_grad_zh = np.zeros(len(fit_df))
        tau_grad_za = np.zeros(len(fit_df))
        tau_grad_rho = np.zeros(len(fit_df))

        d00 = 1 - lam_home[mask00] * lam_away[mask00] * rho
        tau[mask00] = d00
        tau_grad_zh[mask00] = -lam_home[mask00] * lam_away[mask00] * rho / d00
        tau_grad_za[mask00] = -lam_home[mask00] * lam_away[mask00] * rho / d00
        tau_grad_rho[mask00] = -lam_home[mask00] * lam_away[mask00] / d00

        d01 = 1 + lam_home[mask01] * rho
        tau[mask01] = d01
        tau_grad_zh[mask01] = lam_home[mask01] * rho / d01
        tau_grad_rho[mask01] = lam_home[mask01] / d01

        d10 = 1 + lam_away[mask10] * rho
        tau[mask10] = d10
        tau_grad_za[mask10] = lam_away[mask10] * rho / d10
        tau_grad_rho[mask10] = lam_away[mask10] / d10

        tau[mask11] = 1 - rho
        tau_grad_rho[mask11] = -1 / (1 - rho)

        tau = np.clip(tau, 1e-6, None)

        weighted_log_lik = fit_weight * (log_pmf + np.log(tau))
        nll = -np.sum(weighted_log_lik) + reg_lambda * (np.sum(attack ** 2) + np.sum(defense ** 2))

        g_h = -fit_weight * ((home_goals - lam_home) + tau_grad_zh)
        g_a = -fit_weight * ((away_goals - lam_away) + tau_grad_za)
        g_rho = -np.sum(fit_weight * tau_grad_rho)

        grad = np.zeros_like(params)
        grad[0] = np.sum(g_h) + np.sum(g_a)
        grad[1] = np.sum(g_h[not_neutral])
        grad[2] = g_rho

        grad_attack = np.zeros(n)
        grad_defense = np.zeros(n)
        np.add.at(grad_attack, home_idx, g_h)
        np.add.at(grad_attack, away_idx, g_a)
        np.add.at(grad_defense, away_idx, -g_h)
        np.add.at(grad_defense, home_idx, -g_a)

        grad_attack += 2 * reg_lambda * attack
        grad_defense += 2 * reg_lambda * defense

        grad[3:3 + n] = grad_attack
        grad[3 + n:3 + 2 * n] = grad_defense

        return nll, grad

    x0 = np.zeros(3 + 2 * n)
    x0[1] = 0.2  # home_adv initial guess

    result = minimize(
        neg_log_likelihood_and_grad,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": maxfun},
    )
    mu, home_adv, rho, attack, defense = unpack(result.x)

    model = DixonColesModel(
        mu=float(mu),
        home_adv=float(home_adv),
        rho=float(rho),
        attack={t: float(attack[i]) for t, i in team_index.items()},
        defense={t: float(defense[i]) for t, i in team_index.items()},
        max_goals=max_goals,
    )
    return model, result
