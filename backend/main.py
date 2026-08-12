import json
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .competitions.knockout import run_bracket
from .competitions.registry import REPO_ROOT, get_competition
from .config import CONFIG
from .core.dixon_coles import fit_dixon_coles
from .core.elo_classifier import make_feature_row, train_model
from .core.neutral import symmetrize
from .core.ratings import compute_ratings
from .markets.clv import edge as market_edge
from .markets.clv import fair_probabilities
from .markets.kelly import kelly_fraction
from .markets.manual_odds import record_snapshot, snapshots_for_match
from .simulation.monte_carlo import simulate_knockout_bracket

# Phase 1: a single active competition. Adding a second (e.g. a weekly
# league) is a matter of registering it in config.toml and wiring a
# selector here — no changes needed to core/ or competitions/knockout.py.
ACTIVE_COMPETITION_KEY = "world_cup"
MAX_MONTE_CARLO_SIMS = 20000

app = FastAPI(title="World Cup AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

competition = get_competition(ACTIVE_COMPETITION_KEY)
raw_df = competition.data_source.load()

ratings_cfg = CONFIG["ratings"]
df, ratings = compute_ratings(
    raw_df,
    base_rating=ratings_cfg["base_rating"],
    elo_k_default=ratings_cfg["elo_k_default"],
    elo_k_overrides=competition.elo_k_overrides,
    attack_defense_k=ratings_cfg["attack_defense_k"],
    elo_divisor=ratings_cfg["elo_divisor"],
    goals_divisor=ratings_cfg["goals_divisor"],
    rating_diff_clip=ratings_cfg["rating_diff_clip"],
    expected_goals_clip=(ratings_cfg["expected_goals_clip_min"], ratings_cfg["expected_goals_clip_max"]),
)

model_cfg = CONFIG["model"]
model = train_model(
    df,
    n_estimators=model_cfg["n_estimators"],
    max_depth=model_cfg["max_depth"],
    learning_rate=model_cfg["learning_rate"],
    random_seed=model_cfg["random_seed"],
)

dc_cfg = CONFIG["dixon_coles"]
dixon_coles_model, _dc_fit_result = fit_dixon_coles(
    df,
    xi=dc_cfg["xi"],
    reg_lambda=dc_cfg["reg_lambda"],
    min_effective_weight=dc_cfg["min_effective_weight"],
    max_goals=dc_cfg["max_goals"],
    maxiter=dc_cfg["maxiter"],
    maxfun=dc_cfg["maxfun"],
)

TEAMS = sorted(ratings.keys())


class PredictRequest(BaseModel):
    team_a: str
    team_b: str


class SimulateRequest(BaseModel):
    teams: List[str]


class TournamentSimRequest(BaseModel):
    teams: List[str]
    neutral: bool = True
    n_sims: int = 10000


class MarketEdgeRequest(BaseModel):
    team_a: str
    team_b: str
    decimal_odds: Dict[str, float]
    neutral: bool = True


class RecordSnapshotRequest(BaseModel):
    match_id: str
    team_a: str
    team_b: str
    source: str
    decimal_odds: Dict[str, float]
    is_closing: bool = False


def team_rating(name):
    r = ratings.get(name, {"elo": 1500.0, "attack": 1500.0, "defense": 1500.0})
    return {
        "name": name,
        "elo": round(r["elo"]),
        "attack": round(r["attack"]),
        "defense": round(r["defense"]),
    }


def _raw_proba(team_a, team_b):
    return model.predict_proba(make_feature_row(ratings, team_a, team_b))[0]


def predict_probs(team_a, team_b):
    """World Cup matches are played at neutral sites; see core.neutral.symmetrize."""
    return symmetrize(_raw_proba, team_a, team_b)


def dixon_coles_predict_fn(team_a, team_b, neutral=True):
    return dixon_coles_model.match_probabilities(team_a, team_b, neutral=neutral)


def _require_known_teams(*teams):
    unknown = [t for t in teams if t not in ratings]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown teams: {unknown}")


def _require_decimal_odds_shape(decimal_odds):
    missing = {"team_a_win", "team_b_win", "draw"} - set(decimal_odds)
    if missing:
        raise HTTPException(status_code=400, detail=f"decimal_odds missing keys: {missing}")


@app.get("/api/teams")
def get_teams():
    return [team_rating(t) for t in TEAMS]


@app.post("/api/predict")
def predict(req: PredictRequest):
    _require_known_teams(req.team_a, req.team_b)

    return {
        "team_a": team_rating(req.team_a),
        "team_b": team_rating(req.team_b),
        "probabilities": predict_probs(req.team_a, req.team_b),
    }


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    teams = req.teams

    if len(teams) < 2 or (len(teams) & (len(teams) - 1)) != 0:
        raise HTTPException(status_code=400, detail="Number of teams must be a power of two (2, 4, 8, 16, ...)")
    _require_known_teams(*teams)

    rounds, champion = run_bracket(teams, predict_probs)
    return {"rounds": rounds, "champion": champion}


@app.get("/api/scoreline")
def scoreline(team_a: str, team_b: str, neutral: bool = True):
    """Dixon-Coles predicted scoreline distribution for a matchup -- W/D/L, expected goals, and the most likely exact scorelines."""
    _require_known_teams(team_a, team_b)

    xg_a, xg_b = dixon_coles_model.expected_goals(team_a, team_b, neutral)
    return {
        "team_a": team_rating(team_a),
        "team_b": team_rating(team_b),
        "neutral": neutral,
        "probabilities": dixon_coles_predict_fn(team_a, team_b, neutral),
        "expected_goals": {"team_a": xg_a, "team_b": xg_b},
        "top_scorelines": dixon_coles_model.top_scorelines(team_a, team_b, neutral, n=6),
    }


@app.post("/api/tournament/monte-carlo")
def tournament_monte_carlo(req: TournamentSimRequest):
    """
    Monte Carlo knockout simulation (Dixon-Coles powered): simulates the
    bracket thousands of times, drawing a categorical match outcome each
    time, to produce each team's probability of reaching each round and
    winning the whole thing, with 95% confidence intervals.
    """
    teams = req.teams
    if len(teams) < 2 or (len(teams) & (len(teams) - 1)) != 0:
        raise HTTPException(status_code=400, detail="Number of teams must be a power of two (2, 4, 8, 16, ...)")
    _require_known_teams(*teams)

    n_sims = min(max(req.n_sims, 100), MAX_MONTE_CARLO_SIMS)
    return simulate_knockout_bracket(teams, dixon_coles_predict_fn, n_sims=n_sims, neutral=req.neutral)


@app.post("/api/market/edge")
def market_edge_endpoint(req: MarketEdgeRequest):
    """
    De-vigs supplied decimal odds (Shin's method) and compares against the
    Dixon-Coles model's probabilities for the same matchup, returning the
    edge per outcome and a (simulation-only, fractional-capped) Kelly
    stake. Odds are supplied by the caller -- see /api/market/snapshot to
    persist one for later closing-line-value comparison.
    """
    _require_known_teams(req.team_a, req.team_b)
    _require_decimal_odds_shape(req.decimal_odds)

    model_probs = dixon_coles_predict_fn(req.team_a, req.team_b, req.neutral)
    fair, z = fair_probabilities(req.decimal_odds)
    kelly_cap = CONFIG["markets"]["kelly_fraction_cap"]

    return {
        "team_a": team_rating(req.team_a),
        "team_b": team_rating(req.team_b),
        "model_probabilities": model_probs,
        "market_fair_probabilities": fair,
        "market_z": z,
        "edge": market_edge(model_probs, req.decimal_odds),
        "kelly_fraction": {
            side: kelly_fraction(model_probs[side], req.decimal_odds[side], kelly_cap)
            for side in model_probs
        },
    }


@app.post("/api/market/snapshot")
def record_market_snapshot(req: RecordSnapshotRequest):
    """Persists one odds observation (bookmaker or Polymarket) with a capture timestamp, for later CLV comparison."""
    _require_decimal_odds_shape(req.decimal_odds)

    return record_snapshot(
        match_id=req.match_id,
        team_a=req.team_a,
        team_b=req.team_b,
        source=req.source,
        decimal_odds=req.decimal_odds,
        is_closing=req.is_closing,
    )


@app.get("/api/market/snapshots/{match_id}")
def get_market_snapshots(match_id: str):
    return snapshots_for_match(match_id)


@app.get("/api/scorecard")
def scorecard():
    """Serves the latest backtest report (see scripts/run_backtest.py) -- Brier/log loss/calibration per model, plus honest biggest-miss examples."""
    report_path = REPO_ROOT / "reports" / "backtest_metrics.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No backtest report found. Run scripts/run_backtest.py first.")
    with open(report_path) as f:
        return json.load(f)
