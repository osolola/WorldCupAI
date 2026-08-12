# World Cup AI: A Calibrated, Market-Benchmarked Soccer Forecasting System

A probabilistic soccer match/tournament forecasting system, built to be defensible in a technical
interview: two independently-derived models scored against naive baselines on a proper chronological
holdout, calibration measured (not assumed), Monte Carlo tournament simulation with confidence
intervals, and live market benchmarking (Polymarket + manual bookmaker entry) with de-vigging, edge
detection, closing-line value, and simulation-only Kelly staking.

It started as a World Cup bracket toy. It's now a general-purpose engine — the same rating/modeling
core runs on any competition; the World Cup is the current showcase, not a hardcoded assumption.

## Table of Contents

- [What's here](#whats-here)
- [How to run locally](#how-to-run-locally)
- [Methodology](#methodology)
  - [1. Rating engine](#1-rating-engine-walk-forward-no-lookahead)
  - [2. Two independent models](#2-two-independent-models)
  - [3. Evaluation backbone](#3-evaluation-backbone)
  - [4. Monte Carlo tournament simulation](#4-monte-carlo-tournament-simulation)
  - [5. Market benchmarking](#5-market-benchmarking)
- [Reproducing every number in this README](#reproducing-every-number-in-this-readme)
- [API reference](#api-reference)
- [Known limitations](#known-limitations-read-before-trusting-any-of-this)

## What's here

* **Rating engine:** walk-forward Elo (overall strength) plus a separate goal-based Attack/Defense
  Elo pair per team, computed match-by-match from 48,891 international results (1872–present) with
  zero lookahead.
* **Two swappable models**, scored against each other and three baselines:
  * **Dixon-Coles** — a bivariate-Poisson goals model with a low-score correlation correction, MLE-fit
    attack/defense strengths, an explicit home/neutral term, and exponential recency weighting.
  * **XGBoost + Elo/Attack-Defense** — the original 9-feature gradient-boosted classifier.
* **Evaluation backbone:** chronological holdout backtest (refit every year, never trains on a match
  before scoring it), Brier score, log loss, a calibration curve with Expected Calibration Error, and
  a "biggest misses" report — on purpose, not swept under the rug.
* **Monte Carlo tournament simulation:** thousands of simulated brackets/groups yield each team's
  advance/title odds with 95% confidence intervals, instead of one deterministic walk.
* **Market benchmarking:** live Polymarket integration (public API, no auth), manual bookmaker odds
  entry with mandatory capture timestamps, de-vigging via both simple normalization and Shin's method,
  edge detection, closing-line value, and fractional-Kelly stake sizing — simulation only, no real
  betting, no auto-execution, anywhere.
* **81 unit tests**, a `config.toml` with every tunable parameter and a fixed random seed, and a
  one-command script (`scripts/run_backtest.py`) that regenerates every metric in this document.

## Project Structure

```
backend/
  config.py                     # loads config.toml
  data_sources/                  # MatchResultSource interface + CSV implementation
  core/
    ratings.py                    # walk-forward Elo + Attack/Defense engine (competition-agnostic)
    elo_classifier.py              # XGBoost model
    dixon_coles.py                  # Dixon-Coles goals model (MLE fit, analytic gradient)
    neutral.py                       # neutral-site symmetrization for classifier-style models
  competitions/
    registry.py                       # config.toml -> Competition objects
    knockout.py                        # generic single-elimination bracket
  evaluation/
    metrics.py                          # Brier, log loss, calibration curve, ECE
    baselines.py                         # uniform / always-favorite / Elo-only baselines
    backtest.py                           # chronological holdout harness
    paper_trading.py                       # Kelly-staked paper trading, ROI + max drawdown
  simulation/
    monte_carlo.py                          # knockout bracket + group-stage Monte Carlo
  markets/
    devig.py                                 # multiplicative + Shin's method de-vig
    kelly.py                                  # fractional Kelly stake sizing
    clv.py                                     # edge + closing-line value
    manual_odds.py                              # append-only odds snapshot log (JSONL)
    polymarket.py                                # Gamma API client (public, no auth)
  ingestion.py                                    # append new results to results.csv, idempotently
  main.py                                          # FastAPI app
  tests/                                            # 81 tests
scripts/
  run_backtest.py    # regenerates reports/backtest_metrics.json + calibration_curve.png
  ingest_results.py   # CLI wrapper around backend/ingestion.py
frontend/
  src/
    api.js              # fetch wrapper for every backend endpoint
    App.jsx               # tab navigation
    components/            # Head-to-Head, Scoreline, Tournament Odds, Market Edge, Scorecard, ...
config.toml     # every rating/model/evaluation/market hyperparameter, one place
results.csv     # 48,891 international match results, 1872-present
data/           # odds_snapshots.jsonl (created on first manual odds entry)
reports/        # backtest_metrics.json + calibration_curve.png (generated, gitignored)
```

## How to Run Locally

Two processes: the FastAPI backend and the React frontend.

```bash
# Backend
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`). Override the API base with a
`VITE_API_BASE` environment variable if you deploy the backend elsewhere.

To regenerate every evaluation metric from scratch:

```bash
pip install -r requirements-dev.txt   # adds pytest
pytest                                 # 81 tests
python3 scripts/run_backtest.py        # writes reports/backtest_metrics.json + calibration_curve.png
```

## Methodology

### 1. Rating engine (walk-forward, no lookahead)

Every team's Elo rating (`backend/core/ratings.py`) is updated match-by-match in chronological order —
a team's rating *at* any given match reflects only matches strictly before it. This is what makes the
backtest in section 3 legitimate: the ratings columns attached to every historical row were never
computed with knowledge of the future, whether that row later ends up in the training set or the
holdout set.

A separate Attack/Defense Elo pair per team is updated the same way, but from goals: a team's Attack
rating rises when it outscores what its opponent's Defense rating would predict; Defense rises when it
concedes fewer goals than expected. The K-factor is config-driven per tournament type (World Cup
matches get more weight) rather than hardcoded — this is also what let the engine generalize beyond the
World Cup in Phase 1 without knowing what a "World Cup" is.

### 2. Two independent models

**Dixon-Coles** (`backend/core/dixon_coles.py`) fits, via maximum likelihood: an Attack and Defense
strength per team, a single home-advantage term (applied only when `neutral=False`, using the
`neutral` column already present in the data — unused until this project), and `rho`, the Dixon-Coles
correction for the well-documented failure of plain independent-Poisson models to reproduce the true
frequency of 0-0/1-0/0-1/1-1 scorelines. Matches are recency-weighted by exponential decay
(`xi`, default half-life ≈ 5 years — longer than club-football Dixon-Coles implementations typically
use, because international teams play far fewer matches per year and need a longer window to
accumulate signal). Attack/Defense are L2-regularized toward the league average, both for
identifiability (the parameters are only defined up to a shared shift without an anchor) and to shrink
noisy, sparse-data teams rather than let them run to extremes.

Engineering note, because it's a real result: the first working version used finite-difference
gradients and took 50–60 seconds per fit *without even converging* (500+ parameters × numerical
gradient ≈ 500+ extra function evaluations per optimizer step). I derived the analytic gradient
(verified against `scipy.optimize.approx_fprime` to ~1e-4 relative error) and it now fits all 48,891
matches in **~1.5 seconds**, fully converged.

**XGBoost + Elo/Attack-Defense** (`backend/core/elo_classifier.py`) is the original model: a 9-feature
gradient-boosted classifier (Elo, Elo diff, Attack, Defense ×2, and their cross-team diffs) predicting
{home win, away win, draw} directly, with no explicit goals model underneath.

Both models are scored under the identical protocol in section 3, so "which one is actually better"
is an answered question, not an assumption.

### 3. Evaluation backbone

`backend/evaluation/backtest.py` implements a chronological holdout: models are refit at every yearly
boundary using *only* data strictly before that boundary, then scored on matches within that year —
mirroring how a system retrained periodically in production would actually perform, and ensuring no
model, anywhere in this backtest, ever sees a result before being scored on predicting it.

Scored with:
- **Brier score** — mean squared error between the predicted probability vector and the one-hot
  outcome (0 = perfect, 2 = worst possible for 3 classes).
- **Log loss** — mean negative log-likelihood of the actual outcome.
- **Calibration curve + Expected Calibration Error (ECE)** — pooled one-vs-rest reliability diagram
  across all three outcome classes; ECE is the count-weighted average gap between predicted confidence
  and empirical frequency.

Against three baselines: **uniform** (1/3 every outcome), **always-favorite** (three fixed numbers —
the empirical favorite/underdog/draw win rates learned from training data, applied regardless of the
specific matchup), and **Elo-only** (single-feature logistic regression on Elo difference).

**Headline results** (holdout: 2016-01-01 onward, refit yearly, 9,477 matches, `config.toml` seed 42 —
regenerate with `python3 scripts/run_backtest.py`):

| Model               |     N | Brier  | Log loss | ECE    |
|----------------------|------:|-------:|---------:|-------:|
| Dixon-Coles          | 9,477 | 0.5137 |   0.8803 | 0.0090 |
| XGBoost + Elo         | 9,477 | 0.5141 |   0.8737 | 0.0093 |
| Elo-only baseline      | 9,477 | 0.5248 |   0.8908 | 0.0123 |
| Always-favorite baseline| 9,477 | 0.5734 |  0.9685 | 0.0243 |
| Uniform baseline         | 9,477 | 0.6667 |   1.0986 | 0.0000 |

Both real models clearly beat every baseline on every metric, and are well-calibrated (ECE < 0.01) —
when either says 70%, it happens close to 70% of the time (see `reports/calibration_curve.png` after
running the backtest). Dixon-Coles and XGBoost+Elo are within noise of each other despite being
structurally very different approaches (a generative goals model vs. a discriminative classifier),
which reads less like "one approach is right" and more like both are close to the ceiling the data
actually supports.

**Honesty, on purpose:** the report also surfaces each model's most confidently *wrong* predictions.
Dixon-Coles' ten biggest misses are *all* unofficial micro-national teams (Ticino, Raetia, Sint
Maarten, Saint Barthélemy — CONIFA-style entities with a handful of lifetime matches, not FIFA
members) — exactly where a data-driven rating should be most fragile. XGBoost+Elo's biggest misses
include genuine upsets among real national teams (Georgia over Spain, 2016; Andorra over Hungary,
2017), suggesting the classifier is somewhat less conservative on thin-data matchups than the
regularized Dixon-Coles fit. Full lists are in `reports/backtest_metrics.json` under each model's
`biggest_misses`, and rendered in the Scorecard tab of the app.

### 4. Monte Carlo tournament simulation

`backend/simulation/monte_carlo.py` replaces the single deterministic bracket walk with thousands of
simulated tournaments:

- **Knockout** (`simulate_knockout_bracket`): each match draws a categorical outcome
  (team_a_win/team_b_win/draw) from the model's probabilities; a drawn match is resolved via a
  simulated penalty shootout weighted by the two teams' relative in-regulation strength, not a blind
  coin flip. Reports each team's probability of reaching each round and winning it all, with a 95%
  confidence interval from simulation variance.
- **Group stage** (`simulate_group_stage`): full scorelines are sampled (via Dixon-Coles'
  `sample_score`, since group tiebreakers need goal difference, not just win/loss), points/GD/goals-for
  computed per standard football rules, and top-N advancement tallied.

Live example, run against the actual remaining 2026 World Cup bracket (Spain already through to the
final; England vs. Argentina semifinal pending) — 20,000 simulations, ~2.8 seconds:

| Team      | Title probability |
|-----------|-------------------:|
| Spain     | 51.4% |
| Argentina | 29.7% |
| England   | 19.0% |

### 5. Market benchmarking

**De-vig** (`backend/markets/devig.py`): both multiplicative normalization and **Shin's (1992) method**,
which models the bookmaker's overround as partly coming from a share `z` of informed money and
corrects the well-documented favorite-longshot bias — verified in tests to pull probability mass
*from* longshots *toward* favorites relative to naive normalization (and to correctly degenerate to
the multiplicative answer for a genuine 2-outcome market, where there's no extra information to use).

**Polymarket** (`backend/markets/polymarket.py`): a client for the public, no-auth Gamma API,
verified against the *live* 2026 World Cup market during development. Running the model's Monte Carlo
title odds against Polymarket's real prices for the same three remaining teams:

| Team      | Model | Polymarket (de-vigged) | Edge |
|-----------|------:|------------------------:|-----:|
| Spain     | 51.4% | 58.2% | −6.8pp |
| Argentina | 29.7% | 19.4% | **+10.3pp** |
| England   | 19.0% | 23.0% | −4.0pp |

Read as one data point, not proof of alpha — this is a single market snapshot, not a backtested
sample. That's exactly why closing-line value exists (below): a single correct-looking edge is cheap:
consistently beating the closing line is the actual signal.

**Manual bookmaker odds** (`backend/markets/manual_odds.py`): an append-only JSONL log
(`data/odds_snapshots.jsonl`) where every snapshot *must* record a `source` and a `captured_at`
timestamp — CLV is only legitimate if you know exactly when a price was observed relative to kickoff.
The Market Edge tab in the app writes to this log directly.

**Closing-line value** (`backend/markets/clv.py`): `CLV = closing_fair_prob[side] − entry_fair_prob[side]`.
Positive CLV means the market moved toward your position after you took it — the standard proxy for
genuine predictive edge, because it's assessable immediately and doesn't depend on the variance of any
single match's outcome.

**Kelly staking** (`backend/markets/kelly.py`, `backend/evaluation/paper_trading.py`): standard
`f* = (bp − q) / b` with a configurable fractional cap (`config.toml`, default 0.25× — full Kelly is
high-variance under any model uncertainty, and this model's edges are estimates, not ground truth).
The paper-trading engine tracks a simulated bankroll, ROI, **and max drawdown** — not ROI alone, since
a strategy that's up 40% with a 90% drawdown along the way is not the same thing as one that's up 40%
smoothly. **Simulation only, everywhere: nothing in this codebase places, sizes, or executes a real
bet.**

**Current sample size**, stated plainly: the World Cup only has 4 matches left (two semifinals, third
place, final), so any market-benchmark ROI/CLV numbers from this tournament alone are a proof that the
pipeline works end-to-end, not a statistically powered result. Sample size grows if this is pointed at
an ongoing league (see Phase 1's `MatchResultSource` extension point) after the tournament ends.

## Reproducing every number in this README

```bash
pytest -q                       # 81 tests, ~1.5s
python3 scripts/run_backtest.py  # writes reports/backtest_metrics.json, reports/calibration_curve.png
```

`config.toml` holds every hyperparameter used above (K-factors, Dixon-Coles `xi`/regularization/
optimizer budget, XGBoost hyperparameters, the holdout start date, calibration bin count, Kelly cap)
and a fixed `random_seed` — there is no hidden state anywhere else.

## API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/teams` | GET | Every team's current Elo/Attack/Defense rating |
| `/api/predict` | POST | XGBoost+Elo win/draw/loss for a matchup (neutral-site symmetrized) |
| `/api/simulate` | POST | Deterministic single-elimination bracket walk |
| `/api/scoreline` | GET | Dixon-Coles scoreline distribution, expected goals, top scorelines |
| `/api/tournament/monte-carlo` | POST | Monte Carlo bracket simulation with confidence intervals |
| `/api/market/edge` | POST | De-vig supplied odds, compare to model, size a Kelly stake |
| `/api/market/snapshot` | POST | Persist an odds observation (source + timestamp required) |
| `/api/market/snapshots/{match_id}` | GET | Retrieve logged snapshots for a match |
| `/api/scorecard` | GET | Latest backtest report (metrics + calibration + biggest misses) |

## Known Limitations (read before trusting any of this)

- **Dixon-Coles is refit periodically, not continuously.** In the backtest it's refit at each yearly
  boundary and evaluated statically for that whole year; it does not update within a period the way
  the Elo engine does match-by-match. This mirrors a realistic "retrain weekly/monthly" production
  pattern, but it's a real methodological choice, not free lunch.
- **Group-stage tiebreakers are simplified.** Points, then goal difference, then goals scored, in that order.
  Head-to-head record and disciplinary points (real FIFA tiebreakers) aren't modeled.
- **Market benchmark sample size is currently tiny** (see section 5) — treat any single-tournament
  ROI/CLV figure as a pipeline demonstration, not a statistically powered claim.
- **The historical dataset has quirks.** While building the ingestion pipeline, `results.csv` turned
  out to contain a handful of genuinely distinct matches (doubleheaders, same-day replays) that share
  identical date/home_team/away_team/tournament — an earlier version of the dedup logic silently
  deleted three of them before this was caught and fixed (`backend/tests/test_ingestion.py` now
  guards against it explicitly). Worth knowing if you extend the ingestion path further.
- **Penalty shootouts are modeled as a strength-weighted coin flip**, not a separately-fit model —
  there isn't enough shootout-specific data in this dataset to fit one honestly.

---
*Created by [osolola](https://github.com/osolola)*
