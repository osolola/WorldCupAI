const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }

  return res.json();
}

export function getTeams() {
  return request("/api/teams");
}

export function predictMatch(teamA, teamB) {
  return request("/api/predict", {
    method: "POST",
    body: JSON.stringify({ team_a: teamA, team_b: teamB }),
  });
}

export function simulateBracket(teams) {
  return request("/api/simulate", {
    method: "POST",
    body: JSON.stringify({ teams }),
  });
}

export function getScoreline(teamA, teamB, neutral = true) {
  const params = new URLSearchParams({ team_a: teamA, team_b: teamB, neutral });
  return request(`/api/scoreline?${params.toString()}`);
}

export function simulateTournamentMonteCarlo(teams, nSims = 10000, neutral = true) {
  return request("/api/tournament/monte-carlo", {
    method: "POST",
    body: JSON.stringify({ teams, n_sims: nSims, neutral }),
  });
}

export function getMarketEdge(teamA, teamB, decimalOdds, neutral = true) {
  return request("/api/market/edge", {
    method: "POST",
    body: JSON.stringify({ team_a: teamA, team_b: teamB, decimal_odds: decimalOdds, neutral }),
  });
}

export function getScorecard() {
  return request("/api/scorecard");
}

export function recordMarketSnapshot(matchId, teamA, teamB, source, decimalOdds, isClosing = false) {
  return request("/api/market/snapshot", {
    method: "POST",
    body: JSON.stringify({
      match_id: matchId,
      team_a: teamA,
      team_b: teamB,
      source,
      decimal_odds: decimalOdds,
      is_closing: isClosing,
    }),
  });
}
