import { useMemo, useState } from "react";
import TeamCombobox from "./TeamCombobox";
import RatingBadge from "./RatingBadge";
import ProbabilityBar from "./ProbabilityBar";
import { predictMatch } from "../api";

export default function HeadToHead({ teams }) {
  const names = useMemo(() => teams.map((t) => t.name), [teams]);
  const byName = useMemo(() => Object.fromEntries(teams.map((t) => [t.name, t])), [teams]);

  const [teamA, setTeamA] = useState(names.includes("Brazil") ? "Brazil" : names[0] || "");
  const [teamB, setTeamB] = useState(names.includes("France") ? "France" : names[1] || "");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const canPredict = byName[teamA] && byName[teamB] && teamA !== teamB;

  async function handlePredict() {
    setError(null);
    setLoading(true);
    try {
      setResult(await predictMatch(teamA, teamB));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  const verdict = (() => {
    if (!result) return null;
    const { team_a_win, team_b_win, draw } = result.probabilities;
    if (team_a_win > team_b_win && team_a_win > draw) return `${result.team_a.name} wins`;
    if (team_b_win > team_a_win && team_b_win > draw) return `${result.team_b.name} wins`;
    return "Draw likely";
  })();

  return (
    <section className="panel">
      <h2>AI Prediction (Elo-Based)</h2>
      <p className="panel-subtext">Neutral-site matchup — no home-field advantage is applied.</p>

      <div className="team-picker-row">
        <div>
          <TeamCombobox label="Team A" teams={teams} value={teamA} onChange={setTeamA} />
          <RatingBadge team={byName[teamA]} />
        </div>
        <div>
          <TeamCombobox label="Team B" teams={teams} value={teamB} onChange={setTeamB} />
          <RatingBadge team={byName[teamB]} />
        </div>
      </div>

      {teamA && teamB && teamA === teamB && <p className="warning">Pick two different teams.</p>}

      <button onClick={handlePredict} disabled={loading || !canPredict}>
        {loading ? "Predicting..." : "Predict Match"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result-block">
          <h3>Results</h3>
          <p className="result-headline">{verdict}</p>
          <ProbabilityBar label={`${result.team_a.name} Win`} value={result.probabilities.team_a_win} />
          <ProbabilityBar label="Draw" value={result.probabilities.draw} />
          <ProbabilityBar label={`${result.team_b.name} Win`} value={result.probabilities.team_b_win} />
        </div>
      )}
    </section>
  );
}
