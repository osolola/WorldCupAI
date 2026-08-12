import { useMemo, useState } from "react";
import TeamCombobox from "./TeamCombobox";
import RatingBadge from "./RatingBadge";
import ProbabilityBar from "./ProbabilityBar";
import { getScoreline } from "../api";

export default function Scoreline({ teams }) {
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
      setResult(await getScoreline(teamA, teamB, true));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <h2>Predicted Scoreline (Dixon-Coles)</h2>
      <p className="panel-subtext">
        A goals model, not a win/loss classifier: estimates each team's expected goals and the full
        scoreline probability distribution, with a low-score correlation correction.
      </p>

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

      <button onClick={handlePredict} disabled={loading || !canPredict}>
        {loading ? "Predicting..." : "Predict Scoreline"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result-block">
          <h3>Expected Goals</h3>
          <p className="expected-goals-line">
            {result.team_a.name} <strong>{result.expected_goals.team_a.toFixed(2)}</strong> &ndash;{" "}
            <strong>{result.expected_goals.team_b.toFixed(2)}</strong> {result.team_b.name}
          </p>

          <ProbabilityBar label={`${result.team_a.name} Win`} value={result.probabilities.team_a_win} />
          <ProbabilityBar label="Draw" value={result.probabilities.draw} />
          <ProbabilityBar label={`${result.team_b.name} Win`} value={result.probabilities.team_b_win} />

          <h3 className="scoreline-heading">Most Likely Scorelines</h3>
          <table className="scoreline-table">
            <thead>
              <tr>
                <th>Score</th>
                <th>Probability</th>
              </tr>
            </thead>
            <tbody>
              {result.top_scorelines.map((s, i) => (
                <tr key={i}>
                  <td>
                    {result.team_a.name} {s.team_a_goals} &ndash; {s.team_b_goals} {result.team_b.name}
                  </td>
                  <td>{(s.probability * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
