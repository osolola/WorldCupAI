import { useMemo, useState } from "react";
import TeamCombobox from "./TeamCombobox";
import { simulateTournamentMonteCarlo } from "../api";
import { WORLD_CUP_2026_ROUND_OF_16 } from "../teams";

const MATCH_COUNT = 8;

export default function TournamentOdds({ teams }) {
  const names = useMemo(() => teams.map((t) => t.name), [teams]);

  const initial = useMemo(() => {
    const seeded = WORLD_CUP_2026_ROUND_OF_16.filter((name) => names.includes(name));
    const arr = [];
    for (let i = 0; i < MATCH_COUNT * 2; i++) {
      arr.push(seeded[i] || names[i % names.length] || "");
    }
    return arr;
  }, [names]);

  const [selection, setSelection] = useState(initial);
  const [nSims, setNSims] = useState(10000);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  function updateTeam(index, value) {
    setSelection((prev) => {
      const next = [...prev];
      next[index] = value;
      return next;
    });
  }

  const hasDuplicates = new Set(selection).size < selection.length;
  const allValid = selection.every((name) => names.includes(name));

  async function run() {
    setError(null);
    setLoading(true);
    try {
      setData(await simulateTournamentMonteCarlo(selection, Number(nSims), true));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const sortedTeams = data
    ? Object.entries(data.teams).sort((a, b) => b[1].champion.probability - a[1].champion.probability)
    : [];

  return (
    <section className="panel">
      <h2>Tournament Odds (Monte Carlo)</h2>
      <p className="panel-subtext">
        Simulates the bracket thousands of times against the Dixon-Coles model to produce title odds with
        confidence intervals, not a single deterministic walk.
      </p>

      <div className="custom-bracket-grid">
        {Array.from({ length: MATCH_COUNT }, (_, i) => (
          <div className="bracket-match-picker" key={i}>
            <h4>Match {i + 1}</h4>
            <TeamCombobox label="Team A" teams={teams} value={selection[i * 2]} onChange={(v) => updateTeam(i * 2, v)} />
            <TeamCombobox label="Team B" teams={teams} value={selection[i * 2 + 1]} onChange={(v) => updateTeam(i * 2 + 1, v)} />
          </div>
        ))}
      </div>

      <label className="combobox">
        <span className="combobox-label">Simulations</span>
        <input
          className="combobox-input"
          type="number"
          min="100"
          max="20000"
          step="100"
          value={nSims}
          onChange={(e) => setNSims(e.target.value)}
        />
      </label>

      {hasDuplicates && <p className="warning">Warning: duplicate teams selected.</p>}
      {!allValid && <p className="warning">Pick a valid team for every slot.</p>}

      <button onClick={run} disabled={loading || hasDuplicates || !allValid}>
        {loading ? "Simulating..." : "Run Monte Carlo"}
      </button>

      {error && <p className="error">{error}</p>}

      {data && (
        <div className="result-block">
          <h3>Title Odds ({data.n_simulations.toLocaleString()} simulations)</h3>
          <table className="scoreline-table">
            <thead>
              <tr>
                <th>Team</th>
                <th>Champion %</th>
                <th>95% CI</th>
              </tr>
            </thead>
            <tbody>
              {sortedTeams.map(([team, stats]) => (
                <tr key={team}>
                  <td>{team}</td>
                  <td>{(stats.champion.probability * 100).toFixed(1)}%</td>
                  <td>
                    {(stats.champion.ci_low * 100).toFixed(1)}% &ndash; {(stats.champion.ci_high * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
