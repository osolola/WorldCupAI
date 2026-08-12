import { useMemo, useState } from "react";
import TeamCombobox from "./TeamCombobox";
import { getMarketEdge, recordMarketSnapshot } from "../api";

export default function MarketEdge({ teams }) {
  const names = useMemo(() => teams.map((t) => t.name), [teams]);
  const byName = useMemo(() => Object.fromEntries(teams.map((t) => [t.name, t])), [teams]);

  const [teamA, setTeamA] = useState(names[0] || "");
  const [teamB, setTeamB] = useState(names[1] || "");
  const [oddsA, setOddsA] = useState("2.60");
  const [oddsB, setOddsB] = useState("3.10");
  const [oddsDraw, setOddsDraw] = useState("3.30");
  const [source, setSource] = useState("manual");
  const [matchId, setMatchId] = useState("");

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);

  const canSubmit = byName[teamA] && byName[teamB] && teamA !== teamB && oddsA && oddsB && oddsDraw;

  function decimalOdds() {
    return {
      team_a_win: parseFloat(oddsA),
      team_b_win: parseFloat(oddsB),
      draw: parseFloat(oddsDraw),
    };
  }

  async function handleCompute() {
    setError(null);
    setSaveStatus(null);
    setLoading(true);
    try {
      setResult(await getMarketEdge(teamA, teamB, decimalOdds(), true));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleSaveSnapshot(isClosing) {
    setSaveStatus(null);
    const id = matchId.trim() || `${teamA}-vs-${teamB}`.toLowerCase().replace(/\s+/g, "-");
    try {
      await recordMarketSnapshot(id, teamA, teamB, source || "manual", decimalOdds(), isClosing);
      setSaveStatus(`Saved ${isClosing ? "closing" : "entry"} snapshot for "${id}".`);
    } catch (err) {
      setSaveStatus(`Failed to save: ${err.message}`);
    }
  }

  return (
    <section className="panel">
      <h2>Market Edge</h2>
      <p className="panel-subtext">
        Paste decimal odds for a match (bookmaker or Polymarket, converted to decimal odds via 1/price). De-vigs
        with Shin's method, compares to the Dixon-Coles model, and sizes a simulation-only fractional-Kelly stake.
      </p>

      <div className="team-picker-row">
        <TeamCombobox label="Team A" teams={teams} value={teamA} onChange={setTeamA} />
        <TeamCombobox label="Team B" teams={teams} value={teamB} onChange={setTeamB} />
      </div>

      <div className="odds-input-row">
        <label className="combobox">
          <span className="combobox-label">Team A Win (decimal odds)</span>
          <input className="combobox-input" type="number" step="0.01" min="1.01" value={oddsA} onChange={(e) => setOddsA(e.target.value)} />
        </label>
        <label className="combobox">
          <span className="combobox-label">Draw (decimal odds)</span>
          <input className="combobox-input" type="number" step="0.01" min="1.01" value={oddsDraw} onChange={(e) => setOddsDraw(e.target.value)} />
        </label>
        <label className="combobox">
          <span className="combobox-label">Team B Win (decimal odds)</span>
          <input className="combobox-input" type="number" step="0.01" min="1.01" value={oddsB} onChange={(e) => setOddsB(e.target.value)} />
        </label>
      </div>

      <button onClick={handleCompute} disabled={loading || !canSubmit}>
        {loading ? "Computing..." : "Compute Edge"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result-block">
          <table className="scoreline-table">
            <thead>
              <tr>
                <th>Outcome</th>
                <th>Model</th>
                <th>Market (fair)</th>
                <th>Edge</th>
                <th>Kelly stake</th>
              </tr>
            </thead>
            <tbody>
              {["team_a_win", "draw", "team_b_win"].map((side) => (
                <tr key={side}>
                  <td>{side === "team_a_win" ? result.team_a.name : side === "team_b_win" ? result.team_b.name : "Draw"}</td>
                  <td>{(result.model_probabilities[side] * 100).toFixed(1)}%</td>
                  <td>{(result.market_fair_probabilities[side] * 100).toFixed(1)}%</td>
                  <td className={result.edge[side] > 0 ? "edge-positive" : "edge-negative"}>
                    {result.edge[side] > 0 ? "+" : ""}
                    {(result.edge[side] * 100).toFixed(1)}%
                  </td>
                  <td>{(result.kelly_fraction[side] * 100).toFixed(1)}% of bankroll</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="panel-subtext">Shin's z (estimated informed-money share): {(result.market_z * 100).toFixed(2)}%</p>

          <div className="snapshot-save-row">
            <input
              className="combobox-input"
              type="text"
              placeholder="match id (optional)"
              value={matchId}
              onChange={(e) => setMatchId(e.target.value)}
            />
            <input
              className="combobox-input"
              type="text"
              placeholder="source (e.g. bookmaker, polymarket)"
              value={source}
              onChange={(e) => setSource(e.target.value)}
            />
            <button onClick={() => handleSaveSnapshot(false)}>Save Entry Snapshot</button>
            <button onClick={() => handleSaveSnapshot(true)}>Save Closing Snapshot</button>
          </div>
          {saveStatus && <p className="panel-subtext">{saveStatus}</p>}
        </div>
      )}
    </section>
  );
}
