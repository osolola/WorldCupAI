import { useMemo, useState } from "react";
import TeamCombobox from "./TeamCombobox";
import BracketResults from "./BracketResults";
import { simulateBracket } from "../api";
import { WORLD_CUP_2026_ROUND_OF_16 } from "../teams";

const MATCH_COUNT = 8;

export default function CustomBracket({ teams }) {
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
      setData(await simulateBracket(selection));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <h2>Build Your Own Bracket</h2>
      <p className="panel-subtext">Defaults to the 2026 World Cup Round of 16 — edit any matchup.</p>

      <div className="custom-bracket-grid">
        {Array.from({ length: MATCH_COUNT }, (_, i) => (
          <div className="bracket-match-picker" key={i}>
            <h4>Match {i + 1}</h4>
            <TeamCombobox
              label="Team A"
              teams={teams}
              value={selection[i * 2]}
              onChange={(v) => updateTeam(i * 2, v)}
            />
            <TeamCombobox
              label="Team B"
              teams={teams}
              value={selection[i * 2 + 1]}
              onChange={(v) => updateTeam(i * 2 + 1, v)}
            />
          </div>
        ))}
      </div>

      {hasDuplicates && <p className="warning">Warning: duplicate teams selected.</p>}
      {!allValid && <p className="warning">Pick a valid team for every slot.</p>}

      <button onClick={run} disabled={loading || hasDuplicates || !allValid}>
        {loading ? "Simulating..." : "Run Custom Simulation"}
      </button>

      {error && <p className="error">{error}</p>}
      {data && <BracketResults rounds={data.rounds} champion={data.champion} />}
    </section>
  );
}
