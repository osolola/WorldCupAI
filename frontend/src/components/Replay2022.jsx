import { useState } from "react";
import { simulateBracket } from "../api";
import BracketResults from "./BracketResults";
import { WORLD_CUP_2022_ROUND_OF_16 } from "../teams";

export default function Replay2022() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setError(null);
    setLoading(true);
    try {
      setData(await simulateBracket(WORLD_CUP_2022_ROUND_OF_16));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <h2>2022 World Cup Replay</h2>
      <p>Re-simulate the 2022 knockout stage with the current model.</p>
      <button onClick={run} disabled={loading}>
        {loading ? "Simulating..." : "Run 2022 Simulation"}
      </button>
      {error && <p className="error">{error}</p>}
      {data && <BracketResults rounds={data.rounds} champion={data.champion} />}
    </section>
  );
}
