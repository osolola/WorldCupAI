import { useEffect, useState } from "react";
import { getScorecard } from "../api";

const MODEL_LABELS = {
  dixon_coles: "Dixon-Coles",
  xgboost_elo: "XGBoost + Elo",
  elo_only: "Elo-only baseline",
  always_favorite: "Always-favorite baseline",
  uniform: "Uniform baseline",
};

export default function Scorecard() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getScorecard()
      .then(setReport)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const rows = report
    ? Object.entries(report.models).sort((a, b) => a[1].brier_score - b[1].brier_score)
    : [];

  return (
    <section className="panel">
      <h2>Backtest Scorecard</h2>
      <p className="panel-subtext">
        Chronological holdout backtest, refit every period, scored on matches the model never trained on.
        Includes the model's biggest misses on purpose &mdash; honesty over polish.
      </p>

      {loading && <p>Loading backtest report...</p>}
      {error && (
        <p className="error">
          {error}. Run <code>python3 scripts/run_backtest.py</code> from the project root to generate one.
        </p>
      )}

      {report && (
        <>
          <p className="panel-subtext">
            Holdout: {report.holdout_start_date} onward &middot; refit every period (
            {report.periods_evaluated.length} periods evaluated)
          </p>

          <table className="scoreline-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>N</th>
                <th>Brier</th>
                <th>Log loss</th>
                <th>ECE</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([name, m]) => (
                <tr key={name}>
                  <td>{MODEL_LABELS[name] || name}</td>
                  <td>{m.n_matches.toLocaleString()}</td>
                  <td>{m.brier_score.toFixed(4)}</td>
                  <td>{m.log_loss.toFixed(4)}</td>
                  <td>{m.expected_calibration_error.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {["dixon_coles", "xgboost_elo"].map((name) =>
            report.models[name]?.biggest_misses?.length ? (
              <div key={name} className="result-block">
                <h3>{MODEL_LABELS[name]}: Biggest Misses</h3>
                <table className="scoreline-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Match</th>
                      <th>Actual</th>
                      <th>Model's probability of what happened</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.models[name].biggest_misses.map((m, i) => (
                      <tr key={i}>
                        <td>{m.date}</td>
                        <td>
                          {m.team_a} vs {m.team_b}
                        </td>
                        <td>{m.outcome.replace("team_a_win", `${m.team_a} won`).replace("team_b_win", `${m.team_b} won`).replace("draw", "Draw")}</td>
                        <td>{(m.probability_of_actual * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null
          )}
        </>
      )}
    </section>
  );
}
