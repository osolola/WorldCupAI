export default function BracketResults({ rounds, champion }) {
  return (
    <div className="bracket-results">
      {rounds.map((matches, i) => (
        <div key={i} className="bracket-round">
          <h3>Round of {matches.length * 2}</h3>
          <div className="bracket-matches">
            {matches.map((m, j) => (
              <div className="bracket-match" key={j}>
                <div className="bracket-teams">
                  <span className={m.winner === m.team1 ? "winner" : ""}>{m.team1}</span>
                  <span className="vs">vs</span>
                  <span className={m.winner === m.team2 ? "winner" : ""}>{m.team2}</span>
                </div>
                <p className="bracket-winner">
                  Winner: {m.winner} ({Math.round(m.win_probability * 100)}%)
                </p>
                <p className="bracket-split">
                  {Math.round(m.probabilities.team_a_win * 100)}% / {Math.round(m.probabilities.draw * 100)}% /{" "}
                  {Math.round(m.probabilities.team_b_win * 100)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      ))}
      {champion && <p className="champion">Champion: {champion}</p>}
    </div>
  );
}
