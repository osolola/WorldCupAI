export default function ProbabilityBar({ label, value }) {
  const pct = Math.round(value * 100);

  return (
    <div className="prob-bar">
      <div className="prob-bar-label">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="prob-bar-track">
        <div className="prob-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
