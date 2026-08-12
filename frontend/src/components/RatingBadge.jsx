export default function RatingBadge({ team }) {
  if (!team) return null;

  return (
    <div className="rating-badge">
      <div className="rating-item">
        <span className="rating-label">Elo</span>
        <span className="rating-value">{team.elo}</span>
      </div>
      <div className="rating-item">
        <span className="rating-label">Attack</span>
        <span className="rating-value">{team.attack}</span>
      </div>
      <div className="rating-item">
        <span className="rating-label">Defense</span>
        <span className="rating-value">{team.defense}</span>
      </div>
    </div>
  );
}
