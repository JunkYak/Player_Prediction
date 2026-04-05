import PlayerCard from './PlayerCard';

// SECTION 2 — TOP PLAYERS (from frontend_home.json, no OUT players)
export default function TopPlayersSection({ players, onAdd, onDetail }) {
  // Filter out OUT players and sort by status_rank → predicted_rating desc
  const visible = players
    .filter(p => p.status !== 'Out')
    .sort((a, b) => {
      if (a.status_rank !== b.status_rank) return a.status_rank - b.status_rank;
      return (b.predicted_rating ?? 0) - (a.predicted_rating ?? 0);
    });

  return (
    <div>
      <div className="section-header">
        <div className="section-title">Top Players</div>
        <div className="section-subtitle">
          {visible.length} active players across today's games — sorted by availability &amp; predicted rating
        </div>
      </div>

      <hr className="section-divider" />

      {visible.length === 0 ? (
        <div className="empty-state">No player data available</div>
      ) : (
        <div className="players-grid">
          {visible.map(p => (
            <PlayerCard
              key={p.personId}
              player={p}
              onAdd={onAdd}
              onDetail={onDetail}
            />
          ))}
        </div>
      )}
    </div>
  );
}
