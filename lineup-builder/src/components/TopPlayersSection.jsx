import PlayerCard from './PlayerCard';

// SECTION 2 — TOP PLAYERS (from GET /api/v1/players, no OUT players)
export default function TopPlayersSection({ players, onAdd, onDetail }) {
  // Filter out OUT players and sort by status_rank → predicted_rating desc
  const visible = (Array.isArray(players) ? players : [])
    .filter(p => p && p.status !== 'Out')
    .sort((a, b) => {
      const aRank = typeof a.status_rank === 'number' ? a.status_rank : 5;
      const bRank = typeof b.status_rank === 'number' ? b.status_rank : 5;
      if (aRank !== bRank) return aRank - bRank;
      const aRating = typeof a.predicted_rating === 'number' ? a.predicted_rating : 0;
      const bRating = typeof b.predicted_rating === 'number' ? b.predicted_rating : 0;
      return bRating - aRating;
    });

  const noGames = !Array.isArray(players) || players.length === 0;

  return (
    <div>
      <div className="section-header">
        <div className="section-title">Top Players</div>
        <div className="section-subtitle">
          {noGames
            ? "No NBA games scheduled for tomorrow"
            : `${visible.length} active players across tomorrow's games — sorted by availability & predicted rating`}
        </div>
      </div>

      <hr className="section-divider" />

      {noGames ? (
        <div className="empty-state" style={{ padding: '60px 20px', textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)', letterSpacing: '0.06em' }}>
            No NBA games scheduled for tomorrow.
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#555', marginTop: 8 }}>
            The draft slate will activate automatically when upcoming matchups are scheduled.
          </div>
        </div>
      ) : visible.length === 0 ? (
        <div className="empty-state">No active players available for this slate</div>
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
