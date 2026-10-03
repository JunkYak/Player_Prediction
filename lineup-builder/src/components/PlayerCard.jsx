import { User, Plus } from 'lucide-react';
import StatusBadge from './StatusBadge';

// Player card used in both Top Players and Team view
export default function PlayerCard({ player, onAdd, onDetail, showNotPlayed = false }) {
  if (!player) return null;

  const rawRating = player.predicted_rating;
  const hasValidRating = typeof rawRating === 'number' && !isNaN(rawRating);
  const formattedRating = hasValidRating ? rawRating.toFixed(2) : 'N/A';
  const isOut = player.status === 'Out';
  const name = player.playerName || 'Unknown Player';
  const team = player.teamName || '—';
  const opponent = player.opponentName || '—';

  return (
    <div className={`player-card ${isOut ? 'out-player' : ''}`}>
      <div className="player-card-top">
        {/* Headshot */}
        <img
          className="player-headshot"
          src={player.headshot || ''}
          alt={name}
          onError={e => {
            e.target.style.display = 'none';
            if (e.target.nextSibling) e.target.nextSibling.style.display = 'flex';
          }}
        />
        <div className="player-headshot-fallback" style={{ display: 'none' }}>
          <User size={24} />
        </div>

        {/* Info */}
        <div className="player-info">
          <button className="player-name" onClick={() => onDetail && onDetail(player)}>
            {name}
          </button>
          <div className="player-matchup">
            {team} vs {opponent}
          </div>
          <StatusBadge status={player.status} />
        </div>
      </div>

      <div className="player-card-bottom">
        <div className="player-rating-block">
          <div className="player-rating-label">Predicted</div>
          <div className="player-rating-value">
            {showNotPlayed && !hasValidRating ? 'N/A' : formattedRating}
          </div>
        </div>

        <div className="player-card-actions">
          {!isOut && onAdd && (
            <button className="add-btn" onClick={() => onAdd(player)} title="Add to lineup">
              <Plus size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
