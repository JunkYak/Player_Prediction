import { User, Plus } from 'lucide-react';
import StatusBadge from './StatusBadge';

// Player card used in both Top Players and Team view
export default function PlayerCard({ player, onAdd, onDetail, showNotPlayed = false }) {
  const rating = player.predicted_rating;
  const isOut = player.status === 'Out';

  return (
    <div className={`player-card ${isOut ? 'out-player' : ''}`}>
      <div className="player-card-top">
        {/* Headshot */}
        <img
          className="player-headshot"
          src={player.headshot}
          alt={player.playerName}
          onError={e => {
            e.target.style.display = 'none';
            e.target.nextSibling.style.display = 'flex';
          }}
        />
        <div className="player-headshot-fallback" style={{ display: 'none' }}>
          <User size={24} />
        </div>

        {/* Info */}
        <div className="player-info">
          <button className="player-name" onClick={() => onDetail && onDetail(player)}>
            {player.playerName}
          </button>
          <div className="player-matchup">
            {player.teamName} vs {player.opponentName || '—'}
          </div>
          <StatusBadge status={player.status} />
        </div>
      </div>

      <div className="player-card-bottom">
        <div className="player-rating-block">
          <div className="player-rating-label">Predicted</div>
          <div className="player-rating-value">
            {showNotPlayed && !rating ? 'N/A' : rating?.toFixed(2) ?? 'N/A'}
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
