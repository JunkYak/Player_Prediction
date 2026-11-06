import { User } from 'lucide-react';
import StatusBadge from './StatusBadge';
import NbaLogo from './NbaLogo';
import LineupActionButton from './LineupActionButton';

// Player card used across Draft Lineup, Top Players, and Teams views
export default function PlayerCard({
  player,
  onAdd,
  onDetail,
  showNotPlayed = false,
  slots = [],
  onToggleLineup,
}) {
  if (!player) return null;

  const rawRating = player.predicted_rating;
  const hasValidRating = typeof rawRating === 'number' && !isNaN(rawRating);
  const formattedRating = hasValidRating ? rawRating.toFixed(2) : 'N/A';
  const bonus = Number(player.season_performance_bonus) || 0;
  const hasBonus = bonus > 0;
  const adjRating = typeof player.adjusted_predicted_rating === 'number'
    ? player.adjusted_predicted_rating
    : (hasValidRating ? rawRating * (1 + bonus) : 0);
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
            <NbaLogo
              teamId={player.teamId}
              size={13}
              style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 4 }}
            />
            <span>{team}</span>
            {opponent && opponent !== '—' && (
              <>
                <span className="matchup-vs"> vs </span>
                <span>{opponent}</span>
              </>
            )}
          </div>
          <StatusBadge status={player.status} />
        </div>
      </div>

      <div className="player-card-bottom">
        <div className="player-rating-block">
          <div className="player-rating-label">
            {hasBonus ? 'Adj (Base)' : 'Predicted'}
          </div>
          <div className="player-rating-value">
            {hasBonus ? (
              <span title={`Base Rating: ${formattedRating} • Season Bonus: +${bonus.toFixed(1)}x`}>
                {adjRating.toFixed(2)}{' '}
                <span style={{ fontSize: '11px', color: '#888', fontWeight: 400 }}>
                  ({formattedRating})
                </span>
              </span>
            ) : (
              showNotPlayed && !hasValidRating ? 'N/A' : formattedRating
            )}
          </div>
          {hasBonus && (
            <div
              className="player-bonus-badge"
              title={`Season Performance Multiplier: +${bonus.toFixed(1)}x`}
              style={{ fontSize: '10px', color: '#10b981', fontWeight: 600, marginTop: 2 }}
            >
              +{bonus.toFixed(1)}x Bonus
            </div>
          )}
        </div>

        <div className="player-card-actions">
          <LineupActionButton
            player={player}
            slots={slots}
            onToggle={onToggleLineup || onAdd}
          />
        </div>
      </div>
    </div>
  );
}
