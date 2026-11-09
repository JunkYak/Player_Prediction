import { User } from 'lucide-react';
import StatusBadge from './StatusBadge';
import NbaLogo from './NbaLogo';
import LineupActionButton from './LineupActionButton';
import { getOpponentFromMatchups } from '../utils/schedule';

// Player card used across Draft Lineup, Top Players, and Teams views
export default function PlayerCard({
  player,
  onAdd,
  onDetail,
  showNotPlayed = false,
  slots = [],
  onToggleLineup,
  tomorrowSchedule = null,
}) {
  if (!player) return null;

  const rawRating = player.predicted_rating;
  const hasValidRating = typeof rawRating === 'number' && !isNaN(rawRating);
  const formattedRating = hasValidRating ? rawRating.toFixed(2) : 'N/A';
  const bonus = Number(player.season_performance_bonus) || 0;
  const hasBonus = bonus > 0;
  const rankScore = typeof player.ranking_score === 'number'
    ? player.ranking_score
    : (typeof player.adjusted_predicted_rating === 'number'
        ? player.adjusted_predicted_rating
        : (hasValidRating ? rawRating * (1 + bonus) : 0));
  const isOut = player.status === 'Out';
  const name = player.playerName || 'Unknown Player';
  const team = player.teamName || '—';
  const fallbackOpponent = tomorrowSchedule?.matchups ? getOpponentFromMatchups(player.teamId, tomorrowSchedule.matchups) : null;
  const opponent = (player.opponentName && player.opponentName !== '—' && player.opponentName !== '')
    ? player.opponentName
    : (fallbackOpponent || '—');

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
          <div className="player-rating-label">PREDICTED</div>
          <div className="player-rating-value">
            {showNotPlayed && !hasValidRating ? 'N/A' : formattedRating}
          </div>
          {hasBonus && (
            <div className="player-meta-badges" style={{ marginTop: 3 }}>
              <div
                className="player-bonus-badge"
                title={`Season Performance Multiplier: +${bonus.toFixed(1)}x added to Starting Five slot`}
                style={{ fontSize: '10px', color: '#10b981', fontWeight: 600 }}
              >
                +{bonus.toFixed(1)}x Bonus
              </div>
              <div
                className="player-rank-score-tag"
                title={`Top Players Ranking Score: ${rankScore.toFixed(2)} (Predicted ${formattedRating} × (1 + ${bonus.toFixed(1)}x)) • Used for Top Players sorting order`}
                style={{ fontSize: '9px', color: '#888', fontFamily: 'var(--font-mono)', marginTop: 1 }}
              >
                RANK SCORE <span style={{ color: '#bbb', fontWeight: 600 }}>{rankScore.toFixed(2)}</span>
              </div>
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
