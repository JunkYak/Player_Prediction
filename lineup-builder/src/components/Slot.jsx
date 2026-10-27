import { User, X, Plus, Info } from 'lucide-react';
import NbaLogo from './NbaLogo';

// A single slot in the Draft lineup bar
export default function Slot({
  player,
  index,
  multiplier = 1.0,
  projectedValue = 0,
  onClick,
  onRemove,
  onDetail,
}) {
  const slotNum = index + 1;
  const multiplierText = `${multiplier.toFixed(1)}x`;

  return (
    <div
      className={`lineup-slot ${player ? 'is-filled' : 'is-empty'}`}
      onClick={onClick}
      title={
        player
          ? `${player.playerName} (Slot ${slotNum} • ${multiplierText}) — Click to replace`
          : `Slot ${slotNum} • ${multiplierText} — Click to assign`
      }
    >
      {/* Slot Header with Multiplier Badge and Info Button */}
      <div className="slot-header-tag">
        <span className="slot-number">SLOT {slotNum}</span>
        <span className="slot-multiplier-badge">{multiplierText}</span>

        {player && onDetail && (
          <button
            type="button"
            className="slot-info-btn"
            onClick={e => {
              e.stopPropagation();
              onDetail(player);
            }}
            title={`View ${player.playerName} Intel`}
            aria-label={`View ${player.playerName} Intel`}
          >
            <Info size={11} />
          </button>
        )}
      </div>

      <div className={`slot-circle ${player ? 'filled' : ''}`}>
        {player ? (
          <>
            <img
              src={player.headshot}
              alt={player.playerName}
              onError={e => {
                e.target.style.display = 'none';
                if (e.target.nextSibling) e.target.nextSibling.style.display = 'flex';
              }}
            />
            <div
              style={{
                display: 'none',
                width: '100%',
                height: '100%',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#1a1a1a',
              }}
            >
              <User size={32} style={{ color: '#444' }} />
            </div>
            {onRemove && (
              <button
                type="button"
                className="slot-remove-btn"
                onClick={e => {
                  e.stopPropagation();
                  onRemove();
                }}
                title={`Remove ${player.playerName} from Slot ${slotNum}`}
                aria-label={`Remove ${player.playerName} from Slot ${slotNum}`}
              >
                <X size={12} />
              </button>
            )}
          </>
        ) : (
          <div className="slot-empty-content">
            <Plus size={24} className="slot-empty-icon" />
          </div>
        )}
      </div>

      <div className="slot-label">
        {player ? (
          <>
            <button
              type="button"
              className="slot-name slot-name-interactive"
              onClick={e => {
                if (onDetail) {
                  e.stopPropagation();
                  onDetail(player);
                }
              }}
              title="Click to view player dossier"
            >
              {player.playerName || 'Unknown Player'}
            </button>

            <div className="slot-team-sub">
              <NbaLogo
                teamId={player.teamId}
                size={11}
                style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 4 }}
              />
              <span>{player.teamName || 'NBA'}</span>
            </div>

            <div className="slot-score-breakdown">
              <div className="slot-base-rating" title="Base Predicted Rating">
                <span className="label">Base</span>
                <span className="val">
                  {typeof player.predicted_rating === 'number' && !isNaN(player.predicted_rating)
                    ? player.predicted_rating.toFixed(2)
                    : '—'}
                </span>
              </div>
              <span className="slot-score-math">× {multiplierText} =</span>
              <div className="slot-projected-val" title="Projected Slot Score">
                <span className="label">Proj</span>
                <span className="val">
                  {typeof projectedValue === 'number' && !isNaN(projectedValue)
                    ? projectedValue.toFixed(2)
                    : '0.00'}
                </span>
              </div>
            </div>
          </>
        ) : (
          <div className="slot-empty-label">
            <span className="slot-empty-text">Empty</span>
            <span className="slot-empty-action">+ Pick Player</span>
          </div>
        )}
      </div>
    </div>
  );
}
