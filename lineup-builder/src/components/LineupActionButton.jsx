import { useState } from 'react';
import { Plus, Check, X } from 'lucide-react';

/**
 * LineupActionButton
 *
 * Universal button for adding / removing a player from the Starting Five.
 * Designed to fit seamlessly on player cards, table rows, mover rows, and modal dialogs.
 *
 * States:
 * - Default: "+ Lineup" (with Plus icon)
 * - Added: "✓ Added" (with Check icon, toggles to "✕ Remove" on hover)
 * - Full: "Full (5/5)" (disabled or click explains lineup is full)
 * - Inactive / Out: "Out" (disabled)
 */
export default function LineupActionButton({
  player,
  slots = [],
  onToggle,
  size = 'normal', // 'normal' | 'compact' | 'mini'
}) {
  const [isHovered, setIsHovered] = useState(false);

  if (!player) return null;

  const isOut = player.status === 'Out';
  const isInLineup = Array.isArray(slots) && slots.some(s => s && s.personId === player.personId);
  const isLineupFull = Array.isArray(slots) && slots.filter(Boolean).length >= 5;

  function handleClick(e) {
    e.stopPropagation();
    if (isOut) return;
    if (onToggle) {
      onToggle(player);
    }
  }

  if (isOut) {
    return (
      <button
        type="button"
        className={`lineup-action-btn size-${size} is-out`}
        disabled
        title={`${player.playerName || 'Player'} is currently Out (injured) and cannot be drafted.`}
        aria-label={`${player.playerName || 'Player'} is Out`}
      >
        <span className="btn-label">Out</span>
      </button>
    );
  }

  if (isInLineup) {
    return (
      <button
        type="button"
        className={`lineup-action-btn size-${size} is-added ${isHovered ? 'hover-remove' : ''}`}
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        title={`Remove ${player.playerName || 'player'} from Starting Five`}
        aria-label={`Remove ${player.playerName || 'player'} from Starting Five`}
      >
        {isHovered ? (
          <>
            <X size={size === 'mini' ? 11 : 12} className="btn-icon" />
            <span className="btn-label">{size === 'mini' ? '✕' : 'Remove'}</span>
          </>
        ) : (
          <>
            <Check size={size === 'mini' ? 11 : 12} className="btn-icon" />
            <span className="btn-label">{size === 'mini' ? '✓' : 'Added'}</span>
          </>
        )}
      </button>
    );
  }

  if (isLineupFull) {
    return (
      <button
        type="button"
        className={`lineup-action-btn size-${size} is-full`}
        onClick={handleClick}
        title="Starting Five is full (5/5). Remove a player first."
        aria-label="Starting Five is full (5/5). Remove a player first."
      >
        <Plus size={size === 'mini' ? 11 : 12} className="btn-icon" />
        <span className="btn-label">{size === 'mini' ? 'Full' : 'Full (5/5)'}</span>
      </button>
    );
  }

  return (
    <button
      type="button"
      className={`lineup-action-btn size-${size}`}
      onClick={handleClick}
      title={`Add ${player.playerName || 'player'} to Starting Five`}
      aria-label={`Add ${player.playerName || 'player'} to Starting Five`}
    >
      <Plus size={size === 'mini' ? 11 : 12} className="btn-icon" />
      <span className="btn-label">{size === 'mini' ? '+' : '+ Lineup'}</span>
    </button>
  );
}
