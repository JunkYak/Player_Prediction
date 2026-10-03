import Slot from './Slot';
import { SLOT_MULTIPLIERS, getSlotProjectedValue, getTotalProjectedScore } from '../constants';
import { RotateCcw } from 'lucide-react';

// SECTION 1 — LINEUP / DRAFT BAR
export default function LineupSection({
  slots,
  onSlotClick,
  onRemoveSlot,
  onClearAll,
  gamesAvailable = true,
}) {
  const filledCount = slots.filter(Boolean).length;
  const totalScore = getTotalProjectedScore(slots);

  return (
    <div className="lineup-container">
      <div className="lineup-top-header">
        <div className="section-header" style={{ marginBottom: 0 }}>
          <div className="section-title">Draft Lineup</div>
          <div className="section-subtitle">
            {!gamesAvailable && filledCount === 0
              ? "No games scheduled for tomorrow — draft slate opens on game day"
              : `${filledCount} / 5 slots filled — click any slot to draft or replace a player`}
          </div>
        </div>

        {/* Lineup Scoreboard Card */}
        <div className="lineup-score-card">
          <div className="score-card-item">
            <span className="score-card-label">Total Projected</span>
            <span className="score-card-value">{totalScore.toFixed(2)}</span>
          </div>
          <div className="score-card-divider" />
          <div className="score-card-item">
            <span className="score-card-label">Lineup Status</span>
            <span className="score-card-slots">
              <span className="active-count">{filledCount}</span> / 5 Filled
            </span>
          </div>
          {filledCount > 0 && onClearAll && (
            <button
              type="button"
              className="lineup-clear-all-btn"
              onClick={onClearAll}
              title="Clear all 5 slots"
            >
              <RotateCcw size={12} />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>

      <hr className="section-divider" />

      {/* 5 Slots Row */}
      <div className="lineup-slots-row">
        {slots.map((player, i) => {
          const multiplier = SLOT_MULTIPLIERS[i] ?? 1.0;
          const projectedValue = getSlotProjectedValue(player, i);

          return (
            <Slot
              key={i}
              index={i}
              player={player}
              multiplier={multiplier}
              projectedValue={projectedValue}
              onClick={() => onSlotClick(i)}
              onRemove={() => onRemoveSlot && onRemoveSlot(i)}
            />
          );
        })}
      </div>
    </div>
  );
}
