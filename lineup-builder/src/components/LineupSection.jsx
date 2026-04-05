import Slot from './Slot';

// SECTION 1 — LINEUP / DRAFT BAR
export default function LineupSection({ slots, onSlotClick }) {
  return (
    <div>
      <div className="section-header">
        <div className="section-title">Draft Lineup</div>
        <div className="section-subtitle">
          {slots.filter(Boolean).length} / 5 slots filled — click a slot to pick a player
        </div>
      </div>

      <hr className="section-divider" />

      <div className="lineup-slots-row">
        {slots.map((player, i) => (
          <Slot
            key={i}
            index={i}
            player={player}
            onClick={() => onSlotClick(i)}
          />
        ))}
      </div>
    </div>
  );
}
