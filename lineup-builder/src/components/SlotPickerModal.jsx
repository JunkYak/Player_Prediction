import { X, User, ArrowRight } from 'lucide-react';
import StatusBadge from './StatusBadge';
import { SLOT_MULTIPLIERS, getSlotProjectedValue } from '../constants';

// Modal for picking which slot to assign a player to
export default function SlotPickerModal({ player, slots, onSelectSlot, onClose }) {
  if (!player) return null;

  const rawRating = player.predicted_rating;
  const baseRating = typeof rawRating === 'number' && !isNaN(rawRating) ? rawRating : 0;
  const name = player.playerName || 'Unknown Player';
  const team = player.teamName || '—';
  const opponent = player.opponentName || '—';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box modal-box-wide" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-title">Assign to Lineup Slot</span>
          <button className="modal-close" onClick={onClose}><X size={18} /></button>
        </div>

        {/* Selected player preview */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20, padding: '12px 14px', background: '#111', border: '1px solid var(--border)' }}>
          <img
            src={player.headshot || ''}
            alt={name}
            style={{ width: 48, height: 48, objectFit: 'cover', objectPosition: 'top', borderRadius: 2, border: '1px solid var(--border)' }}
            onError={e => { e.target.style.display = 'none'; }}
          />
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
              {name}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)', marginTop: 3 }}>
              {team} vs {opponent}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)', marginTop: 2 }}>
              Base Rating: <span style={{ color: 'var(--accent)', fontWeight: 700 }}>{baseRating.toFixed(2)}</span>
            </div>
          </div>
          <div style={{ marginLeft: 'auto' }}>
            <StatusBadge status={player.status} />
          </div>
        </div>

        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 14 }}>
          Select target slot (higher multiplier yields higher score)
        </div>

        <div className="slot-picker-grid">
          {slots.map((slot, i) => {
            const multiplier = SLOT_MULTIPLIERS[i] ?? 1.0;
            const projectedIfSelected = (baseRating * multiplier).toFixed(2);
            const isCurrentPlayer = slot?.personId === player.personId;

            return (
              <div
                key={i}
                className={`slot-picker-item ${isCurrentPlayer ? 'is-current' : ''}`}
                onClick={() => onSelectSlot(i)}
                title={slot ? `Replace ${slot.playerName} in Slot ${i + 1}` : `Assign to Slot ${i + 1}`}
              >
                <div className="slot-picker-header">
                  <span className="slot-picker-num">SLOT {i + 1}</span>
                  <span className="slot-picker-mult">{multiplier.toFixed(1)}x</span>
                </div>

                <div className={`slot-picker-circle ${slot ? 'occupied' : ''}`}>
                  {slot ? (
                    <img
                      src={slot.headshot}
                      alt={slot.playerName}
                      onError={e => { e.target.style.display = 'none'; }}
                      style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'top' }}
                    />
                  ) : (
                    <User size={22} style={{ color: '#444' }} />
                  )}
                </div>

                <div className="slot-picker-label">
                  {slot ? (
                    <div className="slot-replace-info">
                      <span className="replace-tag">Replace</span>
                      <span className="current-name">{slot.playerName.split(' ').pop()}</span>
                    </div>
                  ) : (
                    <span className="empty-tag">Empty</span>
                  )}
                </div>

                <div className="slot-picker-projected">
                  <span className="label">Projected</span>
                  <span className="val">{projectedIfSelected}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
