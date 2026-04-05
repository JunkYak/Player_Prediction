import { X, User } from 'lucide-react';
import StatusBadge from './StatusBadge';

// Modal for picking which slot to assign a player to
export default function SlotPickerModal({ player, slots, onSelectSlot, onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-title">Assign to Slot</span>
          <button className="modal-close" onClick={onClose}><X size={18} /></button>
        </div>

        {/* Selected player preview */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20, padding: '12px 14px', background: '#111', border: '1px solid var(--border)' }}>
          <img
            src={player.headshot}
            alt={player.playerName}
            style={{ width: 44, height: 44, objectFit: 'cover', objectPosition: 'top', borderRadius: 2, border: '1px solid var(--border)' }}
            onError={e => { e.target.style.display = 'none'; }}
          />
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
              {player.playerName}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)', marginTop: 3 }}>
              {player.teamName} — Rating: <span style={{ color: 'var(--accent)' }}>{player.predicted_rating?.toFixed(2)}</span>
            </div>
          </div>
          <div style={{ marginLeft: 'auto' }}>
            <StatusBadge status={player.status} />
          </div>
        </div>

        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 14 }}>
          Select a slot
        </div>

        <div className="slot-picker-grid">
          {slots.map((slot, i) => (
            <div
              key={i}
              className="slot-picker-item"
              onClick={() => onSelectSlot(i)}
            >
              <div className={`slot-picker-circle ${slot ? 'occupied' : ''}`}>
                {slot ? (
                  <img
                    src={slot.headshot}
                    alt={slot.playerName}
                    onError={e => { e.target.style.display = 'none'; }}
                    style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'top' }}
                  />
                ) : (
                  <User size={20} style={{ color: '#444' }} />
                )}
              </div>
              <div className="slot-picker-label">
                {slot ? slot.playerName.split(' ').pop() : `Slot ${i + 1}`}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
