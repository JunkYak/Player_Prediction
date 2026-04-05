import { User } from 'lucide-react';
import StatusBadge from './StatusBadge';

// A single slot in the Draft lineup bar
export default function Slot({ player, index, onClick }) {
  return (
    <div className="lineup-slot" onClick={onClick} title={player ? player.playerName : `Slot ${index + 1}`}>
      <div className={`slot-circle ${player ? 'filled' : ''}`}>
        {player ? (
          <img
            src={player.headshot}
            alt={player.playerName}
            onError={e => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
          />
        ) : null}
        {!player && <User size={32} className="slot-empty-icon" />}
        {player && (
          <div style={{ display: 'none', width: '100%', height: '100%', alignItems: 'center', justifyContent: 'center', background: '#1a1a1a' }}>
            <User size={32} style={{ color: '#444' }} />
          </div>
        )}
      </div>
      <div className="slot-label">
        {player ? (
          <>
            <div className="slot-name">{player.playerName}</div>
            <div className="slot-rating">{player.predicted_rating?.toFixed(2)}</div>
          </>
        ) : (
          <span>Empty</span>
        )}
      </div>
      {player && (
        <div className="slot-status">
          <StatusBadge status={player.status} />
        </div>
      )}
    </div>
  );
}
