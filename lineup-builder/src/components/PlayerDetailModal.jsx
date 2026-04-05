import { X } from 'lucide-react';
import StatusBadge from './StatusBadge';

// Floating modal to show detailed player stats
export default function PlayerDetailModal({ player, onClose }) {
  if (!player) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-title">Player Intel</span>
          <button className="modal-close" onClick={onClose}><X size={18} /></button>
        </div>

        {/* Player header */}
        <div className="detail-player-header">
          <img
            className="detail-headshot"
            src={player.headshot}
            alt={player.playerName}
            onError={e => { e.target.style.display = 'none'; }}
          />
          <div className="detail-player-info">
            <div className="detail-player-name">{player.playerName}</div>
            <div className="detail-matchup">{player.teamName} vs {player.opponentName || '—'}</div>
            <StatusBadge status={player.status} />
          </div>
        </div>

        {/* Stats grid */}
        <div className="detail-stats-grid">
          <div className="detail-stat-card">
            <div className="detail-stat-label">Predicted Rating</div>
            <div className="detail-stat-value">{player.predicted_rating?.toFixed(2) ?? '—'}</div>
          </div>
          <div className="detail-stat-card">
            <div className="detail-stat-label">Last Game</div>
            <div className="detail-stat-value" style={{ fontSize: 20, color: 'var(--text-primary)' }}>
              {player.last_game?.toFixed(2) ?? '—'}
            </div>
          </div>
          <div className="detail-stat-card">
            <div className="detail-stat-label">Last 3 Avg</div>
            <div className="detail-stat-value" style={{ fontSize: 20, color: 'var(--text-secondary)' }}>
              {player.last3_avg?.toFixed(2) ?? '—'}
            </div>
          </div>
          <div className="detail-stat-card">
            <div className="detail-stat-label">Last 7 Avg</div>
            <div className="detail-stat-value" style={{ fontSize: 20, color: 'var(--text-secondary)' }}>
              {player.last7_avg?.toFixed(2) ?? '—'}
            </div>
          </div>
        </div>

        {/* Opponent */}
        <div style={{ marginTop: 2 }} className="detail-stat-card">
          <div className="detail-stat-label">Opponent</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, color: 'var(--text-primary)', marginTop: 2, fontWeight: 600 }}>
            {player.opponentName || 'Unknown'}
          </div>
        </div>
      </div>
    </div>
  );
}
