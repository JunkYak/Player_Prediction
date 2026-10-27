import { useState } from 'react';
import { ChevronLeft, Users, Activity } from 'lucide-react';
import PlayerCard from './PlayerCard';
import NbaLogo from './NbaLogo';
import { getTeamInfo } from '../constants/teams';

/**
 * TeamsSection
 *
 * Displays all 30 NBA teams with official logos, active/total roster counts,
 * mean projected player rating, and interactive keyboard-accessible navigation.
 */
export default function TeamsSection({ allPlayers, onAdd, onDetail }) {
  const [selectedTeamId, setSelectedTeamId] = useState(null);

  // Group players by teamId
  const teamMap = {};
  const validPlayers = (Array.isArray(allPlayers) ? allPlayers : []).filter(p => p && p.teamId);

  validPlayers.forEach(p => {
    const key = p.teamId;
    if (!teamMap[key]) {
      const info = getTeamInfo(key, p.teamName);
      teamMap[key] = {
        teamId: key,
        teamName: p.teamName || info.name,
        fullName: info.fullName,
        city: info.city,
        tricode: info.tricode,
        players: [],
      };
    }
    teamMap[key].players.push(p);
  });

  const teams = Object.values(teamMap).sort((a, b) =>
    (a.fullName || a.teamName).localeCompare(b.fullName || b.teamName)
  );

  // ─── TEAM ROSTER DRILLDOWN VIEW ──────────────────────────────────────────
  if (selectedTeamId) {
    const team = teamMap[selectedTeamId];
    // Sort: status_rank asc → predicted_rating desc; OUT players at bottom
    const teamPlayers = (team?.players || []).slice().sort((a, b) => {
      if (a.status_rank !== b.status_rank) return a.status_rank - b.status_rank;
      return (b.predicted_rating ?? 0) - (a.predicted_rating ?? 0);
    });

    const activeCount = teamPlayers.filter(p => p.status !== 'Out').length;
    const avgRating = teamPlayers.length
      ? (teamPlayers.reduce((acc, p) => acc + (p.predicted_rating || 0), 0) / teamPlayers.length).toFixed(2)
      : '0.00';

    return (
      <div className="team-detail-view">
        <div className="team-detail-header">
          <button
            type="button"
            className="team-back-btn"
            onClick={() => setSelectedTeamId(null)}
            aria-label="Back to all teams"
          >
            <ChevronLeft size={16} /> All Teams
          </button>

          <div className="team-detail-identity">
            <NbaLogo teamId={team?.teamId} teamName={team?.teamName} size={48} />
            <div>
              <div className="team-detail-name">{team?.fullName || team?.teamName}</div>
              <div className="team-detail-meta">
                <span>{activeCount} Active</span>
                <span className="dot-sep">•</span>
                <span>{teamPlayers.length} Total</span>
                <span className="dot-sep">•</span>
                <span className="team-avg-badge">Avg Rating: {avgRating}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="players-grid">
          {teamPlayers.map(p => (
            <PlayerCard
              key={p.personId}
              player={p}
              onAdd={p.status !== 'Out' ? onAdd : undefined}
              onDetail={onDetail}
              showNotPlayed
            />
          ))}
        </div>
      </div>
    );
  }

  // ─── ALL 30 TEAMS OVERVIEW GRID ──────────────────────────────────────────
  return (
    <div className="teams-section-container">
      <div className="section-header">
        <div className="section-title">NBA Teams</div>
        <div className="section-subtitle">
          {teams.length} franchises — official rosters &amp; active performance projections
        </div>
      </div>

      <hr className="section-divider" />

      <div className="teams-grid" role="list" aria-label="NBA Teams list">
        {teams.map(team => {
          const activeCount = team.players.filter(p => p && p.status !== 'Out').length;
          const totalCount = team.players.length;
          const avgRating = totalCount
            ? (team.players.reduce((sum, p) => sum + (p.predicted_rating || 0), 0) / totalCount).toFixed(2)
            : '0.00';

          return (
            <div
              key={team.teamId}
              role="button"
              tabIndex={0}
              className="team-card"
              onClick={() => setSelectedTeamId(team.teamId)}
              onKeyDown={e => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSelectedTeamId(team.teamId);
                }
              }}
              aria-label={`View roster for ${team.fullName}`}
            >
              <div className="team-card-logo-wrap">
                <NbaLogo teamId={team.teamId} teamName={team.teamName} size={52} />
              </div>
              <div className="team-card-info">
                <div className="team-card-city">{team.city}</div>
                <div className="team-card-name">{team.fullName || team.teamName}</div>
                <div className="team-card-stats">
                  <span className="team-stat-item">
                    <Users size={12} style={{ marginRight: 4, opacity: 0.7 }} />
                    {activeCount} / {totalCount}
                  </span>
                  <span className="team-stat-item team-rating-metric">
                    <Activity size={12} style={{ marginRight: 4, opacity: 0.7 }} />
                    {avgRating}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
