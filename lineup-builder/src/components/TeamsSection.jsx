import { useState } from 'react';
import { ChevronLeft } from 'lucide-react';
import PlayerCard from './PlayerCard';

// SECTION 3 — TEAMS (from frontend_all.json, includes OUT players)
export default function TeamsSection({ allPlayers, onAdd, onDetail }) {
  const [selectedTeam, setSelectedTeam] = useState(null);

  // Build team list from all players
  const teamMap = {};
  allPlayers.forEach(p => {
    const key = p.teamId;
    if (!teamMap[key]) {
      teamMap[key] = { teamId: key, teamName: p.teamName, players: [] };
    }
    teamMap[key].players.push(p);
  });

  const teams = Object.values(teamMap).sort((a, b) => a.teamName.localeCompare(b.teamName));

  if (selectedTeam) {
    // Sort: status_rank asc → predicted_rating desc; OUT players always at bottom
    const teamPlayers = (teamMap[selectedTeam]?.players || []).sort((a, b) => {
      if (a.status_rank !== b.status_rank) return a.status_rank - b.status_rank;
      return (b.predicted_rating ?? 0) - (a.predicted_rating ?? 0);
    });

    const team = teamMap[selectedTeam];

    return (
      <div>
        <div className="team-detail-header">
          <button className="team-back-btn" onClick={() => setSelectedTeam(null)}>
            <ChevronLeft size={14} /> All Teams
          </button>
          <div className="team-detail-name">{team?.teamName}</div>
          <div style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)', letterSpacing: '0.08em' }}>
            {teamPlayers.length} players
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

  return (
    <div>
      <div className="section-header">
        <div className="section-title">Teams</div>
        <div className="section-subtitle">
          {teams.length} NBA teams — click to view roster &amp; predictions
        </div>
      </div>

      <hr className="section-divider" />

      <div className="teams-grid">
        {teams.map(team => {
          const abbr = team.teamName.slice(0, 3).toUpperCase();
          const activeCount = team.players.filter(p => p.status !== 'Out').length;
          return (
            <div
              key={team.teamId}
              className="team-card"
              onClick={() => setSelectedTeam(team.teamId)}
            >
              <div className="team-logo-placeholder">{abbr}</div>
              <div className="team-card-name">{team.teamName}</div>
              <div className="team-card-count">{activeCount} active · {team.players.length} total</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
