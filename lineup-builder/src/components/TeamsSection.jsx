import { useState, useMemo } from 'react';
import { ChevronLeft, Users, Activity, Calendar, Search, X, ShieldAlert } from 'lucide-react';
import PlayerCard from './PlayerCard';
import NbaLogo from './NbaLogo';
import { getTeamInfo } from '../constants/teams';
import {
  evaluateScheduleStatus,
  filterTeamsPlayingTomorrow,
  formatDisplayDate,
} from '../utils/schedule';

/**
 * TeamsSection
 *
 * Displays all 30 NBA teams with official logos, active/total roster counts,
 * mean projected player rating, and interactive navigation.
 *
 * Features:
 * 1. Filter teams playing tomorrow (timezone-aware schedule verification)
 * 2. Search teams by name or city
 * 3. Roster drilldown with Start Five drafting actions
 */
export default function TeamsSection({
  allPlayers,
  onAdd,
  onDetail,
  slots = [],
  onToggleLineup,
  homeData = [],
  gameDate = null,
  gamesScheduled = false,
  generatedAt = null,
  tomorrowSchedule = null,
  referenceDate = undefined, // Injected for deterministic testing
}) {
  const [selectedTeamId, setSelectedTeamId] = useState(null);
  const [teamSearchQuery, setTeamSearchQuery] = useState('');
  const [playingTomorrowOnly, setPlayingTomorrowOnly] = useState(false);

  // 1. Group all league players by teamId
  const teamMap = useMemo(() => {
    const map = {};
    const validPlayers = (Array.isArray(allPlayers) ? allPlayers : []).filter(p => p && p.teamId);

    validPlayers.forEach(p => {
      const key = p.teamId;
      if (!map[key]) {
        const info = getTeamInfo(key, p.teamName);
        map[key] = {
          teamId: key,
          teamName: p.teamName || info.name,
          fullName: info.fullName,
          city: info.city,
          tricode: info.tricode,
          players: [],
        };
      }
      map[key].players.push(p);
    });
    return map;
  }, [allPlayers]);

  const teams = useMemo(() => {
    return Object.values(teamMap).sort((a, b) =>
      (a.fullName || a.teamName).localeCompare(b.fullName || b.teamName)
    );
  }, [teamMap]);

  // 2. Evaluate schedule status deterministically using America/New_York timezone policy
  const scheduleStatus = useMemo(() => {
    return evaluateScheduleStatus({
      gameDate,
      generatedAt,
      gamesScheduled,
      homeData,
      referenceDate,
      tomorrowSchedule,
    });
  }, [gameDate, generatedAt, gamesScheduled, homeData, referenceDate, tomorrowSchedule]);

  // 3. Derive qualifying teams and deduplicated matchups for tomorrow
  const scheduledTeamInfo = useMemo(() => {
    return filterTeamsPlayingTomorrow({
      teams,
      homeData,
      allPlayers,
      scheduleStatus,
      tomorrowSchedule,
    });
  }, [teams, homeData, allPlayers, scheduleStatus, tomorrowSchedule]);

  // 4. Filter teams based on search query and "Playing Tomorrow" toggle
  const filteredTeams = useMemo(() => {
    let list = teams;

    // Filter by schedule (Playing Tomorrow)
    if (playingTomorrowOnly) {
      if (scheduleStatus.status === 'CONFIRMED') {
        const qualifyingIds = new Set(scheduledTeamInfo.qualifyingTeams.map(t => t.teamId));
        list = list.filter(t => qualifyingIds.has(t.teamId));
      } else {
        list = [];
      }
    }

    // Filter by name/city search
    if (teamSearchQuery.trim()) {
      const q = teamSearchQuery.toLowerCase().trim();
      list = list.filter(t =>
        (t.fullName || '').toLowerCase().includes(q) ||
        (t.teamName || '').toLowerCase().includes(q) ||
        (t.city || '').toLowerCase().includes(q) ||
        (t.tricode || '').toLowerCase().includes(q)
      );
    }

    return list;
  }, [teams, playingTomorrowOnly, scheduleStatus.status, scheduledTeamInfo.qualifyingTeams, teamSearchQuery]);

  // ─── TEAM ROSTER DRILLDOWN VIEW ──────────────────────────────────────────
  if (selectedTeamId) {
    const team = teamMap[selectedTeamId];
    // Sort: status_rank asc -> predicted_rating desc; OUT players at bottom
    const teamPlayers = (team?.players || []).slice().sort((a, b) => {
      if (a.status_rank !== b.status_rank) return a.status_rank - b.status_rank;
      return (b.predicted_rating ?? 0) - (a.predicted_rating ?? 0);
    });

    const activeCount = teamPlayers.filter(p => p.status !== 'Out').length;
    const avgRating = teamPlayers.length
      ? (teamPlayers.reduce((acc, p) => acc + (p.predicted_rating || 0), 0) / teamPlayers.length).toFixed(2)
      : '0.00';

    const isPlayingTomorrow = scheduleStatus.status === 'CONFIRMED' &&
      scheduledTeamInfo.qualifyingTeams.some(t => t.teamId === team?.teamId);
    const opponent = isPlayingTomorrow ? scheduledTeamInfo.opponentMap.get(team?.teamId) : null;

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
                {opponent && (
                  <>
                    <span className="dot-sep">•</span>
                    <span className="team-matchup-badge">
                      <Calendar size={12} style={{ marginRight: 4 }} />
                      Scheduled vs {opponent}
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="players-grid">
          {teamPlayers.map(p => (
            <PlayerCard
              key={p.personId}
              player={p}
              onAdd={onAdd}
              onDetail={onDetail}
              slots={slots}
              onToggleLineup={onToggleLineup}
              showNotPlayed
            />
          ))}
        </div>
      </div>
    );
  }

  // ─── ALL TEAMS GRID VIEW ─────────────────────────────────────────────────
  let subtitle = `${teams.length} franchises — official rosters & active performance projections`;
  if (playingTomorrowOnly) {
    if (scheduleStatus.status === 'CONFIRMED') {
      subtitle = `${filteredTeams.length} teams with confirmed games scheduled for today (${formatDisplayDate(scheduleStatus.targetDate)})`;
    } else if (scheduleStatus.status === 'OFF_DAY') {
      subtitle = `NBA Off-Day: No games scheduled for today (${formatDisplayDate(scheduleStatus.targetDate)})`;
    } else if (scheduleStatus.status === 'TODAY_ACTIVE') {
      subtitle = `Active slate is tracking games for ${formatDisplayDate(scheduleStatus.todayDate)}`;
    } else if (scheduleStatus.status === 'STALE') {
      subtitle = `Schedule telemetry is stale (${formatDisplayDate(scheduleStatus.slateDate)}) — schedule for ${formatDisplayDate(scheduleStatus.targetDate)} not yet synced`;
    } else if (scheduleStatus.status === 'FUTURE_SLATE') {
      subtitle = `Next slate is on ${formatDisplayDate(scheduleStatus.slateDate)} — no games scheduled for ${formatDisplayDate(scheduleStatus.targetDate)}`;
    } else {
      subtitle = 'Schedule telemetry is currently unavailable';
    }
  }

  return (
    <div className="teams-section-container">
      {/* ── HEADER ─────────────────────────────────────────────────── */}
      <div className="section-header">
        <div className="section-title">NBA Teams</div>
        <div className="section-subtitle">{subtitle}</div>
      </div>

      <hr className="section-divider" />

      {/* ── TOOLBAR: SEARCH & PLAYING TODAY FILTER ───────────────────── */}
      <div className="teams-toolbar">
        {/* Search */}
        <div className="teams-search-box">
          <Search size={14} className="search-icon" aria-hidden="true" />
          <input
            type="text"
            className="teams-search-input"
            placeholder="Search teams by name or city..."
            value={teamSearchQuery}
            onChange={e => setTeamSearchQuery(e.target.value)}
            aria-label="Search teams by name or city"
            autoComplete="off"
            spellCheck="false"
          />
          {teamSearchQuery && (
            <button
              type="button"
              className="search-clear-btn"
              onClick={() => setTeamSearchQuery('')}
              title="Clear team search"
              aria-label="Clear team search"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Filter Button: Playing Today */}
        <div className="teams-filter-group">
          <button
            type="button"
            className={`team-filter-btn ${playingTomorrowOnly ? 'active' : ''}`}
            onClick={() => setPlayingTomorrowOnly(prev => !prev)}
            aria-pressed={playingTomorrowOnly}
            title={
              scheduleStatus.status === 'CONFIRMED'
                ? `Filter to teams playing today (${formatDisplayDate(scheduleStatus.targetDate)})`
                : scheduleStatus.message
            }
          >
            <Calendar size={14} className="filter-icon" />
            <span className="filter-label">Playing Today</span>
            {scheduleStatus.status === 'CONFIRMED' && scheduledTeamInfo.count > 0 && (
              <span className="filter-count-badge">{scheduledTeamInfo.count}</span>
            )}
          </button>
        </div>
      </div>

      {/* ── CONFIRMED MATCHUPS BANNER WHEN FILTER IS ACTIVE ─────────── */}
      {playingTomorrowOnly && scheduleStatus.status === 'CONFIRMED' && scheduledTeamInfo.matchups.length > 0 && (
        <div className="scheduled-matchups-banner">
          <div className="matchup-banner-title">
            <Calendar size={14} style={{ color: 'var(--accent)' }} />
            <span>Confirmed Matchups for Today ({formatDisplayDate(scheduleStatus.targetDate)}):</span>
          </div>
          <div className="matchup-banner-list">
            {scheduledTeamInfo.matchups.map((m, idx) => (
              <span key={idx} className="matchup-pill">{m}</span>
            ))}
          </div>
        </div>
      )}

      {/* ── DISTINCT SCHEDULE STATUS EMPTY / INFORMATIVE STATES ────── */}
      {playingTomorrowOnly && scheduleStatus.status !== 'CONFIRMED' ? (
        scheduleStatus.status === 'OFF_DAY' ? (
          <div className="empty-state schedule-empty-state">
            <Calendar size={36} className="empty-icon text-muted" />
            <div className="empty-title">NBA Off-Day: No Games Scheduled for Today</div>
            <div className="empty-subtitle">
              The official NBA schedule confirms 0 games scheduled for {formatDisplayDate(scheduleStatus.targetDate)}. All 30 franchises are resting.
            </div>
            <button
              type="button"
              className="clear-search-action-btn"
              onClick={() => setPlayingTomorrowOnly(false)}
            >
              View All 30 Teams
            </button>
          </div>
        ) : scheduleStatus.status === 'TODAY_ACTIVE' ? (
          <div className="empty-state schedule-empty-state">
            <Activity size={36} className="empty-icon text-muted" />
            <div className="empty-title">Schedule Not Yet Active</div>
            <div className="empty-subtitle">
              The active prediction slate is tracking games for {formatDisplayDate(scheduleStatus.todayDate)}.
            </div>
            <button
              type="button"
              className="clear-search-action-btn"
              onClick={() => setPlayingTomorrowOnly(false)}
            >
              View All 30 Teams
            </button>
          </div>
        ) : scheduleStatus.status === 'STALE' ? (
          <div className="empty-state schedule-empty-state">
            <ShieldAlert size={36} className="empty-icon text-muted" />
            <div className="empty-title">Schedule Telemetry Stale</div>
            <div className="empty-subtitle">
              The latest available snapshot is from {formatDisplayDate(scheduleStatus.slateDate)}. The schedule for {formatDisplayDate(scheduleStatus.targetDate)} has not yet been synced by the ingestion pipeline.
            </div>
            <button
              type="button"
              className="clear-search-action-btn"
              onClick={() => setPlayingTomorrowOnly(false)}
            >
              View All 30 Teams
            </button>
          </div>
        ) : scheduleStatus.status === 'FUTURE_SLATE' ? (
          <div className="empty-state schedule-empty-state">
            <Calendar size={36} className="empty-icon text-muted" />
            <div className="empty-title">No Games Scheduled for Today</div>
            <div className="empty-subtitle">
              The next available prediction slate is scheduled for {formatDisplayDate(scheduleStatus.slateDate)}, not today ({formatDisplayDate(scheduleStatus.targetDate)}).
            </div>
            <button
              type="button"
              className="clear-search-action-btn"
              onClick={() => setPlayingTomorrowOnly(false)}
            >
              View All 30 Teams
            </button>
          </div>
        ) : (
          <div className="empty-state schedule-empty-state">
            <ShieldAlert size={36} className="empty-icon text-muted" />
            <div className="empty-title">Schedule Telemetry Unavailable</div>
            <div className="empty-subtitle">
              Unable to retrieve schedule telemetry from the prediction service ({formatDisplayDate(scheduleStatus.targetDate)}).
            </div>
            <button
              type="button"
              className="clear-search-action-btn"
              onClick={() => setPlayingTomorrowOnly(false)}
            >
              View All 30 Teams
            </button>
          </div>
        )
      ) : filteredTeams.length === 0 ? (
        <div className="empty-state search-empty-state">
          <Search size={32} className="search-empty-icon" />
          <div className="search-empty-title">No teams match "{teamSearchQuery}"</div>
          <div className="search-empty-subtitle">
            {playingTomorrowOnly ? 'Try turning off "Playing Today" or modifying your search.' : 'Check the spelling or clear the filter.'}
          </div>
          <button
            type="button"
            className="clear-search-action-btn"
            onClick={() => {
              setTeamSearchQuery('');
              setPlayingTomorrowOnly(false);
            }}
          >
            Reset Filters
          </button>
        </div>
      ) : (
        /* ── TEAMS GRID ───────────────────────────────────────────── */
        <div className="teams-grid" role="list" aria-label="NBA Teams list">
          {filteredTeams.map(team => {
            const activeCount = team.players.filter(p => p && p.status !== 'Out').length;
            const totalCount = team.players.length;
            const avgRating = totalCount
              ? (team.players.reduce((sum, p) => sum + (p.predicted_rating || 0), 0) / totalCount).toFixed(2)
              : '0.00';

            const isPlayingTomorrow = scheduleStatus.status === 'CONFIRMED' &&
              scheduledTeamInfo.qualifyingTeams.some(t => t.teamId === team.teamId);
            const opponent = isPlayingTomorrow ? scheduledTeamInfo.opponentMap.get(team.teamId) : null;

            return (
              <div
                key={team.teamId}
                role="button"
                tabIndex={0}
                className={`team-card ${isPlayingTomorrow ? 'has-game-tomorrow' : ''}`}
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
                  <div className="team-card-top-meta">
                    <span className="team-card-city">{team.city}</span>
                    {isPlayingTomorrow && opponent && (
                      <span className="team-tomorrow-tag" title={`Scheduled vs ${opponent}`}>
                        vs {opponent}
                      </span>
                    )}
                  </div>
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
      )}
    </div>
  );
}
