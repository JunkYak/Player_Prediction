import { useState, useMemo } from 'react';
import { Search, X, Users, Calendar, ShieldAlert, Activity } from 'lucide-react';
import PlayerCard from './PlayerCard';
import { evaluateScheduleStatus, formatDisplayDate } from '../utils/schedule';

/**
 * Normalizes text for case-insensitive, accent-insensitive search.
 * e.g., "Luka Dončić" -> "luka doncic"
 */
function normalizeSearchText(str) {
  return (str || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .trim();
}

// SECTION 2 — TOP PLAYERS (derived from confirmed tomorrow schedule & predictions)
export default function TopPlayersSection({
  players,
  gameDate = null,
  tomorrowSchedule = null,
  onAdd,
  onDetail,
  slots = [],
  onToggleLineup,
  referenceDate = undefined,
}) {
  const [searchQuery, setSearchQuery] = useState('');

  // 1. Evaluate schedule status deterministically using America/New_York timezone policy
  const scheduleStatus = useMemo(() => {
    return evaluateScheduleStatus({
      gameDate,
      homeData: players,
      tomorrowSchedule,
      referenceDate,
    });
  }, [gameDate, players, tomorrowSchedule, referenceDate]);

  // 2. Evaluate runtime alignment between player slate and tomorrow's confirmed schedule
  const slateAlignment = useMemo(() => {
    const status = scheduleStatus.status;

    // A. Off-day: explicitly 0 games scheduled
    if (status === 'OFF_DAY') {
      return {
        isAligned: false,
        status: 'OFF_DAY',
        title: 'NBA Off-Day: No Games Scheduled for Tomorrow',
        subtitle: `The official NBA schedule confirms 0 games scheduled for ${formatDisplayDate(scheduleStatus.targetDate)}. All 30 franchises are resting.`,
      };
    }

    // B. Telemetry Stale: schedule date < tomorrowET
    if (status === 'STALE') {
      return {
        isAligned: false,
        status: 'STALE',
        title: 'Schedule Telemetry Stale',
        subtitle: `The latest available snapshot is from ${formatDisplayDate(scheduleStatus.slateDate)}. Tomorrow's confirmed schedule (${formatDisplayDate(scheduleStatus.targetDate)}) has not yet been synced by the ingestion pipeline.`,
      };
    }

    // C. Telemetry Unavailable / Fetch Failed
    if (status === 'UNAVAILABLE') {
      return {
        isAligned: false,
        status: 'UNAVAILABLE',
        title: 'Schedule Telemetry Unavailable',
        subtitle: `Unable to retrieve tomorrow's schedule telemetry from the prediction service (${formatDisplayDate(scheduleStatus.targetDate)}).`,
      };
    }

    // D. Today active
    if (status === 'TODAY_ACTIVE') {
      return {
        isAligned: false,
        status: 'TODAY_ACTIVE',
        title: "Tomorrow's Schedule Not Yet Active",
        subtitle: `The active prediction slate is tracking today's games (${formatDisplayDate(scheduleStatus.todayDate)}). Tomorrow's schedule (${formatDisplayDate(scheduleStatus.targetDate)}) will publish once today's games conclude.`,
      };
    }

    // E. Future slate
    if (status === 'FUTURE_SLATE') {
      return {
        isAligned: false,
        status: 'FUTURE_SLATE',
        title: 'No Games Scheduled for Tomorrow',
        subtitle: `The next available prediction slate is scheduled for ${formatDisplayDate(scheduleStatus.slateDate)}, not tomorrow (${formatDisplayDate(scheduleStatus.targetDate)}).`,
      };
    }

    // F. Confirmed: Verify player slate matches tomorrow's schedule date
    if (status === 'CONFIRMED') {
      const schedDate = tomorrowSchedule?.schedule_date;
      if (gameDate && schedDate && gameDate !== schedDate) {
        return {
          isAligned: false,
          status: 'SYNCING',
          title: "Tomorrow's Player Slate Syncing",
          subtitle: `Active player predictions (${formatDisplayDate(gameDate)}) are not yet synchronized with tomorrow's confirmed schedule (${formatDisplayDate(schedDate)}). Awaiting pipeline sync.`,
        };
      }
      return { isAligned: true, status: 'CONFIRMED' };
    }

    return { isAligned: true, status };
  }, [scheduleStatus, tomorrowSchedule, gameDate]);

  // Set of confirmed team IDs to exclude unscheduled teams
  const confirmedTeamIds = useMemo(() => {
    if (tomorrowSchedule && Array.isArray(tomorrowSchedule.team_ids) && tomorrowSchedule.team_ids.length > 0) {
      return new Set(tomorrowSchedule.team_ids.map(Number));
    }
    return null;
  }, [tomorrowSchedule]);

  // 3. Filter out OUT players and ensure players belong strictly to confirmed tomorrow teams
  const eligiblePlayers = useMemo(() => {
    if (!slateAlignment.isAligned) return [];
    return (Array.isArray(players) ? players : []).filter(p => {
      if (!p || p.status === 'Out') return false;
      if (confirmedTeamIds && !confirmedTeamIds.has(Number(p.teamId))) return false;
      return true;
    });
  }, [players, confirmedTeamIds, slateAlignment.isAligned]);

  // 4. Sort eligible players: status_rank -> ranking_score desc -> predicted_rating desc
  const sortedPlayers = useMemo(() => {
    return eligiblePlayers.slice().sort((a, b) => {
      const aRank = typeof a.status_rank === 'number' ? a.status_rank : 5;
      const bRank = typeof b.status_rank === 'number' ? b.status_rank : 5;
      if (aRank !== bRank) return aRank - bRank;

      const aScore = typeof a.ranking_score === 'number'
        ? a.ranking_score
        : (typeof a.adjusted_predicted_rating === 'number'
            ? a.adjusted_predicted_rating
            : (Number(a.predicted_rating) || 0) * (1 + (Number(a.season_performance_bonus) || 0)));
      const bScore = typeof b.ranking_score === 'number'
        ? b.ranking_score
        : (typeof b.adjusted_predicted_rating === 'number'
            ? b.adjusted_predicted_rating
            : (Number(b.predicted_rating) || 0) * (1 + (Number(b.season_performance_bonus) || 0)));
      if (bScore !== aScore) return bScore - aScore;

      const aRating = typeof a.predicted_rating === 'number' ? a.predicted_rating : 0;
      const bRating = typeof b.predicted_rating === 'number' ? b.predicted_rating : 0;
      return bRating - aRating;
    });
  }, [eligiblePlayers]);

  // 5. Filter matching search query (case-insensitive, accent-tolerant, partial-match)
  const filteredPlayers = useMemo(() => {
    const q = normalizeSearchText(searchQuery);
    if (!q) return sortedPlayers;

    return sortedPlayers.filter(p => {
      const name = normalizeSearchText(p.playerName);
      const team = normalizeSearchText(p.teamName);
      const opponent = normalizeSearchText(p.opponentName);
      return name.includes(q) || team.includes(q) || opponent.includes(q);
    });
  }, [sortedPlayers, searchQuery]);

  const isSearching = searchQuery.trim().length > 0;

  // Header Subtitle Logic
  let subtitle;
  if (!slateAlignment.isAligned) {
    subtitle = slateAlignment.subtitle;
  } else if (sortedPlayers.length === 0) {
    subtitle = `No active players available for tomorrow's games (${formatDisplayDate(scheduleStatus.targetDate)})`;
  } else if (isSearching) {
    subtitle = `Showing ${filteredPlayers.length} of ${sortedPlayers.length} players matching "${searchQuery}"`;
  } else {
    subtitle = `${sortedPlayers.length} active players across tomorrow's confirmed games (${formatDisplayDate(scheduleStatus.targetDate)}) — sorted by availability & ranking score`;
  }

  return (
    <div className="top-players-container">
      {/* ── HEADER ─────────────────────────────────────────────────── */}
      <div className="section-header">
        <div className="section-title">Top Players</div>
        <div className="section-subtitle">{subtitle}</div>
      </div>

      <hr className="section-divider" />

      {/* ── SEARCH INPUT TOOLBAR ───────────────────────────────────── */}
      {slateAlignment.isAligned && sortedPlayers.length > 0 && (
        <div className="players-search-toolbar">
          <div className="players-search-box">
            <Search size={15} className="search-icon" aria-hidden="true" />
            <input
              type="text"
              className="players-search-input"
              placeholder="Search players by name, team, or opponent..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              aria-label="Search players by name, team, or opponent"
              autoComplete="off"
              spellCheck="false"
            />
            {isSearching && (
              <button
                type="button"
                className="search-clear-btn"
                onClick={() => setSearchQuery('')}
                title="Clear search"
                aria-label="Clear search text"
              >
                <X size={14} />
              </button>
            )}
          </div>

          {isSearching && (
            <div className="search-status-tag">
              <Users size={12} style={{ marginRight: 4 }} />
              <span>{filteredPlayers.length} {filteredPlayers.length === 1 ? 'match' : 'matches'}</span>
            </div>
          )}
        </div>
      )}

      {/* ── CONTENT AREA ───────────────────────────────────────────── */}
      {!slateAlignment.isAligned ? (
        slateAlignment.status === 'OFF_DAY' ? (
          <div className="empty-state schedule-empty-state">
            <Calendar size={36} className="empty-icon text-muted" />
            <div className="empty-title">{slateAlignment.title}</div>
            <div className="empty-subtitle">{slateAlignment.subtitle}</div>
          </div>
        ) : slateAlignment.status === 'STALE' ? (
          <div className="empty-state schedule-empty-state">
            <ShieldAlert size={36} className="empty-icon text-muted" />
            <div className="empty-title">{slateAlignment.title}</div>
            <div className="empty-subtitle">{slateAlignment.subtitle}</div>
          </div>
        ) : slateAlignment.status === 'SYNCING' ? (
          <div className="empty-state schedule-empty-state">
            <Activity size={36} className="empty-icon text-muted" />
            <div className="empty-title">{slateAlignment.title}</div>
            <div className="empty-subtitle">{slateAlignment.subtitle}</div>
          </div>
        ) : slateAlignment.status === 'TODAY_ACTIVE' ? (
          <div className="empty-state schedule-empty-state">
            <Activity size={36} className="empty-icon text-muted" />
            <div className="empty-title">{slateAlignment.title}</div>
            <div className="empty-subtitle">{slateAlignment.subtitle}</div>
          </div>
        ) : slateAlignment.status === 'FUTURE_SLATE' ? (
          <div className="empty-state schedule-empty-state">
            <Calendar size={36} className="empty-icon text-muted" />
            <div className="empty-title">{slateAlignment.title}</div>
            <div className="empty-subtitle">{slateAlignment.subtitle}</div>
          </div>
        ) : (
          <div className="empty-state schedule-empty-state">
            <ShieldAlert size={36} className="empty-icon text-muted" />
            <div className="empty-title">{slateAlignment.title}</div>
            <div className="empty-subtitle">{slateAlignment.subtitle}</div>
          </div>
        )
      ) : sortedPlayers.length === 0 ? (
        <div className="empty-state" style={{ padding: '60px 20px', textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)', letterSpacing: '0.06em' }}>
            No active players available for tomorrow's confirmed slate.
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#555', marginTop: 8 }}>
            Players will appear once available rosters for tomorrow's games are loaded.
          </div>
        </div>
      ) : isSearching && filteredPlayers.length === 0 ? (
        <div className="empty-state search-empty-state">
          <Search size={32} className="search-empty-icon" />
          <div className="search-empty-title">No players found matching "{searchQuery}"</div>
          <div className="search-empty-subtitle">
            Try searching by first name, last name, team, or opponent.
          </div>
          <button
            type="button"
            className="clear-search-action-btn"
            onClick={() => setSearchQuery('')}
          >
            Clear Search
          </button>
        </div>
      ) : (
        <div className="players-grid">
          {filteredPlayers.map(p => (
            <PlayerCard
              key={p.personId}
              player={p}
              onAdd={onAdd}
              onDetail={onDetail}
              slots={slots}
              onToggleLineup={onToggleLineup}
              tomorrowSchedule={tomorrowSchedule}
            />
          ))}
        </div>
      )}
    </div>
  );
}
