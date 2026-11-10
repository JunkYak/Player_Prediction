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

/**
 * Computes the ranking score for a player:
 * ranking_score = round(predicted_rating * (1 + season_performance_bonus), 2)
 */
function computeRankingScore(p) {
  if (typeof p?.ranking_score === 'number' && !isNaN(p.ranking_score)) {
    return p.ranking_score;
  }
  const rating = Number(p?.predicted_rating) || 0;
  const bonus = Number(p?.season_performance_bonus) || 0;
  return Math.round(rating * (1 + bonus) * 100) / 100;
}

// SECTION 2 — TOP PLAYERS (Complete prediction pool sorted by ranking_score descending)
export default function TopPlayersSection({
  players,
  allPlayers,
  homePlayers,
  gameDate = null,
  tomorrowSchedule = null,
  onAdd,
  onDetail,
  slots = [],
  onToggleLineup,
  referenceDate = undefined,
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all'); // 'all' | 'scheduled'

  // 1. Evaluate schedule status deterministically using America/New_York timezone policy
  const scheduleStatus = useMemo(() => {
    return evaluateScheduleStatus({
      gameDate,
      homeData: players,
      tomorrowSchedule,
      referenceDate,
    });
  }, [gameDate, players, tomorrowSchedule, referenceDate]);

  // 2. Evaluate runtime alignment between player slate and confirmed schedule
  const slateAlignment = useMemo(() => {
    const status = scheduleStatus.status;

    // A. Off-day: explicitly 0 games scheduled
    if (status === 'OFF_DAY') {
      return {
        isAligned: false,
        status: 'OFF_DAY',
        title: 'NBA Off-Day: No Games Scheduled for Target Date',
        subtitle: `The official NBA schedule confirms 0 games scheduled for ${formatDisplayDate(scheduleStatus.targetDate)}. All 30 franchises are resting.`,
      };
    }

    // B. Telemetry Stale: schedule date < targetET
    if (status === 'STALE') {
      return {
        isAligned: false,
        status: 'STALE',
        title: 'Schedule Telemetry Stale',
        subtitle: `The latest available snapshot is from ${formatDisplayDate(scheduleStatus.slateDate)}. The confirmed schedule for ${formatDisplayDate(scheduleStatus.targetDate)} has not yet been synced by the ingestion pipeline.`,
      };
    }

    // C. Telemetry Unavailable / Fetch Failed
    if (status === 'UNAVAILABLE') {
      return {
        isAligned: false,
        status: 'UNAVAILABLE',
        title: 'Schedule Telemetry Unavailable',
        subtitle: `Unable to retrieve schedule telemetry from the prediction service for ${formatDisplayDate(scheduleStatus.targetDate)}.`,
      };
    }

    // D. Today active
    if (status === 'TODAY_ACTIVE') {
      return {
        isAligned: false,
        status: 'TODAY_ACTIVE',
        title: "Target Date Schedule Not Yet Active",
        subtitle: `The active prediction slate is tracking today's games (${formatDisplayDate(scheduleStatus.todayDate)}). The schedule for ${formatDisplayDate(scheduleStatus.targetDate)} will publish once today's games conclude.`,
      };
    }

    // E. Future slate
    if (status === 'FUTURE_SLATE') {
      return {
        isAligned: false,
        status: 'FUTURE_SLATE',
        title: 'Future Slate Telemetry',
        subtitle: `The next available prediction slate is scheduled for ${formatDisplayDate(scheduleStatus.slateDate)}, not the current target date (${formatDisplayDate(scheduleStatus.targetDate)}).`,
      };
    }

    // F. Confirmed: Verify player slate matches schedule date
    if (status === 'CONFIRMED') {
      const schedDate = tomorrowSchedule?.schedule_date;
      if (gameDate && schedDate && gameDate !== schedDate) {
        return {
          isAligned: false,
          status: 'SYNCING',
          title: "Player Slate Syncing",
          subtitle: `Active player predictions (${formatDisplayDate(gameDate)}) are not yet synchronized with confirmed schedule (${formatDisplayDate(schedDate)}). Awaiting pipeline sync.`,
        };
      }
      return { isAligned: true, status: 'CONFIRMED' };
    }

    return { isAligned: true, status };
  }, [scheduleStatus, tomorrowSchedule, gameDate]);

  // Set of confirmed team IDs to identify scheduled teams
  const confirmedTeamIds = useMemo(() => {
    if (tomorrowSchedule && Array.isArray(tomorrowSchedule.team_ids) && tomorrowSchedule.team_ids.length > 0) {
      return new Set(tomorrowSchedule.team_ids.map(Number));
    }
    return null;
  }, [tomorrowSchedule]);

  // 3. Expose complete available prediction pool (every player with valid prediction > 0)
  const fullPredictionPool = useMemo(() => {
    const source = (Array.isArray(allPlayers) && allPlayers.length > 0)
      ? allPlayers
      : (Array.isArray(players) ? players : []);

    return source.filter(p => {
      if (!p) return false;
      const r = Number(p.predicted_rating);
      return typeof p.predicted_rating === 'number' && !isNaN(r) && r > 0;
    });
  }, [allPlayers, players]);

  // Scheduled count
  const scheduledCount = useMemo(() => {
    if (confirmedTeamIds) {
      return fullPredictionPool.filter(p => confirmedTeamIds.has(Number(p.teamId))).length;
    }
    if (Array.isArray(homePlayers) && homePlayers.length > 0) {
      return homePlayers.length;
    }
    return 0;
  }, [fullPredictionPool, confirmedTeamIds, homePlayers]);

  // 4. Select display pool based on activeFilter
  const poolToDisplay = useMemo(() => {
    if (activeFilter === 'scheduled') {
      if (confirmedTeamIds) {
        return fullPredictionPool.filter(p => confirmedTeamIds.has(Number(p.teamId)));
      }
      if (Array.isArray(homePlayers) && homePlayers.length > 0) {
        return homePlayers;
      }
    }
    return fullPredictionPool;
  }, [fullPredictionPool, activeFilter, confirmedTeamIds, homePlayers]);

  // 5. Sort players: status_rank -> ranking_score desc -> predicted_rating desc
  const sortedPlayers = useMemo(() => {
    return poolToDisplay.slice().sort((a, b) => {
      const aRank = typeof a.status_rank === 'number' ? a.status_rank : 5;
      const bRank = typeof b.status_rank === 'number' ? b.status_rank : 5;
      if (aRank !== bRank) return aRank - bRank;

      const aScore = computeRankingScore(a);
      const bScore = computeRankingScore(b);
      if (Math.abs(bScore - aScore) > 1e-6) return bScore - aScore;

      const aRating = typeof a.predicted_rating === 'number' ? a.predicted_rating : 0;
      const bRating = typeof b.predicted_rating === 'number' ? b.predicted_rating : 0;
      if (Math.abs(bRating - aRating) > 1e-6) return bRating - aRating;

      return (a.playerName || '').localeCompare(b.playerName || '');
    });
  }, [poolToDisplay]);

  // 6. Filter matching search query across complete pool
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
  if (sortedPlayers.length === 0) {
    subtitle = `No player predictions available in the serving dataset`;
  } else if (isSearching) {
    subtitle = `Showing ${filteredPlayers.length} of ${sortedPlayers.length} players matching "${searchQuery}"`;
  } else if (activeFilter === 'scheduled') {
    subtitle = `${sortedPlayers.length} active players across confirmed games (${formatDisplayDate(scheduleStatus.targetDate)}) — sorted by availability & ranking score`;
  } else {
    subtitle = `${sortedPlayers.length} players with valid model predictions across all 30 NBA rosters — sorted by availability & ranking score`;
  }

  return (
    <div className="top-players-container">
      {/* ── HEADER ─────────────────────────────────────────────────── */}
      <div className="section-header">
        <div className="section-title">Top Players</div>
        <div className="section-subtitle">{subtitle}</div>
      </div>

      <hr className="section-divider" />

      {/* ── SCHEDULE STATUS NOTICE (Distinguishes schedule issues without hiding predictions) ── */}
      {!slateAlignment.isAligned && (
        <div
          className="schedule-alert-banner"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '10px 14px',
            marginBottom: 16,
            background: 'rgba(255, 107, 0, 0.08)',
            border: '1px solid rgba(255, 107, 0, 0.25)',
            borderRadius: 4,
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
          }}
        >
          {slateAlignment.status === 'OFF_DAY' ? (
            <Calendar size={20} style={{ color: 'var(--accent)', flexShrink: 0 }} />
          ) : slateAlignment.status === 'SYNCING' ? (
            <Activity size={20} style={{ color: 'var(--accent)', flexShrink: 0 }} />
          ) : (
            <ShieldAlert size={20} style={{ color: 'var(--accent)', flexShrink: 0 }} />
          )}
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 2 }}>
              {slateAlignment.title}
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '10.5px', lineHeight: 1.4 }}>
              {slateAlignment.subtitle}
            </div>
          </div>
        </div>
      )}

      {/* ── FILTER TABS & SEARCH TOOLBAR ───────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
        {confirmedTeamIds && scheduledCount > 0 && (
          <div className="players-filter-tabs" style={{ display: 'flex', gap: 8 }}>
            <button
              type="button"
              className={`players-filter-pill ${activeFilter === 'all' ? 'active' : ''}`}
              onClick={() => setActiveFilter('all')}
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                fontWeight: 600,
                padding: '4px 10px',
                borderRadius: 3,
                background: activeFilter === 'all' ? 'rgba(255, 107, 0, 0.15)' : 'rgba(255, 255, 255, 0.04)',
                border: `1px solid ${activeFilter === 'all' ? 'rgba(255, 107, 0, 0.4)' : 'rgba(255, 255, 255, 0.1)'}`,
                color: activeFilter === 'all' ? 'var(--accent)' : 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              All Predictions ({fullPredictionPool.length})
            </button>
            <button
              type="button"
              className={`players-filter-pill ${activeFilter === 'scheduled' ? 'active' : ''}`}
              onClick={() => setActiveFilter('scheduled')}
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                fontWeight: 600,
                padding: '4px 10px',
                borderRadius: 3,
                background: activeFilter === 'scheduled' ? 'rgba(255, 107, 0, 0.15)' : 'rgba(255, 255, 255, 0.04)',
                border: `1px solid ${activeFilter === 'scheduled' ? 'rgba(255, 107, 0, 0.4)' : 'rgba(255, 255, 255, 0.1)'}`,
                color: activeFilter === 'scheduled' ? 'var(--accent)' : 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              Playing Scheduled Slate ({scheduledCount})
            </button>
          </div>
        )}

        {/* ── SEARCH INPUT TOOLBAR ───────────────────────────────────── */}
        {sortedPlayers.length > 0 && (
          <div className="players-search-toolbar" style={{ margin: 0, flex: 1, minWidth: 260, justifyContent: 'flex-end' }}>
            <div className="players-search-box" style={{ maxWidth: 360, width: '100%' }}>
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
      </div>

      {/* ── CONTENT AREA ───────────────────────────────────────────── */}
      {sortedPlayers.length === 0 ? (
        <div className="empty-state" style={{ padding: '60px 20px', textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)', letterSpacing: '0.06em' }}>
            No player predictions available in current dataset.
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#555', marginTop: 8 }}>
            Player predictions will appear once model inference outputs are published.
          </div>
        </div>
      ) : isSearching && filteredPlayers.length === 0 ? (
        <div className="empty-state search-empty-state">
          <Search size={32} className="search-empty-icon" />
          <div className="search-empty-title">No players found matching "{searchQuery}"</div>
          <div className="search-empty-subtitle">
            Try searching by first name, last name, team, or opponent across the complete pool.
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
