/**
 * src/utils/schedule.js
 *
 * Deterministic, timezone-aware NBA schedule and date calculation utilities.
 *
 * TIMEZONE POLICY:
 * Official NBA game schedules and slates are strictly defined by US Eastern Time
 * ('America/New_York', ET). All calendar date boundaries for 'today' and 'tomorrow'
 * are computed relative to America/New_York to ensure exact alignment with NBA
 * scheduling standards, avoiding premature slate shifts caused by UTC or local user timezones.
 */

export const NBA_TIMEZONE = 'America/New_York';

/**
 * Returns the calendar date in America/New_York for a given reference date as YYYY-MM-DD.
 * Handles daylight saving time, month-end, and year-end transitions via Intl.DateTimeFormat.
 *
 * @param {Date} [referenceDate=new Date()]
 * @returns {string} YYYY-MM-DD
 */
export function getETDateString(referenceDate = new Date()) {
  const formatter = new Intl.DateTimeFormat('en-CA', {
    timeZone: NBA_TIMEZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
  return formatter.format(referenceDate);
}

/**
 * Calculates tomorrow's calendar date in America/New_York (ET).
 *
 * @param {Date} [referenceDate=new Date()]
 * @returns {string} YYYY-MM-DD for tomorrow in ET
 */
export function getTomorrowET(referenceDate = new Date()) {
  const todayET = getETDateString(referenceDate);
  const [year, month, day] = todayET.split('-').map(Number);

  // Use UTC date arithmetic on year/month/day components to reliably add 1 calendar day
  // without local machine clock or DST interference.
  const nextDay = new Date(Date.UTC(year, month - 1, day + 1));
  const y = nextDay.getUTCFullYear();
  const m = String(nextDay.getUTCMonth() + 1).padStart(2, '0');
  const d = String(nextDay.getUTCDate()).padStart(2, '0');

  return `${y}-${m}-${d}`;
}

/**
 * Calculates today's calendar date in America/New_York (ET).
 *
 * @param {Date} [referenceDate=new Date()]
 * @returns {string} YYYY-MM-DD for today in ET
 */
export function getTodayET(referenceDate = new Date()) {
  return getETDateString(referenceDate);
}

/**
 * Formats a YYYY-MM-DD date string into a readable format (e.g. "Oct 29, 2026").
 *
 * @param {string|null} dateStr - YYYY-MM-DD
 * @returns {string} Formatted date or fallback
 */
export function formatDisplayDate(dateStr) {
  if (!dateStr || typeof dateStr !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
    return dateStr || '';
  }
  const [year, month, day] = dateStr.split('-').map(Number);
  const dt = new Date(Date.UTC(year, month - 1, day));
  return new Intl.DateTimeFormat('en-US', {
    timeZone: 'UTC',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(dt);
}

/**
 * Schedule evaluation status enum:
 * - 'CONFIRMED': Slate matches tomorrow's target date with scheduled games.
 * - 'OFF_DAY': Schedule data is current, but 0 games are scheduled for tomorrow (NBA off-day).
 * - 'TODAY_ACTIVE': Prediction slate is for today's live/upcoming games; tomorrow not yet published.
 * - 'STALE': Schedule date in API is in the past (< todayET).
 * - 'FUTURE_SLATE': Slate date is further out than tomorrow (> tomorrowET).
 * - 'UNAVAILABLE': Schedule data is missing or telemetry is offline.
 *
 * @param {Object} params
 * @param {string|null} params.gameDate - YYYY-MM-DD from API
 * @param {string|null} [params.generatedAt] - Timestamp from API
 * @param {boolean} [params.gamesScheduled] - Boolean flag from API
 * @param {Array} [params.homeData] - Players array for active slate
 * @param {Date} [params.referenceDate=new Date()] - Deterministic clock
 * @returns {Object} { status, targetDate, todayDate, slateDate, message }
 */
export function evaluateScheduleStatus({
  gameDate = null,
  generatedAt = null,
  gamesScheduled = undefined,
  homeData = [],
  referenceDate = new Date(),
  tomorrowSchedule = null,
} = {}) {
  const todayET = getTodayET(referenceDate);
  const tomorrowET = getTomorrowET(referenceDate);

  // If dedicated tomorrowSchedule is provided
  if (tomorrowSchedule && typeof tomorrowSchedule === 'object') {
    const schedDate = tomorrowSchedule.schedule_date || null;
    const schedGen = tomorrowSchedule.generated_at || null;
    const schedStatus = tomorrowSchedule.status || 'UNAVAILABLE';

    if (schedDate && schedDate > tomorrowET) {
      return {
        status: 'FUTURE_SLATE',
        targetDate: tomorrowET,
        todayDate: todayET,
        slateDate: schedDate,
        generatedAt: schedGen,
        message: `Available schedule data is for a future slate (${formatDisplayDate(schedDate)}), not tomorrow (${formatDisplayDate(tomorrowET)}).`,
      };
    }

    if (schedStatus === 'CONFIRMED' && tomorrowSchedule.games_scheduled) {
      return {
        status: 'CONFIRMED',
        targetDate: schedDate || tomorrowET,
        todayDate: todayET,
        slateDate: schedDate,
        generatedAt: schedGen,
        message: `Confirmed games scheduled for tomorrow (${formatDisplayDate(schedDate || tomorrowET)}).`,
      };
    }

    if (schedStatus === 'OFF_DAY' || (schedStatus === 'CONFIRMED' && !tomorrowSchedule.games_scheduled)) {
      return {
        status: 'OFF_DAY',
        targetDate: schedDate || tomorrowET,
        todayDate: todayET,
        slateDate: schedDate,
        generatedAt: schedGen,
        message: `The official NBA schedule has 0 confirmed matchups for tomorrow (${formatDisplayDate(schedDate || tomorrowET)}). All 30 franchises are resting.`,
      };
    }

    if (schedStatus === 'STALE' || (schedDate && schedDate < tomorrowET)) {
      return {
        status: 'STALE',
        targetDate: tomorrowET,
        todayDate: todayET,
        slateDate: schedDate,
        generatedAt: schedGen,
        message: `The latest available snapshot is from ${formatDisplayDate(schedDate)}. Tomorrow's confirmed schedule (${formatDisplayDate(tomorrowET)}) has not yet been synced.`,
      };
    }

    if (schedStatus === 'UNAVAILABLE' || schedStatus === 'FETCH_FAILED') {
      return {
        status: 'UNAVAILABLE',
        targetDate: tomorrowET,
        todayDate: todayET,
        slateDate: schedDate,
        generatedAt: schedGen,
        message: 'Schedule telemetry is currently unavailable or offline.',
      };
    }
  }

  // Missing or uninitialized schedule telemetry
  if (!gameDate && (!Array.isArray(homeData) || homeData.length === 0) && gamesScheduled === undefined) {
    return {
      status: 'UNAVAILABLE',
      targetDate: tomorrowET,
      todayDate: todayET,
      slateDate: null,
      generatedAt: generatedAt || null,
      message: 'Schedule telemetry is currently unavailable or offline.',
    };
  }

  // Stale past date: slate date is earlier than today in ET
  if (gameDate && gameDate < todayET) {
    return {
      status: 'STALE',
      targetDate: tomorrowET,
      todayDate: todayET,
      slateDate: gameDate,
      generatedAt: generatedAt || null,
      message: `The latest available snapshot is from ${formatDisplayDate(gameDate)}. Tomorrow's confirmed schedule (${formatDisplayDate(tomorrowET)}) has not yet been synced.`,
    };
  }

  // Today's active slate: games on today's ET date are still active or upcoming
  if (gameDate === todayET) {
    return {
      status: 'TODAY_ACTIVE',
      targetDate: tomorrowET,
      todayDate: todayET,
      slateDate: gameDate,
      generatedAt: generatedAt || null,
      message: `The active prediction slate is tracking today's games (${formatDisplayDate(todayET)}). Tomorrow's schedule (${formatDisplayDate(tomorrowET)}) will activate once today's slate concludes.`,
    };
  }

  // Future slate beyond tomorrow
  if (gameDate && gameDate > tomorrowET) {
    return {
      status: 'FUTURE_SLATE',
      targetDate: tomorrowET,
      todayDate: todayET,
      slateDate: gameDate,
      generatedAt: generatedAt || null,
      message: `Available schedule data is for a future slate (${formatDisplayDate(gameDate)}), not tomorrow (${formatDisplayDate(tomorrowET)}).`,
    };
  }

  // Exact match with tomorrow's date
  if (gameDate === tomorrowET) {
    const hasGames = gamesScheduled !== false && Array.isArray(homeData) && homeData.length > 0;
    if (!hasGames) {
      return {
        status: 'OFF_DAY',
        targetDate: tomorrowET,
        todayDate: todayET,
        slateDate: gameDate,
        generatedAt: generatedAt || null,
        message: `The official NBA schedule has 0 confirmed matchups for tomorrow (${formatDisplayDate(tomorrowET)}). All 30 franchises are resting.`,
      };
    }

    return {
      status: 'CONFIRMED',
      targetDate: tomorrowET,
      todayDate: todayET,
      slateDate: gameDate,
      generatedAt: generatedAt || null,
      message: `Confirmed games scheduled for tomorrow (${formatDisplayDate(tomorrowET)}).`,
    };
  }

  // If gameDate is null but gamesScheduled is explicitly false (off-season / empty schedule state)
  if (gameDate === null && gamesScheduled === false) {
    return {
      status: 'OFF_DAY',
      targetDate: tomorrowET,
      todayDate: todayET,
      slateDate: null,
      generatedAt: generatedAt || null,
      message: `The official NBA schedule confirms 0 active matchups for tomorrow (${formatDisplayDate(tomorrowET)}).`,
    };
  }

  return {
    status: 'UNAVAILABLE',
    targetDate: tomorrowET,
    todayDate: todayET,
    slateDate: gameDate || null,
    generatedAt: generatedAt || null,
    message: 'Unable to verify tomorrow schedule data.',
  };
}

/**
 * Extracts qualifying teams and unique matchups for the confirmed target date.
 *
 * @param {Object} params
 * @param {Array} params.teams - All 30 teams from teamMap
 * @param {Array} params.homeData - Active draft slate players from /api/v1/players
 * @param {Array} params.allPlayers - League-wide players from /api/v1/teams
 * @param {Object} params.scheduleStatus - Output of evaluateScheduleStatus
 * @returns {Object} { qualifyingTeams, matchups, opponentMap, count, status }
 */
export function filterTeamsPlayingTomorrow({
  teams = [],
  homeData = [],
  allPlayers = [],
  scheduleStatus,
  tomorrowSchedule = null,
}) {
  // If schedule is not confirmed for tomorrow, return empty list
  if (!scheduleStatus || scheduleStatus.status !== 'CONFIRMED') {
    return {
      qualifyingTeams: [],
      matchups: [],
      opponentMap: new Map(),
      count: 0,
      status: scheduleStatus?.status || 'UNAVAILABLE',
    };
  }

  // If dedicated tomorrowSchedule is available with team_ids
  if (tomorrowSchedule && Array.isArray(tomorrowSchedule.team_ids) && tomorrowSchedule.team_ids.length > 0) {
    const scheduledTeamIds = new Set(tomorrowSchedule.team_ids.map(Number));
    const opponentMap = new Map();
    const matchupPairs = new Set();

    (tomorrowSchedule.matchups || []).forEach(m => {
      const homeId = Number(m.home_team_id);
      const awayId = Number(m.away_team_id);
      const homeName = m.home_team_name;
      const awayName = m.away_team_name;

      if (homeId && awayName) {
        opponentMap.set(homeId, awayName);
      }
      if (awayId && homeName) {
        opponentMap.set(awayId, homeName);
      }
      if (homeName && awayName) {
        matchupPairs.add([homeName, awayName].sort().join(' vs '));
      }
    });

    const qualifyingTeams = (Array.isArray(teams) ? teams : []).filter(t =>
      t && t.teamId && scheduledTeamIds.has(Number(t.teamId))
    );

    return {
      qualifyingTeams,
      matchups: Array.from(matchupPairs),
      opponentMap,
      count: qualifyingTeams.length,
      status: 'CONFIRMED',
    };
  }

  const teamIds = new Set();
  const opponentMap = new Map(); // teamId -> opponentName
  const matchupPairs = new Set();

  // Create name/tricode lookup for teams to resolve opponents by name
  const nameToTeam = new Map();
  (Array.isArray(teams) ? teams : []).forEach(t => {
    if (t && t.teamId) {
      if (t.teamName) nameToTeam.set(t.teamName.toLowerCase(), t);
      if (t.fullName) nameToTeam.set(t.fullName.toLowerCase(), t);
      if (t.name) nameToTeam.set(t.name.toLowerCase(), t);
      if (t.tricode) nameToTeam.set(t.tricode.toLowerCase(), t);
    }
  });

  // 1. Extract from homeData (active slate for tomorrow)
  if (Array.isArray(homeData)) {
    homeData.forEach(p => {
      if (!p) return;
      if (p.teamId) {
        teamIds.add(p.teamId);
      }
      if (p.opponentId) {
        teamIds.add(p.opponentId);
      }
      if (p.opponentName && p.opponentName !== '—' && p.opponentName !== '') {
        if (p.teamId) {
          opponentMap.set(p.teamId, p.opponentName);
        }
        // Match opponentName to known team
        const oppTeam = nameToTeam.get(p.opponentName.toLowerCase());
        if (oppTeam) {
          teamIds.add(oppTeam.teamId);
          if (p.teamName && !opponentMap.has(oppTeam.teamId)) {
            opponentMap.set(oppTeam.teamId, p.teamName);
          }
        }
        const homeName = p.teamName || (p.teamId && teams.find(t => t.teamId === p.teamId)?.teamName) || '';
        const pair = [homeName, p.opponentName].sort().join(' vs ');
        if (pair.trim()) {
          matchupPairs.add(pair);
        }
      }
    });
  }

  // 2. Cross-reference with allPlayers for full team representation
  if (Array.isArray(allPlayers)) {
    allPlayers.forEach(p => {
      if (!p) return;
      if (p.teamId && p.opponentName && p.opponentName !== '—' && p.opponentName !== '') {
        // Only include if the team is part of the verified schedule
        if (teamIds.has(p.teamId) || teamIds.has(p.opponentId)) {
          teamIds.add(p.teamId);
          if (!opponentMap.has(p.teamId)) {
            opponentMap.set(p.teamId, p.opponentName);
          }
          const oppTeam = nameToTeam.get(p.opponentName.toLowerCase());
          if (oppTeam) {
            teamIds.add(oppTeam.teamId);
            if (p.teamName && !opponentMap.has(oppTeam.teamId)) {
              opponentMap.set(oppTeam.teamId, p.teamName);
            }
          }
        }
      }
    });
  }

  // Deduplicate and filter full team objects
  const qualifyingTeams = (Array.isArray(teams) ? teams : []).filter(t =>
    t && t.teamId && teamIds.has(t.teamId)
  );

  return {
    qualifyingTeams,
    matchups: Array.from(matchupPairs),
    opponentMap,
    count: qualifyingTeams.length,
    status: 'CONFIRMED',
  };
}

/**
 * Resolves opponent team name for a given teamId from tomorrow's schedule matchups.
 *
 * @param {number|string} teamId
 * @param {Array} matchups
 * @returns {string|null} Opponent team name or null
 */
export function getOpponentFromMatchups(teamId, matchups = []) {
  if (!teamId || !Array.isArray(matchups)) return null;
  const tid = Number(teamId);
  for (const m of matchups) {
    if (Number(m.home_team_id) === tid) return m.away_team_name || null;
    if (Number(m.away_team_id) === tid) return m.home_team_name || null;
  }
  return null;
}
