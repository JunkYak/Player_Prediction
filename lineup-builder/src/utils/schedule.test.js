import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  NBA_TIMEZONE,
  getETDateString,
  getTomorrowET,
  getTodayET,
  formatDisplayDate,
  evaluateScheduleStatus,
  filterTeamsPlayingTomorrow,
} from './schedule.js';

// Sample Mock Teams
const MOCK_TEAMS = [
  { teamId: 1610612738, name: 'Celtics', teamName: 'Celtics', fullName: 'Boston Celtics', city: 'Boston', tricode: 'BOS' },
  { teamId: 1610612752, name: 'Knicks', teamName: 'Knicks', fullName: 'New York Knicks', city: 'New York', tricode: 'NYK' },
  { teamId: 1610612747, name: 'Lakers', teamName: 'Lakers', fullName: 'Los Angeles Lakers', city: 'Los Angeles', tricode: 'LAL' },
  { teamId: 1610612743, name: 'Nuggets', teamName: 'Nuggets', fullName: 'Denver Nuggets', city: 'Denver', tricode: 'DEN' },
  { teamId: 1610612744, name: 'Warriors', teamName: 'Warriors', fullName: 'Golden State Warriors', city: 'Golden State', tricode: 'GSW' },
];

describe('Playing Tomorrow Schedule & Date-Handling Suite', () => {
  // Reference date: 2026-10-28 at 14:00:00 UTC (10:00 AM ET)
  const FIXED_NOW = new Date('2026-10-28T14:00:00Z');
  // In ET, today is 2026-10-28, tomorrow is 2026-10-29.
  const TODAY_ET = '2026-10-28';
  const TOMORROW_ET = '2026-10-29';
  const YESTERDAY_ET = '2026-10-27';
  const FUTURE_ET = '2026-10-31';

  // ── TEST 1: Games scheduled for tomorrow are included ────────────────────
  it('1. Games scheduled for tomorrow are included with status CONFIRMED', () => {
    const homeData = [
      { personId: 1, playerName: 'Jayson Tatum', teamId: 1610612738, teamName: 'Celtics', opponentName: 'Knicks', last_game: '2026-10-27' },
      { personId: 2, playerName: 'Jalen Brunson', teamId: 1610612752, teamName: 'Knicks', opponentName: 'Celtics', last_game: '2026-10-26' },
    ];

    const status = evaluateScheduleStatus({
      gameDate: TOMORROW_ET,
      homeData,
      referenceDate: FIXED_NOW,
    });

    assert.equal(status.status, 'CONFIRMED');
    assert.equal(status.targetDate, TOMORROW_ET);

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      homeData,
      scheduleStatus: status,
    });

    assert.equal(result.count, 2);
    const teamIds = result.qualifyingTeams.map(t => t.teamId);
    assert.ok(teamIds.includes(1610612738)); // Celtics
    assert.ok(teamIds.includes(1610612752)); // Knicks
    assert.equal(result.matchups.length, 1);
    assert.equal(result.matchups[0], 'Celtics vs Knicks');
  });

  // ── TEST 2: Games scheduled for today are excluded ───────────────────────
  it('2. Games scheduled for today are excluded from tomorrow filter (status TODAY_ACTIVE)', () => {
    const homeData = [
      { personId: 1, playerName: 'LeBron James', teamId: 1610612747, teamName: 'Lakers', opponentName: 'Nuggets' },
    ];

    const status = evaluateScheduleStatus({
      gameDate: TODAY_ET, // Today's date
      homeData,
      referenceDate: FIXED_NOW,
    });

    assert.equal(status.status, 'TODAY_ACTIVE');
    assert.equal(status.targetDate, TOMORROW_ET);
    assert.equal(status.slateDate, TODAY_ET);

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      homeData,
      scheduleStatus: status,
    });

    assert.equal(result.count, 0);
    assert.equal(result.qualifyingTeams.length, 0);
    assert.equal(result.matchups.length, 0);
  });

  // ── TEST 3: Games scheduled for yesterday are excluded ───────────────────
  it('3. Games scheduled for yesterday are excluded as STALE', () => {
    const homeData = [
      { personId: 1, playerName: 'Stephen Curry', teamId: 1610612744, teamName: 'Warriors', opponentName: 'Lakers' },
    ];

    const status = evaluateScheduleStatus({
      gameDate: YESTERDAY_ET, // Yesterday's date
      homeData,
      referenceDate: FIXED_NOW,
    });

    assert.equal(status.status, 'STALE');
    assert.equal(status.targetDate, TOMORROW_ET);
    assert.equal(status.slateDate, YESTERDAY_ET);

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      homeData,
      scheduleStatus: status,
    });

    assert.equal(result.count, 0);
    assert.equal(result.qualifyingTeams.length, 0);
  });

  // ── TEST 4: Games on a different future date are excluded ────────────────
  it('4. Games on a different future date are excluded (status FUTURE_SLATE)', () => {
    const homeData = [
      { personId: 1, playerName: 'Nikola Jokic', teamId: 1610612743, teamName: 'Nuggets', opponentName: 'Warriors' },
    ];

    const status = evaluateScheduleStatus({
      gameDate: FUTURE_ET, // Further in future than tomorrow
      homeData,
      referenceDate: FIXED_NOW,
    });

    assert.equal(status.status, 'FUTURE_SLATE');
    assert.equal(status.targetDate, TOMORROW_ET);
    assert.equal(status.slateDate, FUTURE_ET);

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      homeData,
      scheduleStatus: status,
    });

    assert.equal(result.count, 0);
    assert.equal(result.qualifyingTeams.length, 0);
  });

  // ── TEST 5: Both home and away teams appear ───────────────────────────────
  it('5. Both home and away teams appear, even if only one team has players in homeData', () => {
    // Only Celtics players present in homeData, but opponent is Knicks
    const homeData = [
      { personId: 10, playerName: 'Jaylen Brown', teamId: 1610612738, teamName: 'Celtics', opponentName: 'Knicks' },
    ];

    const status = evaluateScheduleStatus({
      gameDate: TOMORROW_ET,
      homeData,
      referenceDate: FIXED_NOW,
    });

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      homeData,
      scheduleStatus: status,
    });

    // Both Boston (home) and New York (away) must be included
    assert.equal(result.count, 2);
    const qualifyingNames = result.qualifyingTeams.map(t => t.name);
    assert.ok(qualifyingNames.includes('Celtics'));
    assert.ok(qualifyingNames.includes('Knicks'));
    assert.equal(result.opponentMap.get(1610612738), 'Knicks');
    assert.equal(result.opponentMap.get(1610612752), 'Celtics');
  });

  // ── TEST 6: Team participating in multiple games is not duplicated ───────
  it('6. A team participating in multiple entries or games is not duplicated in the team list', () => {
    // Corrupted or duplicate test records
    const homeData = [
      { personId: 1, playerName: 'Player A', teamId: 1610612747, teamName: 'Lakers', opponentName: 'Nuggets' },
      { personId: 2, playerName: 'Player B', teamId: 1610612747, teamName: 'Lakers', opponentName: 'Nuggets' },
      { personId: 3, playerName: 'Player C', teamId: 1610612747, teamName: 'Lakers', opponentName: 'Warriors' },
      { personId: 4, playerName: 'Player D', teamId: 1610612743, teamName: 'Nuggets', opponentName: 'Lakers' },
    ];

    const status = evaluateScheduleStatus({
      gameDate: TOMORROW_ET,
      homeData,
      referenceDate: FIXED_NOW,
    });

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      homeData,
      scheduleStatus: status,
    });

    // Lakers should only appear ONCE in qualifyingTeams
    const lakersCount = result.qualifyingTeams.filter(t => t.teamId === 1610612747).length;
    assert.equal(lakersCount, 1);

    // Denver Nuggets should only appear ONCE
    const nuggetsCount = result.qualifyingTeams.filter(t => t.teamId === 1610612743).length;
    assert.equal(nuggetsCount, 1);
  });

  // ── TEST 7: Empty but valid schedule produces OFF_DAY state ──────────────
  it('7. An empty but valid schedule produces the correct OFF_DAY state', () => {
    // API explicitly says tomorrow has gamesScheduled: false and empty players
    const status = evaluateScheduleStatus({
      gameDate: TOMORROW_ET,
      gamesScheduled: false,
      homeData: [],
      referenceDate: FIXED_NOW,
    });

    assert.equal(status.status, 'OFF_DAY');
    assert.equal(status.targetDate, TOMORROW_ET);
    assert.match(status.message, /0 confirmed matchups/);

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      homeData: [],
      scheduleStatus: status,
    });

    assert.equal(result.count, 0);
    assert.equal(result.status, 'OFF_DAY');
  });

  // ── TEST 8: Missing schedule data does not falsely claim OFF_DAY ──────────
  it('8. Missing schedule data does not falsely claim that tomorrow has no games', () => {
    // Telemetry offline / missing response (nulls and undefined)
    const status = evaluateScheduleStatus({
      gameDate: null,
      gamesScheduled: undefined,
      homeData: [],
      referenceDate: FIXED_NOW,
    });

    assert.equal(status.status, 'UNAVAILABLE');
    assert.notEqual(status.status, 'OFF_DAY');
    assert.match(status.message, /unavailable/i);

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      homeData: [],
      scheduleStatus: status,
    });

    assert.equal(result.count, 0);
    assert.equal(result.status, 'UNAVAILABLE');
  });

  // ── TEST 9: Stale data does not silently appear to be tomorrow schedule ───
  it('9. Stale data (e.g. 2026-10-22) does not silently appear to be tomorrow schedule', () => {
    const STALE_DATE = '2026-10-22';
    const staleHomeData = [
      { personId: 101, playerName: 'Old Player', teamId: 1610612738, teamName: 'Celtics', opponentName: 'Knicks' },
    ];

    const status = evaluateScheduleStatus({
      gameDate: STALE_DATE,
      homeData: staleHomeData,
      referenceDate: FIXED_NOW, // 2026-10-28
    });

    assert.equal(status.status, 'STALE');
    assert.equal(status.slateDate, STALE_DATE);
    assert.equal(status.targetDate, TOMORROW_ET);

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      homeData: staleHomeData,
      scheduleStatus: status,
    });

    // Zero teams should qualify despite homeData having rows
    assert.equal(result.count, 0);
    assert.equal(result.qualifyingTeams.length, 0);
    assert.equal(result.matchups.length, 0);
  });

  // ── TEST 10: Month-end and year-end transitions ──────────────────────────
  it('10. Correctly calculates tomorrow across month-end and year-end boundaries in ET', () => {
    // Case A: October 31 -> November 1
    const oct31 = new Date('2026-10-31T16:00:00Z'); // 12:00 PM ET on Oct 31
    assert.equal(getTodayET(oct31), '2026-10-31');
    assert.equal(getTomorrowET(oct31), '2026-11-01');

    // Case B: December 31 -> January 1 (Year rollover)
    const dec31 = new Date('2026-12-31T20:00:00Z'); // 3:00 PM ET on Dec 31
    assert.equal(getTodayET(dec31), '2026-12-31');
    assert.equal(getTomorrowET(dec31), '2027-01-01');

    // Case C: February 28 in non-leap year (2026) -> March 1
    const feb28 = new Date('2026-02-28T18:00:00Z');
    assert.equal(getTodayET(feb28), '2026-02-28');
    assert.equal(getTomorrowET(feb28), '2026-03-01');

    // Case D: February 28 in leap year (2028) -> February 29
    const leapFeb28 = new Date('2028-02-28T18:00:00Z');
    assert.equal(getTodayET(leapFeb28), '2028-02-28');
    assert.equal(getTomorrowET(leapFeb28), '2028-02-29');

    // Case E: February 29 in leap year (2028) -> March 1
    const leapFeb29 = new Date('2028-02-29T18:00:00Z');
    assert.equal(getTodayET(leapFeb29), '2028-02-29');
    assert.equal(getTomorrowET(leapFeb29), '2028-03-01');
  });

  // ── TEST 11: Timezone boundaries (UTC vs ET date difference) ─────────────
  it('11. Handles timezone boundaries when UTC and ET fall on different calendar dates', () => {
    // 2026-10-29T02:30:00Z
    // In UTC: Date is 2026-10-29 (at 2:30 AM)
    // In ET (UTC-4 in EDT): Date is 2026-10-28 (at 10:30 PM!)
    const lateNightET = new Date('2026-10-29T02:30:00Z');

    assert.equal(NBA_TIMEZONE, 'America/New_York');
    assert.equal(getETDateString(lateNightET), '2026-10-28', 'getETDateString must format to ET calendar date');
    assert.equal(getTodayET(lateNightET), '2026-10-28', 'ET today must be Oct 28 even though UTC is Oct 29');
    assert.equal(getTomorrowET(lateNightET), '2026-10-29', 'ET tomorrow must be Oct 29');

    // Now evaluate status: A slate scheduled for 2026-10-29 is indeed TOMORROW in ET
    const status = evaluateScheduleStatus({
      gameDate: '2026-10-29',
      homeData: [{ teamId: 1610612738, opponentName: 'Knicks' }],
      referenceDate: lateNightET,
    });

    assert.equal(status.status, 'CONFIRMED');
    assert.equal(status.targetDate, '2026-10-29');
  });

  // ── TEST 12: Matchup banner displays target date, not last completed game ─
  it('12. Target date reflects tomorrow and never adopts last_game from player objects', () => {
    const playerWithPastGame = {
      personId: 99,
      playerName: 'Star Player',
      teamId: 1610612738,
      teamName: 'Celtics',
      opponentName: 'Knicks',
      last_game: '2026-10-21', // Previous completed game date
    };

    const status = evaluateScheduleStatus({
      gameDate: TOMORROW_ET, // '2026-10-29'
      homeData: [playerWithPastGame],
      referenceDate: FIXED_NOW,
    });

    assert.equal(status.targetDate, '2026-10-29');
    assert.notEqual(status.targetDate, playerWithPastGame.last_game);
    assert.equal(formatDisplayDate(status.targetDate), 'Oct 29, 2026');
  });

  // ── TEST 13: Dedicated tomorrowSchedule with status CONFIRMED ───────────
  it('13. Dedicated tomorrowSchedule returns CONFIRMED with exact matchups and team_ids', () => {
    const tomorrowSchedule = {
      schedule_date: TOMORROW_ET,
      generated_at: '2026-10-28T12:00:00Z',
      status: 'CONFIRMED',
      games_scheduled: true,
      games_count: 1,
      matchups: [
        {
          game_id: '0022600101',
          home_team_id: 1610612738,
          home_team_name: 'Celtics',
          away_team_id: 1610612747,
          away_team_name: 'Lakers',
        },
      ],
      team_ids: [1610612738, 1610612747],
    };

    const status = evaluateScheduleStatus({
      referenceDate: FIXED_NOW,
      tomorrowSchedule,
    });

    assert.equal(status.status, 'CONFIRMED');
    assert.equal(status.targetDate, TOMORROW_ET);

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      scheduleStatus: status,
      tomorrowSchedule,
    });

    assert.equal(result.count, 2);
    const ids = result.qualifyingTeams.map(t => t.teamId);
    assert.ok(ids.includes(1610612738));
    assert.ok(ids.includes(1610612747));
    assert.equal(result.opponentMap.get(1610612738), 'Lakers');
    assert.equal(result.opponentMap.get(1610612747), 'Celtics');
  });

  // ── TEST 14: Dedicated tomorrowSchedule with status OFF_DAY ─────────────
  it('14. Dedicated tomorrowSchedule with OFF_DAY returns status OFF_DAY and empty teams', () => {
    const tomorrowSchedule = {
      schedule_date: TOMORROW_ET,
      generated_at: '2026-10-28T12:00:00Z',
      status: 'OFF_DAY',
      games_scheduled: false,
      games_count: 0,
      matchups: [],
      team_ids: [],
    };

    const status = evaluateScheduleStatus({
      referenceDate: FIXED_NOW,
      tomorrowSchedule,
    });

    assert.equal(status.status, 'OFF_DAY');

    const result = filterTeamsPlayingTomorrow({
      teams: MOCK_TEAMS,
      scheduleStatus: status,
      tomorrowSchedule,
    });

    assert.equal(result.count, 0);
    assert.equal(result.qualifyingTeams.length, 0);
  });

  // ── TEST 15: Dedicated tomorrowSchedule with STALE date ─────────────────
  it('15. Dedicated tomorrowSchedule with past date is flagged as STALE', () => {
    const tomorrowSchedule = {
      schedule_date: YESTERDAY_ET,
      generated_at: '2026-10-27T12:00:00Z',
      status: 'STALE',
      games_scheduled: false,
      games_count: 0,
      matchups: [],
      team_ids: [],
    };

    const status = evaluateScheduleStatus({
      referenceDate: FIXED_NOW,
      tomorrowSchedule,
    });

    assert.equal(status.status, 'STALE');
  });
});
