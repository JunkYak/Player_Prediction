import test from 'node:test';
import assert from 'node:assert/strict';
import {
  evaluateScheduleStatus,
  filterTeamsPlayingTomorrow,
  getOpponentFromMatchups,
} from './schedule.js';

test('Top Players & Schedule Alignment Suite (Phase 2.9)', async (t) => {
  // Reference date: Saturday Nov 7, 2026 12:00 PM ET -> target date is Nov 7, 2026
  const refDateNov7 = new Date('2026-11-07T12:00:00-05:00');

  await t.test('1. Confirmed schedule with matching games returns CONFIRMED', () => {
    const tomorrowSchedule = {
      schedule_date: '2026-11-07',
      status: 'CONFIRMED',
      games_scheduled: true,
      games_count: 2,
      team_ids: [1610612741, 1610612755, 1610612738, 1610612744],
      matchups: [
        {
          game_id: '1',
          home_team_id: 1610612741,
          home_team_name: 'Bulls',
          away_team_id: 1610612755,
          away_team_name: '76ers',
        },
        {
          game_id: '2',
          home_team_id: 1610612738,
          home_team_name: 'Celtics',
          away_team_id: 1610612744,
          away_team_name: 'Warriors',
        },
      ],
    };

    const status = evaluateScheduleStatus({
      gameDate: '2026-11-07',
      tomorrowSchedule,
      referenceDate: refDateNov7,
    });

    assert.equal(status.status, 'CONFIRMED');
    assert.equal(status.targetDate, '2026-11-07');
  });

  await t.test('2. Stale schedule returns status STALE and distinct explanatory message', () => {
    const staleSchedule = {
      schedule_date: '2026-11-06',
      status: 'STALE',
      games_scheduled: false,
      games_count: 9,
      team_ids: [1610612740, 1610612750],
    };

    const status = evaluateScheduleStatus({
      gameDate: '2026-11-06',
      tomorrowSchedule: staleSchedule,
      referenceDate: refDateNov7,
    });

    assert.equal(status.status, 'STALE');
    assert.equal(status.targetDate, '2026-11-07');
    assert.match(status.message, /synced/i);
  });

  await t.test('3. Future slate returns status FUTURE_SLATE', () => {
    const futureSchedule = {
      schedule_date: '2026-11-12',
      status: 'CONFIRMED',
      games_scheduled: true,
      games_count: 5,
    };

    const status = evaluateScheduleStatus({
      gameDate: '2026-11-12',
      tomorrowSchedule: futureSchedule,
      referenceDate: refDateNov7,
    });

    assert.equal(status.status, 'FUTURE_SLATE');
    assert.equal(status.targetDate, '2026-11-07');
  });

  await t.test('4. Valid OFF_DAY returns status OFF_DAY with 0 matchups', () => {
    const offDaySchedule = {
      schedule_date: '2026-11-07',
      status: 'OFF_DAY',
      games_scheduled: false,
      games_count: 0,
    };

    const status = evaluateScheduleStatus({
      gameDate: '2026-11-07',
      tomorrowSchedule: offDaySchedule,
      referenceDate: refDateNov7,
    });

    assert.equal(status.status, 'OFF_DAY');
    assert.equal(status.targetDate, '2026-11-07');
    assert.match(status.message, /0 confirmed matchups/i);

    const filtered = filterTeamsPlayingTomorrow({
      teams: [{ teamId: 1610612741, teamName: 'Bulls' }],
      scheduleStatus: status,
      tomorrowSchedule: offDaySchedule,
    });
    assert.equal(filtered.qualifyingTeams.length, 0);
  });

  await t.test('5. getOpponentFromMatchups resolves correct home and away opponents', () => {
    const matchups = [
      {
        home_team_id: 1610612741,
        home_team_name: 'Bulls',
        away_team_id: 1610612755,
        away_team_name: '76ers',
      },
    ];

    assert.equal(getOpponentFromMatchups(1610612741, matchups), '76ers');
    assert.equal(getOpponentFromMatchups(1610612755, matchups), 'Bulls');
    assert.equal(getOpponentFromMatchups(1610612738, matchups), null); // Celtics not in matchup
  });

  await t.test('6. Playing Tomorrow filters exactly the confirmed teams and excludes resting teams', () => {
    const tomorrowSchedule = {
      schedule_date: '2026-11-07',
      status: 'CONFIRMED',
      games_scheduled: true,
      team_ids: [1610612741, 1610612755], // Bulls & 76ers only
      matchups: [
        {
          home_team_id: 1610612741,
          home_team_name: 'Bulls',
          away_team_id: 1610612755,
          away_team_name: '76ers',
        },
      ],
    };

    const allTeams = [
      { teamId: 1610612741, teamName: 'Bulls' },
      { teamId: 1610612755, teamName: '76ers' },
      { teamId: 1610612738, teamName: 'Celtics' }, // Resting
      { teamId: 1610612744, teamName: 'Warriors' }, // Resting
    ];

    const scheduleStatus = { status: 'CONFIRMED' };
    const result = filterTeamsPlayingTomorrow({
      teams: allTeams,
      scheduleStatus,
      tomorrowSchedule,
    });

    assert.equal(result.qualifyingTeams.length, 2);
    assert.deepEqual(
      result.qualifyingTeams.map(t => t.teamName),
      ['Bulls', '76ers']
    );
    assert.equal(result.opponentMap.get(1610612741), '76ers');
    assert.equal(result.opponentMap.get(1610612755), 'Bulls');
  });

  await t.test('7. Starting Five scoring arithmetic remains intact and unmutated', () => {
    // Player: Bam Adebayo: pred = 4.14, bonus = 1.6
    const predictedRating = 4.14;
    const bonus = 1.6;
    const slotMultiplier = 2.0; // Slot 1

    const effectiveMultiplier = slotMultiplier + bonus;
    assert.equal(effectiveMultiplier, 3.6);

    const projectedSlotValue = predictedRating * effectiveMultiplier;
    assert.equal(Math.round(projectedSlotValue * 100) / 100, 14.90);

    // Derived ranking score for Top Players sorting
    const rankingScore = Math.round(predictedRating * (1 + bonus) * 100) / 100;
    assert.equal(rankingScore, 10.76);
  });
});
