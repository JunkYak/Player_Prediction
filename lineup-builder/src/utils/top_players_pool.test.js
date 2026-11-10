import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

/**
 * Top Players Complete Pool & Filter Suite (Phase 2.9)
 *
 * Verifies:
 * 1. Exposes complete player pool with valid predictions.
 * 2. Filters out players without valid prediction records (predicted_rating <= 0 or NaN).
 * 3. Sorts all players by descending ranking_score = round(predicted_rating * (1 + bonus), 2).
 * 4. Preserves raw predicted_rating intact.
 * 5. Schedule issues do not hide predictions; full pool remains available.
 * 6. Search operates across complete pool (name, team, opponent).
 * 7. Active vs scheduled filtering behavior.
 */

function computeRankingScore(p) {
  if (typeof p?.ranking_score === 'number' && !isNaN(p.ranking_score)) {
    return p.ranking_score;
  }
  const rating = Number(p?.predicted_rating) || 0;
  const bonus = Number(p?.season_performance_bonus) || 0;
  return Math.round(rating * (1 + bonus) * 100) / 100;
}

function filterAndSortTopPlayers({
  allPlayers = [],
  filterMode = 'all',
  confirmedTeamIds = null,
  searchQuery = '',
}) {
  // 1. Filter for valid model predictions (> 0)
  const validPool = allPlayers.filter(p => {
    if (!p) return false;
    const r = Number(p.predicted_rating);
    return typeof p.predicted_rating === 'number' && !isNaN(r) && r > 0;
  });

  // 2. Filter by scheduled teams if requested
  const displayedPool = (filterMode === 'scheduled' && confirmedTeamIds)
    ? validPool.filter(p => confirmedTeamIds.has(Number(p.teamId)))
    : validPool;

  // 3. Sort by status_rank -> ranking_score desc -> predicted_rating desc
  const sorted = displayedPool.slice().sort((a, b) => {
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

  // 4. Search query
  if (!searchQuery.trim()) return sorted;
  const q = searchQuery.toLowerCase().trim();
  return sorted.filter(p => {
    const name = (p.playerName || '').toLowerCase();
    const team = (p.teamName || '').toLowerCase();
    const opponent = (p.opponentName || '').toLowerCase();
    return name.includes(q) || team.includes(q) || opponent.includes(q);
  });
}

describe('Top Players Complete Pool & Filtering Suite (Phase 2.9)', () => {
  const MOCK_POOL = [
    {
      personId: 1,
      playerName: 'Onyeka Okongwu',
      teamId: 1610612737,
      teamName: 'Hawks',
      predicted_rating: 2.53,
      season_performance_bonus: 2.4,
      ranking_score: 8.62,
      status: 'Active',
      status_rank: 0,
    },
    {
      personId: 2,
      playerName: 'CJ McCollum',
      teamId: 1610612740,
      teamName: 'Pelicans',
      predicted_rating: 3.21,
      season_performance_bonus: 1.5,
      ranking_score: 8.02,
      status: 'Active',
      status_rank: 0,
    },
    {
      personId: 3,
      playerName: 'Bench Player With No Prediction',
      teamId: 1610612740,
      teamName: 'Pelicans',
      predicted_rating: 0.0,
      season_performance_bonus: 0.0,
      ranking_score: 0.0,
      status: 'Active',
      status_rank: 0,
    },
    {
      personId: 4,
      playerName: 'Unranked Roster Member',
      teamId: 1610612737,
      teamName: 'Hawks',
      predicted_rating: null,
      season_performance_bonus: null,
      status: 'Active',
      status_rank: 0,
    },
    {
      personId: 5,
      playerName: 'Bam Adebayo',
      teamId: 1610612748,
      teamName: 'Heat',
      predicted_rating: 4.14,
      season_performance_bonus: 1.6,
      ranking_score: 10.76,
      status: 'Active',
      status_rank: 0,
      opponentName: 'Knicks',
    },
    {
      personId: 6,
      playerName: 'Injured Star',
      teamId: 1610612752,
      teamName: 'Knicks',
      predicted_rating: 4.50,
      season_performance_bonus: 1.0,
      ranking_score: 9.00,
      status: 'Out',
      status_rank: 4,
      opponentName: 'Heat',
    },
  ];

  it('1. Filters out players without valid prediction records (> 0)', () => {
    const result = filterAndSortTopPlayers({ allPlayers: MOCK_POOL });
    // Bench Player (0.0) and Unranked (null) must be excluded
    const ids = result.map(p => p.personId);
    assert.ok(!ids.includes(3));
    assert.ok(!ids.includes(4));
    assert.equal(result.length, 4);
  });

  it('2. Sorts by ranking_score descending with active availability precedence', () => {
    const result = filterAndSortTopPlayers({ allPlayers: MOCK_POOL });
    // Bam Adebayo (10.76, Active) -> Onyeka Okongwu (8.62, Active) -> CJ McCollum (8.02, Active) -> Injured Star (9.00, Out)
    assert.equal(result[0].personId, 5); // Bam (10.76)
    assert.equal(result[1].personId, 1); // Onyeka (8.62)
    assert.equal(result[2].personId, 2); // CJ McCollum (8.02)
    assert.equal(result[3].personId, 6); // Injured Star (9.00, but Out)
  });

  it('3. Raw predicted_rating is preserved exactly and not mutated', () => {
    const result = filterAndSortTopPlayers({ allPlayers: MOCK_POOL });
    const bam = result.find(p => p.personId === 5);
    assert.equal(bam.predicted_rating, 4.14);
    assert.equal(bam.season_performance_bonus, 1.6);
    assert.equal(bam.ranking_score, 10.76);
  });

  it('4. Allows filtering to scheduled teams without hiding other players from full pool', () => {
    const confirmedTeamIds = new Set([1610612748, 1610612752]); // Heat & Knicks
    const allResult = filterAndSortTopPlayers({ allPlayers: MOCK_POOL, filterMode: 'all' });
    const scheduledResult = filterAndSortTopPlayers({
      allPlayers: MOCK_POOL,
      filterMode: 'scheduled',
      confirmedTeamIds,
    });

    assert.equal(allResult.length, 4); // Full pool
    assert.equal(scheduledResult.length, 2); // Only Heat & Knicks (Bam & Injured Star)
    assert.ok(scheduledResult.every(p => confirmedTeamIds.has(p.teamId)));
  });

  it('5. Search operates across complete pool by player name, team, or opponent', () => {
    const searchByName = filterAndSortTopPlayers({ allPlayers: MOCK_POOL, searchQuery: 'mccollum' });
    assert.equal(searchByName.length, 1);
    assert.equal(searchByName[0].playerName, 'CJ McCollum');

    const searchByTeam = filterAndSortTopPlayers({ allPlayers: MOCK_POOL, searchQuery: 'hawks' });
    assert.equal(searchByTeam.length, 1);
    assert.equal(searchByTeam[0].playerName, 'Onyeka Okongwu');

    const searchByOpponent = filterAndSortTopPlayers({ allPlayers: MOCK_POOL, searchQuery: 'knicks' });
    assert.equal(searchByOpponent.length, 2); // Bam (vs Knicks) and Injured Star (Knicks)
  });
});

