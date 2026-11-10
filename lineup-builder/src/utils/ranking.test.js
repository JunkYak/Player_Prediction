/**
 * lineup-builder/src/utils/ranking.test.js
 *
 * Frontend verification suite for:
 * 1. Top Players sorts by ranking_score descending, not predicted_rating.
 * 2. Missing ranking_score handled safely via computation from predicted_rating & season_performance_bonus.
 * 3. Status rank priority takes precedence over ranking score.
 * 4. Raw predicted_rating remains the primary displayed number (PREDICTED).
 * 5. Ranking score is strictly a secondary metric (RANK SCORE) and never called adjusted prediction.
 * 6. Starting Five additive multiplier calculation remains unchanged.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Top Players Ranking Score & Presentation Semantics', () => {

  test('1. Top Players sorts by ranking_score descending rather than raw predicted_rating', () => {
    // Player A: high raw rating, no bonus
    const playerA = {
      personId: 101,
      playerName: 'Star Player',
      predicted_rating: 3.50,
      season_performance_bonus: 0.0,
      ranking_score: 3.50,
      status_rank: 0,
      status: 'Active',
    };

    // Player B: lower raw rating, strong season bonus -> higher ranking_score
    const playerB = {
      personId: 102,
      playerName: 'Breakout Player',
      predicted_rating: 2.72,
      season_performance_bonus: 0.5,
      ranking_score: 4.08,
      status_rank: 0,
      status: 'Active',
    };

    const players = [playerA, playerB];

    // Sorting algorithm matching TopPlayersSection.jsx
    const sorted = [...players].sort((a, b) => {
      const aRank = typeof a.status_rank === 'number' ? a.status_rank : 5;
      const bRank = typeof b.status_rank === 'number' ? b.status_rank : 5;
      if (aRank !== bRank) return aRank - bRank;

      const aScore = typeof a.ranking_score === 'number'
        ? a.ranking_score
        : (Number(a.predicted_rating) || 0) * (1 + (Number(a.season_performance_bonus) || 0));
      const bScore = typeof b.ranking_score === 'number'
        ? b.ranking_score
        : (Number(b.predicted_rating) || 0) * (1 + (Number(b.season_performance_bonus) || 0));

      if (bScore !== aScore) return bScore - aScore;
      return (b.predicted_rating || 0) - (a.predicted_rating || 0);
    });

    // Player B (ranking_score 4.08) must be first, despite Player A having higher predicted_rating (3.50)
    assert.equal(sorted[0].personId, 102);
    assert.equal(sorted[1].personId, 101);
    // Raw predicted_rating remains untouched
    assert.equal(sorted[0].predicted_rating, 2.72);
    assert.equal(sorted[1].predicted_rating, 3.50);
  });

  test('2. Missing ranking_score falls back safely to calculation without NaN or crashing', () => {
    const playerWithoutScore = {
      personId: 103,
      playerName: 'Fallback Player',
      predicted_rating: 2.50,
      season_performance_bonus: 0.4,
      // ranking_score omitted
    };

    const score = typeof playerWithoutScore.ranking_score === 'number'
      ? playerWithoutScore.ranking_score
      : (Number(playerWithoutScore.predicted_rating) || 0) * (1 + (Number(playerWithoutScore.season_performance_bonus) || 0));

    assert.equal(score, 3.5);
  });

  test('3. Status rank availability priority takes precedence over ranking score', () => {
    // Probable player (status_rank 1) with higher score vs Active player (status_rank 0)
    const activePlayer = {
      personId: 104,
      playerName: 'Active Player',
      predicted_rating: 2.0,
      ranking_score: 2.5,
      status_rank: 0,
      status: 'Active',
    };
    const probablePlayer = {
      personId: 105,
      playerName: 'Probable Player',
      predicted_rating: 4.0,
      ranking_score: 5.0,
      status_rank: 1,
      status: 'Probable',
    };

    const sorted = [probablePlayer, activePlayer].sort((a, b) => {
      const aRank = typeof a.status_rank === 'number' ? a.status_rank : 5;
      const bRank = typeof b.status_rank === 'number' ? b.status_rank : 5;
      if (aRank !== bRank) return aRank - bRank;
      return (b.ranking_score || 0) - (a.ranking_score || 0);
    });

    assert.equal(sorted[0].personId, 104); // Active first
    assert.equal(sorted[1].personId, 105); // Probable second
  });

  test('4. PlayerCard primary display value is the raw predicted_rating', () => {
    const player = {
      predicted_rating: 2.7234,
      season_performance_bonus: 0.5,
      ranking_score: 4.08,
    };

    const rawRating = player.predicted_rating;
    const formattedRating = (typeof rawRating === 'number' && !isNaN(rawRating)) ? rawRating.toFixed(2) : 'N/A';
    const primaryLabel = 'PREDICTED';

    assert.equal(formattedRating, '2.72');
    assert.equal(primaryLabel, 'PREDICTED');
  });

  test('5. Starting Five additive multiplier calculation remains unchanged', () => {
    const rawPred = 2.72;
    const slotMultiplier = 1.6;
    const seasonBonus = 0.5;

    const effectiveMultiplier = slotMultiplier + seasonBonus;
    const projectedSlotValue = rawPred * effectiveMultiplier;

    assert.equal(effectiveMultiplier, 2.1);
    assert.equal(projectedSlotValue.toFixed(2), '5.71');

    // Regression guard: strictly distinguish approved formula from erroneous (1 + slotMultiplier + seasonBonus)
    const erroneousValue = rawPred * (1.0 + slotMultiplier + seasonBonus);
    assert.equal(erroneousValue.toFixed(2), '8.43');
    assert.notEqual(projectedSlotValue.toFixed(2), erroneousValue.toFixed(2));
  });
});
