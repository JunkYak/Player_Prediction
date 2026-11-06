/**
 * api.js
 *
 * Lightweight API client connecting React to the FastAPI serving layer.
 * Replaces static JSON file fetching with dynamic REST endpoints.
 */

const envBase = import.meta.env.VITE_API_BASE_URL;
const API_BASE_URL = (
  envBase !== undefined ? envBase : 'http://localhost:8000'
).replace(/\/+$/, '');

export { API_BASE_URL };

/**
 * Fetch the active draft slate players.
 * Maps to GET /api/v1/players
 *
 * In offseason / zero-games state:
 * returns { game_date: null, generated_at: string, games_scheduled: false, players: [] }
 */
export async function fetchPlayers() {
  const url = `${API_BASE_URL}/api/v1/players`;
  const response = await fetch(url);
  if (!response.ok) {
    let errorDetail = `Failed to fetch players (HTTP ${response.status})`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = errorJson.detail;
      }
    } catch {
      // response was not JSON
    }
    throw new Error(errorDetail);
  }
  const data = await response.json();
  if (!data || typeof data !== 'object') {
    throw new Error('Malformed API response: expected JSON object from /api/v1/players');
  }
  return {
    game_date: data.game_date || null,
    generated_at: data.generated_at || null,
    games_scheduled: Boolean(data.games_scheduled),
    players: Array.isArray(data.players) ? data.players : [],
  };
}

/**
 * Fetch league-wide roster predictions for all 30 teams.
 * Maps to GET /api/v1/teams
 *
 * returns { generated_at: string, total_players: number, players: [...] }
 */
export async function fetchTeams() {
  const url = `${API_BASE_URL}/api/v1/teams`;
  const response = await fetch(url);
  if (!response.ok) {
    let errorDetail = `Failed to fetch teams (HTTP ${response.status})`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = errorJson.detail;
      }
    } catch {
      // response was not JSON
    }
    throw new Error(errorDetail);
  }
  const data = await response.json();
  if (!data || typeof data !== 'object') {
    throw new Error('Malformed API response: expected JSON object from /api/v1/teams');
  }
  const players = Array.isArray(data.players) ? data.players : [];
  return {
    generated_at: data.generated_at || null,
    total_players: typeof data.total_players === 'number' ? data.total_players : players.length,
    players,
  };
}

/**
 * Fetch tomorrow's NBA game schedule.
 * Maps to GET /api/v1/schedule/tomorrow
 *
 * returns {
 *   schedule_date: string|null,
 *   generated_at: string|null,
 *   status: string,
 *   games_scheduled: boolean,
 *   games_count: number,
 *   matchups: Array<object>,
 *   team_ids: Array<number>
 * }
 */
export async function fetchTomorrowSchedule() {
  const url = `${API_BASE_URL}/api/v1/schedule/tomorrow`;
  const response = await fetch(url);
  if (!response.ok) {
    let errorDetail = `Failed to fetch tomorrow schedule (HTTP ${response.status})`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = errorJson.detail;
      }
    } catch {
      // response was not JSON
    }
    throw new Error(errorDetail);
  }
  const data = await response.json();
  if (!data || typeof data !== 'object') {
    throw new Error('Malformed API response: expected JSON object from /api/v1/schedule/tomorrow');
  }
  return {
    schedule_date: data.schedule_date || null,
    generated_at: data.generated_at || null,
    status: data.status || 'UNAVAILABLE',
    games_scheduled: Boolean(data.games_scheduled),
    games_count: typeof data.games_count === 'number' ? data.games_count : 0,
    matchups: Array.isArray(data.matchups) ? data.matchups : [],
    team_ids: Array.isArray(data.team_ids) ? data.team_ids : [],
  };
}
