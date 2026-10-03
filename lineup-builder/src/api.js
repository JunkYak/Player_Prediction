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
