import { useState, useMemo } from 'react';
import { Search, X, Users } from 'lucide-react';
import PlayerCard from './PlayerCard';

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

// SECTION 2 — TOP PLAYERS (from GET /api/v1/players, no OUT players)
export default function TopPlayersSection({
  players,
  onAdd,
  onDetail,
  slots = [],
  onToggleLineup,
}) {
  const [searchQuery, setSearchQuery] = useState('');

  // 1. Filter out OUT players and sort by status_rank -> adjusted_predicted_rating desc -> predicted_rating desc
  const sortedPlayers = useMemo(() => {
    return (Array.isArray(players) ? players : [])
      .filter(p => p && p.status !== 'Out')
      .sort((a, b) => {
        const aRank = typeof a.status_rank === 'number' ? a.status_rank : 5;
        const bRank = typeof b.status_rank === 'number' ? b.status_rank : 5;
        if (aRank !== bRank) return aRank - bRank;
        const aAdj = typeof a.adjusted_predicted_rating === 'number' ? a.adjusted_predicted_rating : (a.predicted_rating || 0);
        const bAdj = typeof b.adjusted_predicted_rating === 'number' ? b.adjusted_predicted_rating : (b.predicted_rating || 0);
        if (bAdj !== aAdj) return bAdj - aAdj;
        const aRating = typeof a.predicted_rating === 'number' ? a.predicted_rating : 0;
        const bRating = typeof b.predicted_rating === 'number' ? b.predicted_rating : 0;
        return bRating - aRating;
      });
  }, [players]);

  // 2. Filter matching search query (case-insensitive, accent-tolerant, partial-match)
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

  const noGames = !Array.isArray(players) || players.length === 0;
  const isSearching = searchQuery.trim().length > 0;

  return (
    <div className="top-players-container">
      {/* ── HEADER ─────────────────────────────────────────────────── */}
      <div className="section-header">
        <div className="section-title">Top Players</div>
        <div className="section-subtitle">
          {noGames
            ? "No NBA games scheduled for tomorrow"
            : isSearching
            ? `Showing ${filteredPlayers.length} of ${sortedPlayers.length} players matching "${searchQuery}"`
            : `${sortedPlayers.length} active players across tomorrow's games — sorted by availability & predicted rating`}
        </div>
      </div>

      <hr className="section-divider" />

      {/* ── SEARCH INPUT TOOLBAR ───────────────────────────────────── */}
      {!noGames && (
        <div className="players-search-toolbar">
          <div className="players-search-box">
            <Search size={15} className="search-icon" aria-hidden="true" />
            <input
              type="text"
              className="players-search-input"
              placeholder="Search players by name..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              aria-label="Search players by name"
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
      {noGames ? (
        <div className="empty-state" style={{ padding: '60px 20px', textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)', letterSpacing: '0.06em' }}>
            No NBA games scheduled for tomorrow.
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#555', marginTop: 8 }}>
            The draft slate will activate automatically when upcoming matchups are scheduled.
          </div>
        </div>
      ) : sortedPlayers.length === 0 ? (
        <div className="empty-state">No active players available for this slate</div>
      ) : isSearching && filteredPlayers.length === 0 ? (
        <div className="empty-state search-empty-state">
          <Search size={32} className="search-empty-icon" />
          <div className="search-empty-title">No players found matching "{searchQuery}"</div>
          <div className="search-empty-subtitle">
            Try searching by first name, last name, or team name.
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
            />
          ))}
        </div>
      )}
    </div>
  );
}
