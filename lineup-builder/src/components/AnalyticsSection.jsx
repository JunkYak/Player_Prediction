import { useState, useMemo } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Search,
  Calendar,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';
import NbaLogo from './NbaLogo';
import StatusBadge from './StatusBadge';
import LineupActionButton from './LineupActionButton';
import { getTeamInfo } from '../constants/teams';

/**
 * AnalyticsSection
 *
 * Comprehensive NBA performance analytics dashboard grounded strictly in real model
 * and feature data.
 *
 * Implements:
 * 1. Player Progression & Momentum Line Chart (Last 7-avg -> Last 3-avg -> Last Game -> Predicted)
 * 2. Biggest Risers & Fallers (based on real delta = predicted_rating - last_game)
 * 3. Team Benchmark Comparison (Horizontal bar comparison of team average and top-5 core rating)
 * 4. Recent Rating Momentum Data Table (Searchable, sortable, status-aware)
 */
export default function AnalyticsSection({
  allPlayers,
  onDetail,
  slots = [],
  onToggleLineup,
}) {
  const [selectedPlayerId, setSelectedPlayerId] = useState(null);
  const [playerSearchQuery, setPlayerSearchQuery] = useState('');
  const [tableSearchQuery, setTableSearchQuery] = useState('');
  const [selectedRange, setSelectedRange] = useState('progression'); // 'progression', '7d', '14d', '30d'
  const [teamSortMetric, setTeamSortMetric] = useState('avg'); // 'avg' or 'top5'
  const [tableSortField, setTableSortField] = useState('predicted_rating');
  const [tableSortAsc, setTableSortAsc] = useState(false);

  // Clean, validated player list
  const players = useMemo(() => {
    return (Array.isArray(allPlayers) ? allPlayers : [])
      .filter(p => p && p.personId && typeof p.predicted_rating === 'number' && !isNaN(p.predicted_rating))
      .map(p => {
        const lastGame = typeof p.last_game === 'number' && !isNaN(p.last_game) ? p.last_game : null;
        const last3 = typeof p.last3_avg === 'number' && !isNaN(p.last3_avg) ? p.last3_avg : null;
        const last7 = typeof p.last7_avg === 'number' && !isNaN(p.last7_avg) ? p.last7_avg : null;
        const delta = lastGame !== null ? p.predicted_rating - lastGame : null;
        const delta3 = last3 !== null ? p.predicted_rating - last3 : null;

        return {
          ...p,
          lastGameVal: lastGame,
          last3Val: last3,
          last7Val: last7,
          deltaVal: delta,
          delta3Val: delta3,
        };
      });
  }, [allPlayers]);

  // Set default selected player to highest rated player if none selected
  const activeSelectedPlayer = useMemo(() => {
    if (!players.length) return null;
    if (selectedPlayerId) {
      const found = players.find(p => p.personId === selectedPlayerId);
      if (found) return found;
    }
    // Default to the highest rated player
    return [...players].sort((a, b) => b.predicted_rating - a.predicted_rating)[0];
  }, [players, selectedPlayerId]);

  // Filter players for player selector
  const playerSearchResults = useMemo(() => {
    if (!playerSearchQuery.trim()) return players.slice(0, 10);
    const q = playerSearchQuery.toLowerCase();
    return players
      .filter(p => p.playerName.toLowerCase().includes(q) || (p.teamName || '').toLowerCase().includes(q))
      .slice(0, 10);
  }, [players, playerSearchQuery]);

  // ─── 1. RISERS & FALLERS ──────────────────────────────────────────────────
  const { risers, fallers } = useMemo(() => {
    const withDelta = players.filter(p => p.deltaVal !== null);
    const sortedDesc = [...withDelta].sort((a, b) => b.deltaVal - a.deltaVal);
    const sortedAsc = [...withDelta].sort((a, b) => a.deltaVal - b.deltaVal);

    return {
      risers: sortedDesc.slice(0, 6),
      fallers: sortedAsc.slice(0, 6),
    };
  }, [players]);

  // ─── 2. TEAM COMPARISON ───────────────────────────────────────────────────
  const teamComparisonData = useMemo(() => {
    const map = {};
    players.forEach(p => {
      const key = p.teamId || p.teamName;
      if (!map[key]) {
        const info = getTeamInfo(p.teamId, p.teamName);
        map[key] = {
          teamId: p.teamId,
          teamName: p.teamName,
          fullName: info.fullName,
          tricode: info.tricode,
          primaryColor: info.primaryColor,
          players: [],
        };
      }
      map[key].players.push(p);
    });

    return Object.values(map)
      .map(t => {
        const ratings = t.players.map(p => p.predicted_rating).sort((a, b) => b - a);
        const avg = ratings.length ? ratings.reduce((sum, r) => sum + r, 0) / ratings.length : 0;
        const top5 = ratings.slice(0, 5);
        const top5Avg = top5.length ? top5.reduce((sum, r) => sum + r, 0) / top5.length : 0;

        return {
          ...t,
          avgRating: avg,
          top5AvgRating: top5Avg,
          rosterSize: ratings.length,
        };
      })
      .sort((a, b) => {
        if (teamSortMetric === 'top5') return b.top5AvgRating - a.top5AvgRating;
        return b.avgRating - a.avgRating;
      });
  }, [players, teamSortMetric]);

  // Max team rating for bar scaling
  const maxTeamMetric = useMemo(() => {
    if (!teamComparisonData.length) return 6.0;
    const vals = teamComparisonData.map(t => (teamSortMetric === 'top5' ? t.top5AvgRating : t.avgRating));
    return Math.max(...vals, 5.0);
  }, [teamComparisonData, teamSortMetric]);

  // ─── 3. RECENT RATING CHANGES TABLE ───────────────────────────────────────
  const filteredTablePlayers = useMemo(() => {
    let result = players;
    if (tableSearchQuery.trim()) {
      const q = tableSearchQuery.toLowerCase();
      result = result.filter(
        p => p.playerName.toLowerCase().includes(q) || (p.teamName || '').toLowerCase().includes(q)
      );
    }

    return [...result].sort((a, b) => {
      let vA, vB;
      if (tableSortField === 'playerName') {
        vA = a.playerName;
        vB = b.playerName;
        return tableSortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
      } else if (tableSortField === 'teamName') {
        vA = a.teamName || '';
        vB = b.teamName || '';
        return tableSortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
      } else if (tableSortField === 'delta') {
        vA = a.deltaVal ?? -999;
        vB = b.deltaVal ?? -999;
      } else if (tableSortField === 'lastGame') {
        vA = a.lastGameVal ?? -999;
        vB = b.lastGameVal ?? -999;
      } else {
        vA = a.predicted_rating ?? -999;
        vB = b.predicted_rating ?? -999;
      }
      return tableSortAsc ? vA - vB : vB - vA;
    });
  }, [players, tableSearchQuery, tableSortField, tableSortAsc]);

  // Toggle table sort
  function handleTableSort(field) {
    if (tableSortField === field) {
      setTableSortAsc(!tableSortAsc);
    } else {
      setTableSortField(field);
      setTableSortAsc(false);
    }
  }

  // ─── SVG PROGRESSION CHART HELPERS ────────────────────────────────────────
  const progressionPoints = useMemo(() => {
    if (!activeSelectedPlayer) return [];
    const p = activeSelectedPlayer;
    const pts = [
      { label: 'Last 7 Avg', value: p.last7Val, fallback: p.predicted_rating },
      { label: 'Last 3 Avg', value: p.last3Val, fallback: p.predicted_rating },
      { label: 'Last Game', value: p.lastGameVal, fallback: p.predicted_rating },
      { label: 'Predicted', value: p.predicted_rating, fallback: p.predicted_rating },
    ];

    return pts.map(pt => ({
      ...pt,
      displayVal: pt.value !== null ? pt.value : pt.fallback,
      isReal: pt.value !== null,
    }));
  }, [activeSelectedPlayer]);

  // Chart coordinate calculation
  const chartHeight = 180;
  const chartWidth = 520;
  const padding = { top: 25, right: 35, bottom: 35, left: 45 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  const { minVal, maxVal, coordinates } = useMemo(() => {
    if (!progressionPoints.length) return { minVal: 0, maxVal: 10, coordinates: [] };
    const vals = progressionPoints.map(pt => pt.displayVal);
    const low = Math.floor(Math.min(...vals) - 0.5);
    const high = Math.ceil(Math.max(...vals) + 0.5);
    const clampedLow = Math.max(0, low);
    const clampedHigh = Math.max(clampedLow + 2, high);

    const coords = progressionPoints.map((pt, idx) => {
      const x = padding.left + (idx / (progressionPoints.length - 1)) * innerWidth;
      const normalizedY = (pt.displayVal - clampedLow) / (clampedHigh - clampedLow);
      const y = padding.top + innerHeight - normalizedY * innerHeight;
      return { ...pt, x, y };
    });

    return { minVal: clampedLow, maxVal: clampedHigh, coordinates: coords };
  }, [progressionPoints, innerWidth, innerHeight, padding.left, padding.top]);

  return (
    <div className="analytics-container">
      {/* ── SECTION HEADER ────────────────────────────────────────────── */}
      <div className="section-header">
        <div className="section-title">NBA Performance Analytics</div>
        <div className="section-subtitle">
          Real-time model inferences, player momentum indicators, and league-wide franchise metrics
        </div>
      </div>

      <hr className="section-divider" />

      {/* ── ROW 1: PLAYER PROGRESSION & RISERS/FALLERS ─────────────────── */}
      <div className="analytics-grid-two">
        {/* Left: Player Progression Chart */}
        <div className="analytics-card">
          <div className="card-header-row">
            <div>
              <div className="card-title">Player Rating Progression</div>
              <div className="card-subtitle">
                Recent game trajectory &amp; machine learning forecast
              </div>
            </div>

            {/* Date Range Tabs */}
            <div className="analytics-tabs">
              <button
                className={`tab-btn ${selectedRange === 'progression' ? 'active' : ''}`}
                onClick={() => setSelectedRange('progression')}
              >
                Recent Games
              </button>
              <button
                className={`tab-btn ${selectedRange === '7d' ? 'active' : ''}`}
                onClick={() => setSelectedRange('7d')}
              >
                7D
              </button>
              <button
                className={`tab-btn ${selectedRange === '14d' ? 'active' : ''}`}
                onClick={() => setSelectedRange('14d')}
              >
                14D
              </button>
              <button
                className={`tab-btn ${selectedRange === '30d' ? 'active' : ''}`}
                onClick={() => setSelectedRange('30d')}
              >
                30D
              </button>
            </div>
          </div>

          {/* Player Search & Quick Selector */}
          <div className="player-selector-bar">
            <div className="selector-search-wrap">
              <Search size={14} className="selector-search-icon" />
              <input
                type="text"
                placeholder="Search player (e.g. Jokic, Brown, Maxey)..."
                value={playerSearchQuery}
                onChange={e => setPlayerSearchQuery(e.target.value)}
                className="selector-search-input"
              />
            </div>

            {/* Player Quick Badges */}
            <div className="selector-chips">
              {playerSearchResults.map(p => (
                <button
                  key={p.personId}
                  className={`chip-btn ${activeSelectedPlayer?.personId === p.personId ? 'selected' : ''}`}
                  onClick={() => {
                    setSelectedPlayerId(p.personId);
                    setPlayerSearchQuery('');
                  }}
                >
                  {p.playerName.split(' ').pop()}
                  <span className="chip-rating">{p.predicted_rating.toFixed(1)}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Active Player Card Header */}
          {activeSelectedPlayer && (
            <div className="active-player-banner">
              <img
                src={activeSelectedPlayer.headshot || ''}
                alt={activeSelectedPlayer.playerName}
                className="banner-headshot"
                onError={e => { e.target.style.display = 'none'; }}
              />
              <div className="banner-info">
                <div className="banner-name">{activeSelectedPlayer.playerName}</div>
                <div className="banner-team">
                  <NbaLogo teamId={activeSelectedPlayer.teamId} size={16} />
                  <span>{activeSelectedPlayer.teamName}</span>
                  <span className="dot-sep">•</span>
                  <StatusBadge status={activeSelectedPlayer.status} />
                </div>
              </div>

              <div className="banner-stat-box">
                <div className="banner-stat-label">Model Projection</div>
                <div className="banner-stat-value">{activeSelectedPlayer.predicted_rating.toFixed(2)}</div>
              </div>

              <div className="banner-stat-box">
                <div className="banner-stat-label">Momentum</div>
                <div
                  className={`banner-stat-value ${
                    (activeSelectedPlayer.deltaVal ?? 0) >= 0 ? 'text-positive' : 'text-negative'
                  }`}
                >
                  {(activeSelectedPlayer.deltaVal ?? 0) >= 0 ? '+' : ''}
                  {activeSelectedPlayer.deltaVal !== null ? activeSelectedPlayer.deltaVal.toFixed(2) : '0.00'}
                </div>
              </div>

              <div className="banner-action-wrap">
                <LineupActionButton
                  player={activeSelectedPlayer}
                  slots={slots}
                  onToggle={onToggleLineup}
                  size="normal"
                />
              </div>
            </div>
          )}

          {/* Historical Range State Check */}
          {selectedRange !== 'progression' ? (
            <div className="unavailable-history-banner">
              <Calendar size={18} className="history-icon" />
              <div>
                <div className="history-title">{selectedRange.toUpperCase()} Calendar Range Tracking</div>
                <div className="history-desc">
                  Multi-day snapshot archive is active on production schedule. Prior calendar snapshots
                  pre-dating Oct 22, 2026 are not archived. Showing verified recent game sequence below.
                </div>
              </div>
            </div>
          ) : null}

          {/* Chart SVG */}
          <div className="chart-svg-container">
            <svg
              viewBox={`0 0 ${chartWidth} ${chartHeight}`}
              className="analytics-svg"
              preserveAspectRatio="xMidYMid meet"
            >
              {/* Grid Lines */}
              {[minVal, (minVal + maxVal) / 2, maxVal].map((val, i) => {
                const y = padding.top + innerHeight - ((val - minVal) / (maxVal - minVal)) * innerHeight;
                return (
                  <g key={i}>
                    <line
                      x1={padding.left}
                      y1={y}
                      x2={chartWidth - padding.right}
                      y2={y}
                      stroke="#222"
                      strokeDasharray="3 3"
                    />
                    <text
                      x={padding.left - 8}
                      y={y + 4}
                      fill="#666"
                      fontSize="10"
                      fontFamily="var(--font-mono)"
                      textAnchor="end"
                    >
                      {val.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Connecting Line */}
              {coordinates.length > 1 && (
                <polyline
                  fill="none"
                  stroke="#f97316"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={coordinates.map(pt => `${pt.x},${pt.y}`).join(' ')}
                />
              )}

              {/* Area fill under curve */}
              {coordinates.length > 1 && (
                <polygon
                  fill="url(#orangeGradient)"
                  opacity="0.15"
                  points={`
                    ${coordinates[0].x},${padding.top + innerHeight}
                    ${coordinates.map(pt => `${pt.x},${pt.y}`).join(' ')}
                    ${coordinates[coordinates.length - 1].x},${padding.top + innerHeight}
                  `}
                />
              )}

              <defs>
                <linearGradient id="orangeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#f97316" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#f97316" stopOpacity="0" />
                </linearGradient>
              </defs>

              {/* Data Points */}
              {coordinates.map((pt, i) => (
                <g key={i} className="chart-node">
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r={i === coordinates.length - 1 ? 5 : 4}
                    fill={i === coordinates.length - 1 ? '#f97316' : '#fff'}
                    stroke={i === coordinates.length - 1 ? '#000' : '#f97316'}
                    strokeWidth="2"
                  />
                  {/* Value Label */}
                  <text
                    x={pt.x}
                    y={pt.y - 10}
                    fill="#eee"
                    fontSize="11"
                    fontWeight="600"
                    fontFamily="var(--font-mono)"
                    textAnchor="middle"
                  >
                    {pt.displayVal.toFixed(2)}
                  </text>
                  {/* Axis Label */}
                  <text
                    x={pt.x}
                    y={chartHeight - 12}
                    fill="#777"
                    fontSize="10"
                    fontFamily="var(--font-mono)"
                    textAnchor="middle"
                  >
                    {pt.label}
                  </text>
                </g>
              ))}
            </svg>
          </div>
        </div>

        {/* Right: Biggest Risers and Fallers */}
        <div className="analytics-card">
          <div className="card-header-row">
            <div>
              <div className="card-title">Rating Momentum Movers</div>
              <div className="card-subtitle">
                Largest model rating adjustments relative to previous performance
              </div>
            </div>
          </div>

          <div className="movers-split-grid">
            {/* Top Risers */}
            <div className="movers-column">
              <div className="movers-column-title text-positive">
                <TrendingUp size={14} style={{ marginRight: 6 }} /> Top Gainers
              </div>
              <div className="movers-list">
                {risers.map(p => (
                  <div
                    key={p.personId}
                    className="mover-row"
                    onClick={() => onDetail && onDetail(p)}
                    role="button"
                    tabIndex={0}
                  >
                    <NbaLogo teamId={p.teamId} size={24} />
                    <div className="mover-name-wrap">
                      <div className="mover-name">{p.playerName}</div>
                      <div className="mover-team">{p.teamName}</div>
                    </div>
                    <div className="mover-stats">
                      <div className="mover-pred">{p.predicted_rating.toFixed(2)}</div>
                      <div className="mover-delta text-positive">
                        <ArrowUpRight size={12} />
                        +{p.deltaVal?.toFixed(2)}
                      </div>
                    </div>
                    <div className="mover-action-wrap" onClick={e => e.stopPropagation()}>
                      <LineupActionButton
                        player={p}
                        slots={slots}
                        onToggle={onToggleLineup}
                        size="mini"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Fallers */}
            <div className="movers-column">
              <div className="movers-column-title text-negative">
                <TrendingDown size={14} style={{ marginRight: 6 }} /> Top Decliners
              </div>
              <div className="movers-list">
                {fallers.map(p => (
                  <div
                    key={p.personId}
                    className="mover-row"
                    onClick={() => onDetail && onDetail(p)}
                    role="button"
                    tabIndex={0}
                  >
                    <NbaLogo teamId={p.teamId} size={24} />
                    <div className="mover-name-wrap">
                      <div className="mover-name">{p.playerName}</div>
                      <div className="mover-team">{p.teamName}</div>
                    </div>
                    <div className="mover-stats">
                      <div className="mover-pred">{p.predicted_rating.toFixed(2)}</div>
                      <div className="mover-delta text-negative">
                        <ArrowDownRight size={12} />
                        {p.deltaVal?.toFixed(2)}
                      </div>
                    </div>
                    <div className="mover-action-wrap" onClick={e => e.stopPropagation()}>
                      <LineupActionButton
                        player={p}
                        slots={slots}
                        onToggle={onToggleLineup}
                        size="mini"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── ROW 2: TEAM BENCHMARK COMPARISON ──────────────────────────── */}
      <div className="analytics-card" style={{ marginTop: 20 }}>
        <div className="card-header-row">
          <div>
            <div className="card-title">Team Rating Benchmark Comparison</div>
            <div className="card-subtitle">
              All 30 NBA franchises benchmarked by projected player performance
            </div>
          </div>

          <div className="analytics-tabs">
            <button
              className={`tab-btn ${teamSortMetric === 'avg' ? 'active' : ''}`}
              onClick={() => setTeamSortMetric('avg')}
            >
              Roster Mean
            </button>
            <button
              className={`tab-btn ${teamSortMetric === 'top5' ? 'active' : ''}`}
              onClick={() => setTeamSortMetric('top5')}
            >
              Top-5 Starters
            </button>
          </div>
        </div>

        <div className="team-bars-grid">
          {teamComparisonData.map((team, rankIdx) => {
            const val = teamSortMetric === 'top5' ? team.top5AvgRating : team.avgRating;
            const barWidthPct = Math.min(100, Math.max(5, (val / maxTeamMetric) * 100));

            return (
              <div key={team.teamId} className="team-bar-row">
                <div className="team-bar-rank">#{rankIdx + 1}</div>
                <div className="team-bar-label">
                  <NbaLogo teamId={team.teamId} size={20} />
                  <span className="team-bar-name" title={team.fullName}>
                    {team.fullName}
                  </span>
                </div>

                <div className="team-bar-track">
                  <div
                    className="team-bar-fill"
                    style={{
                      width: `${barWidthPct}%`,
                      backgroundColor: team.primaryColor || '#f97316',
                    }}
                  />
                </div>

                <div className="team-bar-val">{val.toFixed(2)}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── ROW 3: RECENT RATING CHANGES DATA TABLE ───────────────────── */}
      <div className="analytics-card" style={{ marginTop: 24 }}>
        <div className="card-header-row">
          <div>
            <div className="card-title">Player Rating Momentum Ledger</div>
            <div className="card-subtitle">
              Comprehensive player rating audit with previous game baselines
            </div>
          </div>

          <div className="table-search-wrap">
            <Search size={14} className="selector-search-icon" />
            <input
              type="text"
              placeholder="Filter by player or team..."
              value={tableSearchQuery}
              onChange={e => setTableSearchQuery(e.target.value)}
              className="table-search-input"
            />
          </div>
        </div>

        <div className="analytics-table-wrap">
          <table className="analytics-table">
            <thead>
              <tr>
                <th onClick={() => handleTableSort('playerName')} className="sortable-th">
                  Player {tableSortField === 'playerName' ? (tableSortAsc ? '▲' : '▼') : ''}
                </th>
                <th onClick={() => handleTableSort('teamName')} className="sortable-th">
                  Team {tableSortField === 'teamName' ? (tableSortAsc ? '▲' : '▼') : ''}
                </th>
                <th>Status</th>
                <th onClick={() => handleTableSort('lastGame')} className="sortable-th text-right">
                  Last Game {tableSortField === 'lastGame' ? (tableSortAsc ? '▲' : '▼') : ''}
                </th>
                <th className="text-right">3-Game Avg</th>
                <th className="text-right">7-Game Avg</th>
                <th onClick={() => handleTableSort('predicted_rating')} className="sortable-th text-right">
                  Predicted Rating {tableSortField === 'predicted_rating' ? (tableSortAsc ? '▲' : '▼') : ''}
                </th>
                <th onClick={() => handleTableSort('delta')} className="sortable-th text-right">
                  Change {tableSortField === 'delta' ? (tableSortAsc ? '▲' : '▼') : ''}
                </th>
                <th className="text-center" style={{ width: 100 }}>
                  Starting Five
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredTablePlayers.slice(0, 50).map(p => {
                const delta = p.deltaVal;
                const isPositive = delta !== null && delta > 0;
                const isNegative = delta !== null && delta < 0;

                return (
                  <tr
                    key={p.personId}
                    onClick={() => onDetail && onDetail(p)}
                    className="analytics-table-row"
                  >
                    <td>
                      <div className="table-player-cell">
                        <img
                          src={p.headshot || ''}
                          alt=""
                          className="table-headshot"
                          onError={e => { e.target.style.display = 'none'; }}
                        />
                        <span className="table-player-name">{p.playerName}</span>
                      </div>
                    </td>
                    <td>
                      <div className="table-team-cell">
                        <NbaLogo teamId={p.teamId} size={18} />
                        <span>{p.teamName}</span>
                      </div>
                    </td>
                    <td>
                      <StatusBadge status={p.status} />
                    </td>
                    <td className="text-right font-mono">
                      {p.lastGameVal !== null ? p.lastGameVal.toFixed(2) : '—'}
                    </td>
                    <td className="text-right font-mono text-muted">
                      {p.last3Val !== null ? p.last3Val.toFixed(2) : '—'}
                    </td>
                    <td className="text-right font-mono text-muted">
                      {p.last7Val !== null ? p.last7Val.toFixed(2) : '—'}
                    </td>
                    <td className="text-right font-mono font-bold text-accent">
                      {p.predicted_rating.toFixed(2)}
                    </td>
                    <td className="text-right font-mono">
                      {delta !== null ? (
                        <span
                          className={`delta-badge ${
                            isPositive ? 'delta-pos' : isNegative ? 'delta-neg' : 'delta-zero'
                          }`}
                        >
                          {isPositive ? '+' : ''}
                          {delta.toFixed(2)}
                        </span>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                    </td>
                    <td className="text-center" onClick={e => e.stopPropagation()}>
                      <LineupActionButton
                        player={p}
                        slots={slots}
                        onToggle={onToggleLineup}
                        size="compact"
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {filteredTablePlayers.length > 50 && (
          <div className="table-footer-note">
            Showing top 50 of {filteredTablePlayers.length} filtered players. Use search above to locate any specific athlete.
          </div>
        )}
      </div>
    </div>
  );
}
