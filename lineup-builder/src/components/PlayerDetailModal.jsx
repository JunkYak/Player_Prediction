import { useEffect } from 'react';
import { X, User, TrendingUp, TrendingDown, Minus, Calendar, Shield, Activity, Info } from 'lucide-react';
import StatusBadge from './StatusBadge';
import NbaLogo from './NbaLogo';
import LineupActionButton from './LineupActionButton';
import { getTeamInfo } from '../constants/teams';
import { getOpponentFromMatchups } from '../utils/schedule';

/**
 * PlayerDetailModal
 *
 * Enhanced Player Intel modal presenting comprehensive performance ratings,
 * verified rolling form trajectory, matchup context, and momentum deltas.
 * Strictly adheres to data integrity: never fabricates artificial game logs.
 */
export default function PlayerDetailModal({
  player,
  gameDate,
  onClose,
  slots = [],
  onToggleLineup,
  tomorrowSchedule = null,
}) {
  // Support closing with Escape key and lock background scroll
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') {
        onClose();
      }
    }
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      document.body.style.overflow = prevOverflow;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  if (!player) return null;

  // Identity fields
  const name = player.playerName || 'Unknown Player';
  const teamId = player.teamId;
  const teamName = player.teamName || '—';
  const teamInfo = getTeamInfo(teamId, teamName);
  const headshot = player.headshot || '';
  const status = player.status || 'Active';
  const fallbackOpp = tomorrowSchedule?.matchups ? getOpponentFromMatchups(teamId, tomorrowSchedule.matchups) : null;
  const opponentName = (player.opponentName && player.opponentName !== '—' && player.opponentName !== '')
    ? player.opponentName
    : fallbackOpp;
  const position = player.position || null;
  const jersey = player.jerseyNumber || player.jersey || null;

  // Rating numbers
  const predRating = typeof player.predicted_rating === 'number' && !isNaN(player.predicted_rating)
    ? player.predicted_rating
    : null;
  const lastGame = typeof player.last_game === 'number' && !isNaN(player.last_game)
    ? player.last_game
    : null;
  const last3 = typeof player.last3_avg === 'number' && !isNaN(player.last3_avg)
    ? player.last3_avg
    : null;
  const last7 = typeof player.last7_avg === 'number' && !isNaN(player.last7_avg)
    ? player.last7_avg
    : null;

  // Season Performance Multiplier metrics
  const seasonGameCount = typeof player.season_game_count === 'number'
    ? player.season_game_count
    : (typeof player.season_games_played === 'number' ? player.season_games_played : 0);
  const seasonAvg = typeof player.season_average_rating === 'number'
    ? player.season_average_rating
    : (typeof player.season_avg_rating === 'number' ? player.season_avg_rating : null);
  const seasonPct = typeof player.season_percentile === 'number' ? player.season_percentile : null;
  const seasonBonus = Number(player.season_performance_bonus) || 0.0;
  const rankingScore = typeof player.ranking_score === 'number'
    ? player.ranking_score
    : (typeof player.adjusted_predicted_rating === 'number'
        ? player.adjusted_predicted_rating
        : (predRating !== null ? predRating * (1 + seasonBonus) : null));

  // Calculated momentum deltas
  const deltaLastGame = predRating !== null && lastGame !== null ? predRating - lastGame : null;
  const delta3Avg = predRating !== null && last3 !== null ? predRating - last3 : null;

  // Recent trajectory data points for the SVG mini-chart
  const chartPoints = [
    { label: '7-Game Avg', value: last7, isReal: last7 !== null },
    { label: '3-Game Avg', value: last3, isReal: last3 !== null },
    { label: 'Last Game', value: lastGame, isReal: lastGame !== null },
    { label: 'Predicted', value: predRating, isReal: predRating !== null, isForecast: true },
  ];

  const hasHistory = last7 !== null || last3 !== null || lastGame !== null;

  // Mini-chart coordinate calculations
  const svgWidth = 440;
  const svgHeight = 130;
  const pad = { top: 22, right: 30, bottom: 28, left: 35 };
  const innerW = svgWidth - pad.left - pad.right;
  const innerH = svgHeight - pad.top - pad.bottom;

  const validVals = chartPoints.filter(p => p.value !== null).map(p => p.value);
  const minVal = validVals.length ? Math.max(0, Math.floor(Math.min(...validVals) - 0.5)) : 0;
  const maxVal = validVals.length ? Math.max(minVal + 2, Math.ceil(Math.max(...validVals) + 0.5)) : 8;

  const coords = chartPoints.map((pt, idx) => {
    const x = pad.left + (idx / (chartPoints.length - 1)) * innerW;
    const yVal = pt.value !== null ? pt.value : predRating ?? 0;
    const norm = (yVal - minVal) / (maxVal - minVal || 1);
    const y = pad.top + innerH - norm * innerH;
    return { ...pt, x, y };
  });

  return (
    <div
      className="modal-overlay"
      onClick={e => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="player-modal-title"
    >
      <div className="modal-box modal-box-intel" onClick={e => e.stopPropagation()}>
        <div className="modal-sheet-handle" aria-hidden="true" />
        {/* Modal Top Header */}
        <div className="modal-header">
          <div className="intel-modal-title-wrap">
            <span className="modal-title" id="player-modal-title">Player Intel</span>
            <span className="intel-badge">Performance Dossier</span>
          </div>
          <div className="modal-header-actions">
            <LineupActionButton
              player={player}
              slots={slots}
              onToggle={onToggleLineup}
              size="normal"
            />
            <button
              type="button"
              className="modal-close"
              onClick={onClose}
              aria-label="Close modal"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* ── 1. PLAYER IDENTITY HEADER ───────────────────────────────── */}
        <div className="detail-player-header">
          <div className="detail-headshot-wrap">
            <img
              className="detail-headshot"
              src={headshot}
              alt={name}
              onError={e => {
                e.target.style.display = 'none';
                if (e.target.nextSibling) e.target.nextSibling.style.display = 'flex';
              }}
            />
            <div className="detail-headshot-fallback" style={{ display: 'none' }}>
              <User size={36} />
            </div>
          </div>

          <div className="detail-player-info">
            <div className="detail-player-name">{name}</div>

            <div className="detail-player-meta-row">
              <div className="detail-team-tag">
                <NbaLogo teamId={teamId} teamName={teamName} size={18} />
                <span>{teamInfo.fullName || teamName}</span>
              </div>

              {(position || jersey) && (
                <>
                  <span className="dot-sep">•</span>
                  <span className="detail-position-tag">
                    {position ? position : ''}
                    {position && jersey ? ' #' : ''}
                    {jersey ? jersey : ''}
                  </span>
                </>
              )}

              <span className="dot-sep">•</span>
              <StatusBadge status={status} />
            </div>

            {/* Matchup Sub-badge */}
            <div className="detail-matchup-pill">
              <Calendar size={12} className="matchup-icon" />
              {opponentName ? (
                <span>
                  vs <strong>{opponentName}</strong>
                  {gameDate ? ` • ${gameDate}` : ' • Upcoming Slate'}
                </span>
              ) : (
                <span className="text-dim">No upcoming game scheduled on active slate</span>
              )}
            </div>
          </div>
        </div>

        {/* ── 2. RATING SUMMARY METRICS ───────────────────────────────── */}
        <div className="intel-section-title">Performance Index Summary</div>
        <div className="detail-stats-grid">
          {/* Predicted Rating Card */}
          <div className="detail-stat-card stat-card-highlight">
            <div className="detail-stat-label">Model Projected Rating</div>
            <div className="detail-stat-value text-accent">
              {predRating !== null ? predRating.toFixed(2) : '—'}
            </div>
            <div className="stat-card-sublabel">Forecasted player impact</div>
          </div>

          {/* Last Game Rating */}
          <div className="detail-stat-card">
            <div className="detail-stat-label">Last Game Rating</div>
            <div className="detail-stat-value">
              {lastGame !== null ? lastGame.toFixed(2) : '—'}
            </div>
            {deltaLastGame !== null ? (
              <div className={`stat-delta-tag ${deltaLastGame >= 0 ? 'delta-pos' : 'delta-neg'}`}>
                {deltaLastGame >= 0 ? '+' : ''}
                {deltaLastGame.toFixed(2)} vs Last Game
              </div>
            ) : (
              <div className="stat-card-sublabel">No prior game</div>
            )}
          </div>

          {/* Last 3-Game Average */}
          <div className="detail-stat-card">
            <div className="detail-stat-label">Last 3-Game Avg</div>
            <div className="detail-stat-value">
              {last3 !== null ? last3.toFixed(2) : '—'}
            </div>
            {delta3Avg !== null ? (
              <div className={`stat-delta-tag ${delta3Avg >= 0 ? 'delta-pos' : 'delta-neg'}`}>
                {delta3Avg >= 0 ? '+' : ''}
                {delta3Avg.toFixed(2)} vs 3-Game Avg
              </div>
            ) : (
              <div className="stat-card-sublabel">Short-term baseline</div>
            )}
          </div>

          {/* Last 7-Game Average */}
          <div className="detail-stat-card">
            <div className="detail-stat-label">Last 7-Game Avg</div>
            <div className="detail-stat-value">
              {last7 !== null ? last7.toFixed(2) : '—'}
            </div>
            <div className="stat-card-sublabel">Multi-game baseline</div>
          </div>
        </div>

        {/* Season Performance Multiplier Section */}
        <div className="intel-section-title" style={{ marginTop: 18 }}>
          Season Performance Multiplier (Current Season)
        </div>
        <div className="detail-stats-grid">
          <div className="detail-stat-card stat-card-highlight">
            <div className="detail-stat-label">Season Bonus Multiplier</div>
            <div
              className="detail-stat-value"
              style={{ color: seasonBonus > 0 ? '#10b981' : 'var(--text-primary)' }}
            >
              +{seasonBonus.toFixed(1)}x
            </div>
            <div className="stat-card-sublabel">Added to Starting Five slot</div>
          </div>

          <div className="detail-stat-card">
            <div className="detail-stat-label">Season Avg Rating</div>
            <div className="detail-stat-value">
              {seasonAvg !== null ? seasonAvg.toFixed(2) : '—'}
            </div>
            <div className="stat-card-sublabel">From {seasonGameCount} completed games</div>
          </div>

          <div className="detail-stat-card">
            <div className="detail-stat-label">League Percentile</div>
            <div className="detail-stat-value">
              {seasonPct !== null ? `${seasonPct.toFixed(1)}%` : '—'}
            </div>
            <div className="stat-card-sublabel">
              {seasonGameCount >= 3 ? 'Qualified (≥ 3 games)' : 'Excluded (< 3 games)'}
            </div>
          </div>

          <div className="detail-stat-card">
            <div className="detail-stat-label">Top Players Ranking Score</div>
            <div className="detail-stat-value text-accent">
              {rankingScore !== null ? rankingScore.toFixed(2) : '—'}
            </div>
            <div className="stat-card-sublabel">Predicted × (1 + Bonus)</div>
          </div>
        </div>

        {/* Ranking Score Clarification Notice */}
        <div className="forecast-disclaimer" style={{ marginTop: 10 }}>
          <Info size={12} className="disclaimer-icon" />
          <span>
            The Top Players Ranking Score is exclusively a sorting metric for draft ordering. It does not alter or replace the Random Forest model predicted rating ({predRating !== null ? predRating.toFixed(2) : '—'}). In Starting Five lineups, the season bonus is added to the slot multiplier.
          </span>
        </div>

        {/* Forecast Non-Guarantee Notice */}
        <div className="forecast-disclaimer">
          <Info size={12} className="disclaimer-icon" />
          <span>
            Model ratings are statistical projections computed from rolling usage, efficiency, rest, and game dynamics. Projections represent expected performance, not a guaranteed outcome.
          </span>
        </div>

        {/* ── 3. RECENT ROLLING TRAJECTORY CHART ──────────────────────── */}
        <div className="intel-chart-card">
          <div className="intel-card-header">
            <div>
              <div className="intel-card-heading">Recent Form &amp; Rolling Trajectory</div>
              <div className="intel-card-sub">
                Progression of verified rolling averages leading to model forecast
              </div>
            </div>
          </div>

          {hasHistory ? (
            <div className="modal-svg-wrap">
              <svg
                viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                className="intel-svg"
                preserveAspectRatio="xMidYMid meet"
              >
                {/* Horizontal reference grid lines */}
                {[minVal, (minVal + maxVal) / 2, maxVal].map((v, i) => {
                  const y = pad.top + innerH - ((v - minVal) / (maxVal - minVal || 1)) * innerH;
                  return (
                    <g key={i}>
                      <line
                        x1={pad.left}
                        y1={y}
                        x2={svgWidth - pad.right}
                        y2={y}
                        stroke="#222"
                        strokeDasharray="2 2"
                      />
                      <text
                        x={pad.left - 6}
                        y={y + 3}
                        fill="#555"
                        fontSize="9"
                        fontFamily="var(--font-mono)"
                        textAnchor="end"
                      >
                        {v.toFixed(1)}
                      </text>
                    </g>
                  );
                })}

                {/* Trajectory Polyline */}
                {coords.length > 1 && (
                  <polyline
                    fill="none"
                    stroke="#f97316"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={coords.map(pt => `${pt.x},${pt.y}`).join(' ')}
                  />
                )}

                {/* Shaded Area Under Line */}
                {coords.length > 1 && (
                  <polygon
                    fill="url(#modalOrangeGlow)"
                    opacity="0.12"
                    points={`
                      ${coords[0].x},${pad.top + innerH}
                      ${coords.map(pt => `${pt.x},${pt.y}`).join(' ')}
                      ${coords[coords.length - 1].x},${pad.top + innerH}
                    `}
                  />
                )}

                <defs>
                  <linearGradient id="modalOrangeGlow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f97316" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="#f97316" stopOpacity="0" />
                  </linearGradient>
                </defs>

                {/* Data Points and Value Tags */}
                {coords.map((pt, i) => (
                  <g key={i}>
                    <circle
                      cx={pt.x}
                      cy={pt.y}
                      r={pt.isForecast ? 4.5 : 3.5}
                      fill={pt.isForecast ? '#f97316' : '#fff'}
                      stroke={pt.isForecast ? '#000' : '#f97316'}
                      strokeWidth="2"
                    />
                    <text
                      x={pt.x}
                      y={pt.y - 7}
                      fill="#eee"
                      fontSize="10"
                      fontWeight="700"
                      fontFamily="var(--font-mono)"
                      textAnchor="middle"
                    >
                      {pt.value !== null ? pt.value.toFixed(2) : '—'}
                    </text>
                    <text
                      x={pt.x}
                      y={svgHeight - 8}
                      fill={pt.isForecast ? 'var(--accent)' : '#777'}
                      fontSize="9"
                      fontWeight={pt.isForecast ? '700' : '400'}
                      fontFamily="var(--font-mono)"
                      textAnchor="middle"
                    >
                      {pt.label}
                    </text>
                  </g>
                ))}
              </svg>

              <div className="chart-honest-note">
                Verified rolling averages and active model projection. Individual game box-scores are not served over active REST payloads.
              </div>
            </div>
          ) : (
            <div className="chart-empty-state">
              <Activity size={24} style={{ color: '#444', marginBottom: 6 }} />
              <div className="empty-title">No Historical Game Logs Recorded</div>
              <div className="empty-desc">
                This player has zero historical appearances in the 2026–27 tracking index. Model projection is established via rookie/cohort baseline imputation.
              </div>
            </div>
          )}
        </div>

        {/* ── 4. MATCHUP INTELLIGENCE & CONTEXT ───────────────────────── */}
        <div className="intel-matchup-section">
          <div className="intel-section-title" style={{ marginTop: 16 }}>Matchup Intelligence</div>

          <div className="matchup-context-cards">
            {/* Opponent Card */}
            <div className="context-card">
              <div className="context-card-label">Opponent Franchise</div>
              <div className="context-card-content">
                {opponentName ? (
                  <div className="context-opp-wrap">
                    <NbaLogo teamName={opponentName} size={24} />
                    <span className="context-opp-name">{opponentName}</span>
                  </div>
                ) : (
                  <span className="text-dim">No upcoming game scheduled</span>
                )}
              </div>
            </div>

            {/* Form Momentum Card */}
            <div className="context-card">
              <div className="context-card-label">Form Momentum</div>
              <div className="context-card-content">
                {deltaLastGame !== null ? (
                  <div className="context-momentum-row">
                    {deltaLastGame > 0.25 ? (
                      <>
                        <TrendingUp size={15} className="text-positive" />
                        <span className="text-positive font-bold">Surging (+{deltaLastGame.toFixed(2)})</span>
                      </>
                    ) : deltaLastGame < -0.25 ? (
                      <>
                        <TrendingDown size={15} className="text-negative" />
                        <span className="text-negative font-bold">Cooling ({deltaLastGame.toFixed(2)})</span>
                      </>
                    ) : (
                      <>
                        <Minus size={15} style={{ color: '#888' }} />
                        <span className="text-dim font-bold">Steady (±{Math.abs(deltaLastGame).toFixed(2)})</span>
                      </>
                    )}
                  </div>
                ) : (
                  <span className="text-dim">Baseline Established</span>
                )}
              </div>
            </div>

            {/* Availability Detail */}
            <div className="context-card">
              <div className="context-card-label">Availability Context</div>
              <div className="context-card-content">
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Shield size={14} style={{ color: 'var(--accent)' }} />
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#ddd' }}>
                    {status === 'Active' ? 'Active / Available' : status}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
