import { useState, useEffect, useCallback } from 'react';
import {
  Layers,
  Users,
  Shield,
  BarChart3,
  Menu,
  X as CloseIcon,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  Info,
  Activity,
} from 'lucide-react';
import './App.css';

import LineupSection from './components/LineupSection';
import TopPlayersSection from './components/TopPlayersSection';
import TeamsSection from './components/TeamsSection';
import AnalyticsSection from './components/AnalyticsSection';
import SlotPickerModal from './components/SlotPickerModal';
import PlayerDetailModal from './components/PlayerDetailModal';

import { fetchPlayers, fetchTeams, fetchTomorrowSchedule, API_BASE_URL } from './api';

const NAV = [
  { id: 'lineup',    label: 'Draft Lineup',  shortLabel: 'Lineup',    icon: Layers    },
  { id: 'players',   label: 'Top Players',   shortLabel: 'Players',   icon: Users     },
  { id: 'teams',     label: 'Teams',         shortLabel: 'Teams',     icon: Shield    },
  { id: 'analytics', label: 'Analytics',     shortLabel: 'Analytics', icon: BarChart3 },
];

function getNow() {
  return new Date().toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

const STORAGE_KEY = 'unreal_starting_five';

export default function App() {
  const [activeSection, setActiveSection] = useState('lineup');
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [homeData, setHomeData] = useState([]);
  const [allData, setAllData] = useState([]);
  const [tomorrowSchedule, setTomorrowSchedule] = useState(null);
  const [gameDate, setGameDate] = useState(null);
  const [gamesScheduled, setGamesScheduled] = useState(false);
  const [generatedAt, setGeneratedAt] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // ─── 5-SLOT PERSISTENT LINEUP STATE ───────────────────────────────────────
  const [slots, setSlots] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length === 5) {
          return parsed;
        }
      }
    } catch {
      // Ignore localStorage read errors
    }
    return [null, null, null, null, null];
  });

  // Sync slots to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(slots));
    } catch {
      // Ignore localStorage write errors
    }
  }, [slots]);

  // ─── TOAST NOTIFICATION SYSTEM ───────────────────────────────────────────
  const [toast, setToast] = useState(null);

  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type, id: Date.now() });
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => {
      setToast(null);
    }, 3200);
    return () => clearTimeout(timer);
  }, [toast]);

  // Modals
  const [slotModal, setSlotModal] = useState(null);
  const [detailModal, setDetailModal] = useState(null);
  const [pendingSlotIndex, setPendingSlotIndex] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);
        const [playersResp, teamsResp, scheduleResp] = await Promise.allSettled([
          fetchPlayers(),
          fetchTeams(),
          fetchTomorrowSchedule(),
        ]);

        if (playersResp.status === 'fulfilled') {
          setHomeData(playersResp.value.players || []);
          setGameDate(playersResp.value.game_date || null);
          setGamesScheduled(Boolean(playersResp.value.games_scheduled));
          setGeneratedAt(playersResp.value.generated_at || null);
        } else {
          throw playersResp.reason;
        }

        if (teamsResp.status === 'fulfilled') {
          setAllData(teamsResp.value.players || []);
          if (teamsResp.value.generated_at) {
            setGeneratedAt(prev => prev || teamsResp.value.generated_at);
          }
        }

        if (scheduleResp.status === 'fulfilled') {
          setTomorrowSchedule(scheduleResp.value);
        }
      } catch (err) {
        setError(err.message || 'Unable to connect to the prediction API.');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // ─── SLOT CLICK FROM LINEUP SECTION -> NAVIGATE TO PLAYERS ───────────────
  function handleSlotClick(index) {
    setPendingSlotIndex(index);
    setActiveSection('players');
    showToast(`Select a player for Slot ${index + 1}`, 'info');
  }

  // ─── REMOVE PLAYER FROM SPECIFIC SLOT ────────────────────────────────────
  function handleRemoveSlot(index) {
    const removedPlayer = slots[index];
    setSlots(prev => {
      const next = [...prev];
      next[index] = null;
      return next;
    });
    if (removedPlayer) {
      showToast(`Removed ${removedPlayer.playerName} from Slot ${index + 1}`, 'info');
    }
  }

  // ─── RESET ENTIRE LINEUP ─────────────────────────────────────────────────
  function handleClearAll() {
    setSlots([null, null, null, null, null]);
    showToast('Starting Five cleared', 'info');
  }

  // ─── UNIVERSAL TOGGLE LINEUP (ADD / REMOVE FROM ANYWHERE) ────────────────
  const handleToggleLineup = useCallback((player) => {
    if (!player || !player.personId) return;

    // 1. Check if player already occupies a slot -> REMOVE
    const existingIndex = slots.findIndex(s => s && s.personId === player.personId);
    if (existingIndex !== -1) {
      setSlots(prev => {
        const next = [...prev];
        next[existingIndex] = null;
        return next;
      });
      showToast(`Removed ${player.playerName} from Starting Five`, 'info');
      return;
    }

    // 2. Prevent drafting OUT players
    if (player.status === 'Out') {
      showToast(`${player.playerName} is currently Out (injured) and cannot be drafted`, 'warning');
      return;
    }

    // 3. Check if Starting Five is full (5/5)
    const emptyIndex = slots.findIndex(s => s === null);
    if (emptyIndex === -1) {
      showToast('Starting Five is full (5/5). Remove a player first.', 'warning');
      return;
    }

    // 4. Determine target slot (pending slot from slot click or next open slot)
    const targetSlot = (
      pendingSlotIndex !== null &&
      pendingSlotIndex >= 0 &&
      pendingSlotIndex < 5 &&
      slots[pendingSlotIndex] === null
    ) ? pendingSlotIndex : emptyIndex;

    setSlots(prev => {
      const next = [...prev];
      next[targetSlot] = player;
      return next;
    });
    setPendingSlotIndex(null);
    showToast(`Added ${player.playerName} to Starting Five (Slot ${targetSlot + 1})`, 'success');
  }, [slots, pendingSlotIndex, showToast]);

  // Legacy handleAdd compatibility
  function handleAdd(player) {
    handleToggleLineup(player);
  }

  // ─── PICK SPECIFIC SLOT VIA MODAL ─────────────────────────────────────────
  function handleSelectSlot(slotIndex) {
    if (slotModal) {
      setSlots(prev => {
        const newSlots = [...prev];
        for (let i = 0; i < 5; i++) {
          if (newSlots[i]?.personId === slotModal.personId) {
            newSlots[i] = null;
          }
        }
        newSlots[slotIndex] = slotModal;
        return newSlots;
      });
      showToast(`Assigned ${slotModal.playerName} to Slot ${slotIndex + 1}`, 'success');
      setSlotModal(null);
      setPendingSlotIndex(null);
      setActiveSection('lineup');
    }
  }

  const filledCount = slots.filter(Boolean).length;
  const sectionLabel = NAV.find(n => n.id === activeSection)?.label ?? '';

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', fontFamily: 'var(--font-mono)', fontSize: 13, letterSpacing: '0.1em', color: '#666', textTransform: 'uppercase' }}>
        Initialising UN-REAL...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', fontFamily: 'var(--font-mono)', fontSize: 12, color: '#ef4444', textTransform: 'uppercase', letterSpacing: '0.08em', textAlign: 'center', padding: 40 }}>
        <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 14 }}>⚠ API CONNECTION ERROR</div>
        <div style={{ color: '#aaa', maxWidth: 480, marginBottom: 16, textTransform: 'none', lineHeight: 1.5 }}>{error}</div>
        <span style={{ color: '#666', fontSize: 11, textTransform: 'none' }}>Ensure FastAPI backend is running at {API_BASE_URL} and serving artifacts are loaded.</span>
      </div>
    );
  }

  return (
    <div className="app-shell">
      {/* ── MOBILE BACKDROP ──────────────────────────────────────── */}
      {mobileNavOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setMobileNavOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* ── SIDEBAR (DESKTOP + MOBILE DRAWER) ────────────────────── */}
      <aside className={`sidebar ${mobileNavOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-logo">
          <div className="sidebar-logo-title">UN-REAL</div>
          <button
            type="button"
            className="sidebar-close-btn"
            onClick={() => setMobileNavOpen(false)}
            aria-label="Close navigation"
          >
            <CloseIcon size={18} />
          </button>
        </div>

        <nav className="sidebar-nav">
          {NAV.map(item => {
            const Icon = item.icon;
            const isLineupTab = item.id === 'lineup';
            return (
              <button
                key={item.id}
                className={`nav-item ${activeSection === item.id ? 'active' : ''}`}
                onClick={() => {
                  setActiveSection(item.id);
                  setPendingSlotIndex(null);
                  setMobileNavOpen(false);
                }}
              >
                <Icon size={16} className="nav-item-icon" />
                <span className="nav-item-text">{item.label}</span>
                {isLineupTab && (
                  <span className={`lineup-count-pill ${filledCount === 5 ? 'pill-complete' : ''}`}>
                    {filledCount}/5
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </aside>

      {/* ── MAIN ────────────────────────────────────────────────── */}
      <div className="main-content">
        {/* Topbar */}
        <header className="topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="mobile-menu-btn"
              onClick={() => setMobileNavOpen(prev => !prev)}
              aria-label={mobileNavOpen ? 'Close navigation menu' : 'Open navigation menu'}
              aria-expanded={mobileNavOpen}
            >
              {mobileNavOpen ? <CloseIcon size={18} /> : <Menu size={18} />}
            </button>
            <div className="topbar-breadcrumb">
              UN-REAL / <span>{sectionLabel}</span>
            </div>
          </div>

          <div className="topbar-right">
            <Activity size={14} style={{ color: 'var(--accent, #f97316)' }} />
            <span className="topbar-date">{getNow()}</span>
          </div>
        </header>

        {/* Section Content */}
        <div className="content-area">
          {activeSection === 'lineup' && (
            <LineupSection
              slots={slots}
              onSlotClick={handleSlotClick}
              onRemoveSlot={handleRemoveSlot}
              onClearAll={handleClearAll}
              onDetail={setDetailModal}
              gamesAvailable={homeData.length > 0}
            />
          )}

          {activeSection === 'players' && (
            <TopPlayersSection
              players={homeData}
              onAdd={handleAdd}
              onDetail={setDetailModal}
              slots={slots}
              onToggleLineup={handleToggleLineup}
            />
          )}

          {activeSection === 'teams' && (
            <TeamsSection
              allPlayers={allData}
              onAdd={handleAdd}
              onDetail={setDetailModal}
              slots={slots}
              onToggleLineup={handleToggleLineup}
              homeData={homeData}
              gameDate={gameDate}
              gamesScheduled={gamesScheduled}
              generatedAt={generatedAt}
              tomorrowSchedule={tomorrowSchedule}
            />
          )}

          {activeSection === 'analytics' && (
            <AnalyticsSection
              allPlayers={allData}
              onDetail={setDetailModal}
              slots={slots}
              onToggleLineup={handleToggleLineup}
            />
          )}
        </div>

        {/* ── MOBILE BOTTOM NAVIGATION ─────────────────────────── */}
        <nav className="mobile-bottom-nav" aria-label="Mobile Navigation">
          {NAV.map(item => {
            const Icon = item.icon;
            const isActive = activeSection === item.id;
            const isLineupTab = item.id === 'lineup';
            return (
              <button
                key={item.id}
                type="button"
                className={`mobile-nav-tab ${isActive ? 'active' : ''}`}
                onClick={() => {
                  setActiveSection(item.id);
                  setPendingSlotIndex(null);
                  setMobileNavOpen(false);
                }}
                aria-label={item.label}
              >
                <div className="mobile-tab-icon-wrap">
                  <Icon size={18} className="mobile-nav-tab-icon" />
                  {isLineupTab && filledCount > 0 && (
                    <span className="mobile-tab-badge">{filledCount}</span>
                  )}
                </div>
                <span className="mobile-nav-tab-label">{item.shortLabel || item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* ── TOAST NOTIFICATION CONTAINER ────────────────────────── */}
      {toast && (
        <div className={`unreal-toast unreal-toast-${toast.type}`} role="status" aria-live="polite">
          <div className="toast-icon">
            {toast.type === 'success' && <CheckCircle2 size={16} />}
            {toast.type === 'error' && <AlertCircle size={16} />}
            {toast.type === 'warning' && <AlertTriangle size={16} />}
            {toast.type === 'info' && <Info size={16} />}
          </div>
          <div className="toast-message">{toast.message}</div>
          <button
            type="button"
            className="toast-close-btn"
            onClick={() => setToast(null)}
            aria-label="Dismiss notification"
          >
            <CloseIcon size={14} />
          </button>
        </div>
      )}

      {/* ── SLOT PICKER MODAL ───────────────────────────────────── */}
      {slotModal && (
        <SlotPickerModal
          player={slotModal}
          slots={slots}
          onSelectSlot={handleSelectSlot}
          onClose={() => setSlotModal(null)}
        />
      )}

      {/* ── PLAYER DETAIL MODAL ─────────────────────────────────── */}
      {detailModal && (
        <PlayerDetailModal
          player={detailModal}
          gameDate={gameDate}
          onClose={() => setDetailModal(null)}
          slots={slots}
          onToggleLineup={handleToggleLineup}
        />
      )}
    </div>
  );
}
