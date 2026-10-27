import { useState, useEffect } from 'react';
import { Layers, Users, Shield, Activity, BarChart3, Menu, X as CloseIcon } from 'lucide-react';
import './App.css';

import LineupSection from './components/LineupSection';
import TopPlayersSection from './components/TopPlayersSection';
import TeamsSection from './components/TeamsSection';
import AnalyticsSection from './components/AnalyticsSection';
import SlotPickerModal from './components/SlotPickerModal';
import PlayerDetailModal from './components/PlayerDetailModal';

import { fetchPlayers, fetchTeams, API_BASE_URL } from './api';

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

export default function App() {
  const [activeSection, setActiveSection] = useState('lineup');
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [homeData, setHomeData] = useState([]);
  const [allData, setAllData] = useState([]);
  const [gameDate, setGameDate] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // 5-slot lineup — null = empty
  const [slots, setSlots] = useState([null, null, null, null, null]);

  // Modals
  const [slotModal, setSlotModal] = useState(null);       // player being added
  const [detailModal, setDetailModal] = useState(null);   // player detail view
  const [pendingSlotIndex, setPendingSlotIndex] = useState(null); // which slot triggered nav

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);
        const [playersResp, teamsResp] = await Promise.all([
          fetchPlayers(),
          fetchTeams(),
        ]);
        setHomeData(playersResp.players || []);
        setAllData(teamsResp.players || []);
        setGameDate(playersResp.game_date || null);
      } catch (err) {
        setError(err.message || 'Unable to connect to the prediction API.');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // ─── Slot click from Lineup section → navigate to Top Players ───────────────
  function handleSlotClick(index) {
    setPendingSlotIndex(index);
    setActiveSection('players');
  }

  // ─── Remove player from specific slot ──────────────────────────────────────
  function handleRemoveSlot(index) {
    setSlots(prev => {
      const next = [...prev];
      next[index] = null;
      return next;
    });
  }

  // ─── Reset entire lineup ───────────────────────────────────────────────────
  function handleClearAll() {
    setSlots([null, null, null, null, null]);
  }

  // ─── "+" button on a player card ───────────────────────────────────────────
  function handleAdd(player) {
    // Prevent adding player if already in any slot
    if (slots.some(s => s?.personId === player.personId)) {
      return;
    }

    // If navigation was triggered from a specific slot click, assign directly or open picker
    if (pendingSlotIndex !== null && pendingSlotIndex >= 0 && pendingSlotIndex < 5) {
      const newSlots = [...slots];
      newSlots[pendingSlotIndex] = player;
      setSlots(newSlots);
      setPendingSlotIndex(null);
      setActiveSection('lineup');
      return;
    }

    setSlotModal(player);
  }

  // ─── Pick which slot ────────────────────────────────────────────────────────
  function handleSelectSlot(slotIndex) {
    if (slotModal) {
      setSlots(prev => {
        const newSlots = [...prev];
        // If player already occupies another slot, clear it to guarantee no duplicates
        for (let i = 0; i < 5; i++) {
          if (newSlots[i]?.personId === slotModal.personId) {
            newSlots[i] = null;
          }
        }
        newSlots[slotIndex] = slotModal;
        return newSlots;
      });
      setSlotModal(null);
      setPendingSlotIndex(null);
      setActiveSection('lineup');
    }
  }

  // ─── Breadcrumb label ───────────────────────────────────────────────────────
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
                {item.label}
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
            />
          )}

          {activeSection === 'teams' && (
            <TeamsSection
              allPlayers={allData}
              onAdd={handleAdd}
              onDetail={setDetailModal}
            />
          )}

          {activeSection === 'analytics' && (
            <AnalyticsSection
              allPlayers={allData}
              onDetail={setDetailModal}
            />
          )}
        </div>

        {/* ── MOBILE BOTTOM NAVIGATION ─────────────────────────── */}
        <nav className="mobile-bottom-nav" aria-label="Mobile Navigation">
          {NAV.map(item => {
            const Icon = item.icon;
            const isActive = activeSection === item.id;
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
                <Icon size={18} className="mobile-nav-tab-icon" />
                <span className="mobile-nav-tab-label">{item.shortLabel || item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

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
        />
      )}
    </div>
  );
}
