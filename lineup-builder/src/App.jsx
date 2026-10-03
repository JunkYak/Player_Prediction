import { useState, useEffect } from 'react';
import { Layers, Users, Shield, Activity } from 'lucide-react';
import './App.css';

import LineupSection from './components/LineupSection';
import TopPlayersSection from './components/TopPlayersSection';
import TeamsSection from './components/TeamsSection';
import SlotPickerModal from './components/SlotPickerModal';
import PlayerDetailModal from './components/PlayerDetailModal';

import { fetchPlayers, fetchTeams, API_BASE_URL } from './api';

const NAV = [
  { id: 'lineup',  label: 'Draft Lineup',  icon: Layers  },
  { id: 'players', label: 'Top Players',   icon: Users   },
  { id: 'teams',   label: 'Teams',         icon: Shield  },
];

function getNow() {
  return new Date().toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

export default function App() {
  const [activeSection, setActiveSection] = useState('lineup');
  const [homeData, setHomeData] = useState([]);
  const [allData, setAllData] = useState([]);
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
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', fontFamily: 'var(--font-mono)', fontSize: 13, letterSpacing: '0.1em', color: '#555', textTransform: 'uppercase' }}>
        Initialising System...
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
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden' }}>

      {/* ── SIDEBAR ─────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-title">ProPredict</div>
          <div className="sidebar-logo-sub">v2.0 // NBA ANALYTICS</div>
        </div>

        <nav className="sidebar-nav">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${activeSection === id ? 'active' : ''}`}
              onClick={() => { setActiveSection(id); setPendingSlotIndex(null); }}
            >
              <Icon size={16} className="nav-item-icon" />
              {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-status">
          <span className="sidebar-status-dot" />
          <span className="sidebar-status-text">System Online</span>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: '#444', marginTop: 6, letterSpacing: '0.06em' }}>
            {homeData.length > 0 ? `${homeData.length} players active` : 'No games scheduled'}
          </div>
        </div>
      </aside>

      {/* ── MAIN ────────────────────────────────────────────────── */}
      <div className="main-content">
        {/* Topbar */}
        <header className="topbar">
          <div className="topbar-breadcrumb">
            ProPredict / <span>{sectionLabel}</span>
          </div>
          <div className="topbar-right">
            <Activity size={14} style={{ color: '#333' }} />
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
        </div>
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
          onClose={() => setDetailModal(null)}
        />
      )}
    </div>
  );
}
