import { useState, useEffect } from 'react';
import { Layers, Users, Shield, Activity } from 'lucide-react';
import './App.css';

import LineupSection from './components/LineupSection';
import TopPlayersSection from './components/TopPlayersSection';
import TeamsSection from './components/TeamsSection';
import SlotPickerModal from './components/SlotPickerModal';
import PlayerDetailModal from './components/PlayerDetailModal';

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
        const [homeRes, allRes] = await Promise.all([
          fetch('/frontend_home.json'),
          fetch('/frontend_all.json'),
        ]);
        if (!homeRes.ok || !allRes.ok) throw new Error('Failed to fetch data files.');
        const [home, all] = await Promise.all([homeRes.json(), allRes.json()]);
        setHomeData(home);
        setAllData(all);
      } catch (err) {
        setError(err.message);
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

  // ─── "+" button on a player card ───────────────────────────────────────────
  function handleAdd(player) {
    // Check duplicate
    if (slots.some(s => s?.personId === player.personId)) return;
    setSlotModal(player);
  }

  // ─── Pick which slot ────────────────────────────────────────────────────────
  function handleSelectSlot(slotIndex) {
    if (slotModal) {
      const newSlots = [...slots];
      newSlots[slotIndex] = slotModal;
      setSlots(newSlots);
      setSlotModal(null);
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
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', fontFamily: 'var(--font-mono)', fontSize: 12, color: '#ef4444', textTransform: 'uppercase', letterSpacing: '0.08em', textAlign: 'center', padding: 40 }}>
        ⚠ DATA ERROR<br /><br />{error}<br /><br />
        <span style={{ color: '#555' }}>Ensure the Python pipeline has run and the app is served via a local server.</span>
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
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: '#333', marginTop: 6, letterSpacing: '0.06em' }}>
            {homeData.length} players loaded
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
