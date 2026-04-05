const NBA_TEAMS = {
    "1610612737": { name: "Atlanta Hawks", color: "#e03a3e" },
    "1610612738": { name: "Boston Celtics", color: "#007A33" },
    "1610612751": { name: "Brooklyn Nets", color: "#ffffff" },
    "1610612766": { name: "Charlotte Hornets", color: "#1d1160" },
    "1610612741": { name: "Chicago Bulls", color: "#ce1141" },
    "1610612739": { name: "Cleveland Cavaliers", color: "#860038" },
    "1610612742": { name: "Dallas Mavericks", color: "#00538c" },
    "1610612743": { name: "Denver Nuggets", color: "#0E2240" },
    "1610612765": { name: "Detroit Pistons", color: "#C8102E" },
    "1610612744": { name: "Golden State Warriors", color: "#1D428A" },
    "1610612745": { name: "Houston Rockets", color: "#CE1141" },
    "1610612754": { name: "Indiana Pacers", color: "#FDBB30" },
    "1610612746": { name: "LA Clippers", color: "#1D428A" },
    "1610612747": { name: "Los Angeles Lakers", color: "#FDB927" },
    "1610612763": { name: "Memphis Grizzlies", color: "#5D76A9" },
    "1610612748": { name: "Miami Heat", color: "#98002E" },
    "1610612749": { name: "Milwaukee Bucks", color: "#00471B" },
    "1610612750": { name: "Minnesota Timberwolves", color: "#0C2340" },
    "1610612740": { name: "New Orleans Pelicans", color: "#0C2340" },
    "1610612752": { name: "New York Knicks", color: "#F58426" },
    "1610612760": { name: "Oklahoma City Thunder", color: "#007ac1" },
    "1610612753": { name: "Orlando Magic", color: "#0077c0" },
    "1610612755": { name: "Philadelphia 76ers", color: "#006bb6" },
    "1610612756": { name: "Phoenix Suns", color: "#1d1160" },
    "1610612757": { name: "Portland Trail Blazers", color: "#E03A3E" },
    "1610612758": { name: "Sacramento Kings", color: "#5a2d81" },
    "1610612759": { name: "San Antonio Spurs", color: "#c4ced4" },
    "1610612761": { name: "Toronto Raptors", color: "#ce1141" },
    "1610612762": { name: "Utah Jazz", color: "#002B5C" },
    "1610612764": { name: "Washington Wizards", color: "#002B5C" }
};

let rawData = null;
let currentTeamId = null;

// Initialization
document.addEventListener("DOMContentLoaded", async () => {
    try {
        const response = await fetch("../data/team_rankings.json");
        if (!response.ok) throw new Error("Failed to fetch data.");
        
        rawData = await response.json();
        const teamIds = Object.keys(rawData);
        
        if (teamIds.length === 0) {
            document.getElementById("roster-grid").innerHTML = `<div class="empty-state"><p>No prediction data found.</p></div>`;
            return;
        }

        buildSidebar(teamIds);
        
        // Select first team
        selectTeam(teamIds[0]);
    } catch (err) {
        console.error(err);
        document.getElementById("roster-grid").innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-triangle-exclamation fa-3x" style="color:var(--danger)"></i>
                <p>Could not load ranking data. Make sure to run the Python backend pipeline first and serve using a valid HTTP server.</p>
            </div>
        `;
    }
});

function getTeamInfo(teamId) {
    if (NBA_TEAMS[teamId]) {
        return NBA_TEAMS[teamId];
    }
    return { name: `Team ${teamId}`, color: "#3b82f6" };
}

function buildSidebar(teamIds) {
    const list = document.getElementById("team-list");
    list.innerHTML = "";

    // Sort team names
    const sortedTeams = teamIds.map(id => ({ id, ...getTeamInfo(id) })).sort((a,b) => a.name.localeCompare(b.name));

    sortedTeams.forEach(team => {
        const btn = document.createElement("button");
        btn.className = "team-btn";
        btn.id = `btn-${team.id}`;
        
        // Use color dot
        btn.innerHTML = `
            <span style="width:10px; height:10px; border-radius:50%; background-color:${team.color}; box-shadow: 0 0 8px ${team.color}"></span>
            ${team.name}
        `;
        
        btn.onclick = () => selectTeam(team.id);
        list.appendChild(btn);
    });
}

function selectTeam(teamId) {
    if(currentTeamId) {
        const oldBtn = document.getElementById(`btn-${currentTeamId}`);
        if(oldBtn) oldBtn.classList.remove("active");
    }

    currentTeamId = teamId;
    const btn = document.getElementById(`btn-${teamId}`);
    if(btn) btn.classList.add("active");

    const tInfo = getTeamInfo(teamId);
    
    // Update Theme Variables dynamically!
    document.documentElement.style.setProperty("--accent-color", tInfo.color);
    
    // Attempt to calculate glow from color (simple opacity hack done in CSS by inheriting var instead, or redefining here)
    const cardGlow = tInfo.color;
    
    // DOM Updates
    document.getElementById("current-team-name").textContent = tInfo.name;
    document.getElementById("current-team-name").style.color = tInfo.color;
    
    renderTeamData(teamId);
}

function renderTeamData(teamId) {
    const players = rawData[teamId] || [];
    
    if (players.length > 0) {
        document.getElementById("latest-game-date").textContent = `Game Date: ${players[0].game_date || "Unknown"}`;
    }

    // Top Stats logic
    const topPlayer = players[0];
    const topImprover = [...players].sort((a,b) => (b.predicted_rating - b.last_game) - (a.predicted_rating - a.last_game))[0];
    const topAverage = [...players].sort((a,b) => b.last3_avg - a.last3_avg)[0];

    document.getElementById("stats-row").innerHTML = `
        <div class="stat-card">
            <span class="label">Top Predicted Player</span>
            <span class="value">${topPlayer ? topPlayer.predicted_rating.toFixed(2) : '-'}</span>
            <span class="player-name">${topPlayer ? topPlayer.playerName : '-'}</span>
        </div>
        <div class="stat-card">
            <span class="label">Best Form (Last 3)</span>
            <span class="value">${topAverage ? topAverage.last3_avg.toFixed(2) : '-'}</span>
            <span class="player-name">${topAverage ? topAverage.playerName : '-'}</span>
        </div>
        <div class="stat-card">
            <span class="label">Biggest Expected Jump</span>
            <span class="value" style="color:var(--success)">+${topImprover && topImprover.last_game ? (topImprover.predicted_rating - topImprover.last_game).toFixed(2) : '-'}</span>
            <span class="player-name">${topImprover ? topImprover.playerName : '-'}</span>
        </div>
    `;

    // Render roster grid
    const grid = document.getElementById("roster-grid");
    grid.innerHTML = "";

    players.forEach(p => {
        let diff = 0;
        let trendHtml = `<span class="trend flat"><i class="fa-solid fa-minus"></i> 0.0</span>`;
        if (p.last_game !== null && p.last_game !== undefined) {
            diff = p.predicted_rating - p.last_game;
            if (diff > 0.1) {
                trendHtml = `<span class="trend up"><i class="fa-solid fa-arrow-trend-up"></i> +${diff.toFixed(2)}</span>`;
            } else if (diff < -0.1) {
                trendHtml = `<span class="trend down"><i class="fa-solid fa-arrow-trend-down"></i> ${diff.toFixed(2)}</span>`;
            }
        }

        const card = document.createElement("div");
        card.className = "player-card";
        card.innerHTML = `
            <div class="player-info">
                <div class="player-name-container">
                    <h3>${p.playerName}</h3>
                    <p>Opponent ID: ${p.gameId}</p>
                </div>
                <div class="predicted-rating">${p.predicted_rating.toFixed(2)}</div>
            </div>
            
            <div class="form-metrics">
                <div class="metric">
                    <span class="label">Last Game</span>
                    <span class="val">${p.last_game ? p.last_game.toFixed(2) : 'N/A'}</span>
                </div>
                <div class="metric">
                    <span class="label">Last 3 Avg</span>
                    <span class="val">${p.last3_avg ? p.last3_avg.toFixed(2) : 'N/A'}</span>
                </div>
                <div class="metric" style="align-items: flex-end;">
                    <span class="label">Trend</span>
                    ${trendHtml}
                </div>
            </div>
        `;
        
        grid.appendChild(card);
    });
}
