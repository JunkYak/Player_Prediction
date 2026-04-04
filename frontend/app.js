let homeData = [];
let allData = [];
let teamsMap = new Map(); // teamId -> { name, color }

const NBA_TEAMS = {
    "1610612737": { name: "Atlanta Hawks", color: "#e03a3e" },
    "1610612738": { name: "Boston Celtics", color: "#007A33" },
    "1610612751": { name: "Brooklyn Nets", color: "#ffffff" },
    "1610612766": { name: "Charlotte Hornets", color: "#00788c" },
    "1610612741": { name: "Chicago Bulls", color: "#ce1141" },
    "1610612739": { name: "Cleveland Cavaliers", color: "#860038" },
    "1610612742": { name: "Dallas Mavericks", color: "#00538c" },
    "1610612743": { name: "Denver Nuggets", color: "#0E2240" },
    "1610612765": { name: "Detroit Pistons", color: "#C8102E" },
    "1610612744": { name: "Golden State Warriors", color: "#1D428A" },
    "1610612745": { name: "Houston Rockets", color: "#CE1141" },
    "1610612754": { name: "Indiana Pacers", color: "#FDBB30" },
    "1610612746": { name: "LA Clippers", color: "#c8102E" },
    "1610612747": { name: "Los Angeles Lakers", color: "#552583" },
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
    "1610612762": { name: "Utah Jazz", color: "#f9a01b" },
    "1610612764": { name: "Washington Wizards", color: "#002B5C" }
};

document.addEventListener("DOMContentLoaded", async () => {
    showLoading();
    try {
        // Fetch required datasets with cache busting
        const timestamp = new Date().getTime();
        const [homeRes, allRes] = await Promise.all([
            fetch(`../data/frontend_home.json?t=${timestamp}`),
            fetch(`../data/frontend_all.json?t=${timestamp}`)
        ]);

        if(!homeRes.ok || !allRes.ok) throw new Error("Could not load backend JSON");

        homeData = await homeRes.json();
        allData = await allRes.json();

        // Extract Teams from allData dynamically to prevent missing teams
        const uniqueTeams = [...new Set(allData.map(p => String(p.teamId)))];
        
        uniqueTeams.forEach(id => {
            const player = allData.find(p => String(p.teamId) === id);
            teamsMap.set(id, {
                name: player.teamName || NBA_TEAMS[id]?.name || `Team ${id}`,
                color: NBA_TEAMS[id]?.color || '#3b82f6'
            });
        });

        buildSidebar(Array.from(teamsMap.keys()));

        // Start on Home view
        goHome();
    } catch(err) {
        console.error(err);
        document.getElementById("view-title").textContent = "Data Error";
        document.getElementById("view-subtitle").textContent = "Check if generate_json.py was run recently.";
    } finally {
        hideLoading();
    }
});

function showLoading() {
    document.getElementById("loading-state").classList.add("active");
    document.getElementById("roster-section").style.opacity = '0';
}
function hideLoading() {
    document.getElementById("loading-state").classList.remove("active");
    document.getElementById("roster-section").style.opacity = '1';
}

function getRatingClass(rating) {
    if (!rating) return 'rate-low';
    if (rating >= 3.0) return 'rate-high';
    if (rating >= 1.5) return 'rate-med';
    return 'rate-low';
}

function createPlayerCard(p, isTop3 = false) {
    const defaultImg = "https://cdn.nba.com/headshots/nba/latest/260x190/logoman.png";
    const imgUrl = p.headshot ? p.headshot : defaultImg;
    const teamInfo = teamsMap.get(String(p.teamId)) || { name: p.teamName || 'N/A' };
    
    // Formatting numbers
    const predStr = p.predicted_rating ? parseFloat(p.predicted_rating).toFixed(2) : '-';
    let lgStr = p.last_game ? parseFloat(p.last_game).toFixed(2) : '-';
    let avgStr = p.last3_avg ? parseFloat(p.last3_avg).toFixed(2) : '-';

    // SVG Fallback behavior
    const onErr = `this.style.display='none'; this.nextElementSibling.style.display='flex';`;

    const topClass = isTop3 ? 'top-3' : '';
    
    return `
        <div class="player-card ${topClass}">
            <div class="pc-header">
                <div class="headshot-container">
                    <img src="${imgUrl}" class="headshot" onerror="${onErr}">
                    <div class="headshot-fallback" style="display:none;">
                        <svg><use href="#icon-user"></use></svg>
                    </div>
                </div>
                <div class="pc-info">
                    <h3>${p.playerName || 'Unknown'}</h3>
                    <p>${teamInfo.name}</p>
                    ${p.opponentName ? `<p class="opponent">vs ${p.opponentName}</p>` : ''}
                </div>
            </div>
            
            <div class="pc-ratings">
                <div class="rating-block">
                    <span class="lbl">Predicted</span>
                    <span class="rating-val ${getRatingClass(p.predicted_rating)}">${predStr}</span>
                </div>
                <div class="rating-block" style="text-align: right;">
                    <span class="lbl">Last Game</span>
                    <span class="rating-sub">${lgStr}</span>
                    <span class="rating-sm">L3 Avg: ${avgStr}</span>
                </div>
            </div>
        </div>
    `;
}

function renderGrid(players, highlightTop3 = false) {
    const grid = document.getElementById("roster-grid");
    grid.innerHTML = "";
    
    players.forEach((p, idx) => {
        grid.innerHTML += createPlayerCard(p, highlightTop3 && idx < 3);
    });
}

function resetNav() {
    document.querySelectorAll(".nav-btn, .team-btn").forEach(el => el.classList.remove("active"));
}

window.goHome = function() {
    showLoading();
    resetNav();
    document.getElementById("btn-home").classList.add("active");
    
    // Reset global theme to default
    document.documentElement.style.setProperty("--accent-color", "#3b82f6");
    
    setTimeout(() => {
        document.getElementById("view-title").textContent = "Top 10 Prediction";
        document.getElementById("view-title").style.color = "#f8fafc";
        document.getElementById("view-subtitle").textContent = "Players predicted to peak tomorrow based on recent form";
        
        renderGrid(homeData, true);
        hideLoading();
    }, 150);
}

window.viewTeam = function(teamId) {
    showLoading();
    resetNav();
    document.getElementById(`btn-${teamId}`).classList.add("active");
    
    setTimeout(() => {
        const teamInfo = teamsMap.get(String(teamId));
        
        // Dynamic theme!
        document.documentElement.style.setProperty("--accent-color", teamInfo.color);
        
        document.getElementById("view-title").textContent = teamInfo.name;
        document.getElementById("view-title").style.color = teamInfo.color;
        document.getElementById("view-subtitle").textContent = "Team Roster sorted by Predicted Form";

        const teamPlayers = allData
            .filter(p => String(p.teamId) === String(teamId))
            .sort((a,b) => parseFloat(b.predicted_rating || 0) - parseFloat(a.predicted_rating || 0));

        renderGrid(teamPlayers, false);
        hideLoading();
    }, 150);
}

function buildSidebar(teamIds) {
    const list = document.getElementById("team-list");
    list.innerHTML = "";

    const sortedTeams = teamIds
        .map(id => ({ id, ...teamsMap.get(id) }))
        .sort((a,b) => a.name.localeCompare(b.name));

    sortedTeams.forEach(team => {
        const btn = document.createElement("button");
        btn.className = "team-btn";
        btn.id = `btn-${team.id}`;
        
        btn.innerHTML = `
            <span style="width:10px; height:10px; border-radius:50%; background-color:${team.color}; box-shadow: 0 0 8px ${team.color}"></span>
            ${team.name}
        `;
        
        btn.onclick = () => viewTeam(team.id);
        list.appendChild(btn);
    });
}
