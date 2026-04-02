const teamNames = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612739: "Cleveland Cavaliers",
    1610612740: "New Orleans Pelicans",
    1610612741: "Chicago Bulls",
    1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",
    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",
    1610612750: "Minnesota Timberwolves",
    1610612751: "Brooklyn Nets",
    1610612752: "New York Knicks",
    1610612753: "Orlando Magic",
    1610612754: "Indiana Pacers",
    1610612755: "Philadelphia 76ers",
    1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",
    1610612760: "Oklahoma City Thunder",
    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",
    1610612763: "Memphis Grizzlies",
    1610612764: "Washington Wizards",
    1610612765: "Detroit Pistons",
    1610612766: "Charlotte Hornets"
};

async function getPrediction() {
    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "⏳ Loading...";

    try {
        const response = await fetch("http://127.0.0.1:5000/predict");
        const data = await response.json();

        let html = "<h2>🏆 Top 10 Players</h2>";

        data.players.slice(0, 10).forEach(player => {
            html += `
            <div class="card">
                <h3>${player.name}</h3>
                <p>Team ID: ${player.team}</p>
                <p>Rating: ${player.rating.toFixed(2)}</p>
            </div>
        `;
        });

        resultDiv.innerHTML = html;

    } catch (err) {
        resultDiv.innerHTML = "❌ Error fetching prediction";
    }

}

let teamsData = {};  // store globally

async function loadTeams() {
    const res = await fetch("http://127.0.0.1:5000/teams");
    const data = await res.json();

    teamsData = data;

    const select = document.getElementById("teamSelect");
    select.innerHTML = `<option value="">-- Choose a Team --</option>`;

    Object.keys(data).forEach(teamId => {
        const option = document.createElement("option");
        option.value = teamId;
        option.text = teamNames[teamId] || teamId;

        select.appendChild(option);
    });
}

function showPlayers(teamId, players) {
    const container = document.getElementById("playersContainer");

    container.innerHTML = `<h2>Top Players - ${teamNames[teamId] || teamId}</h2>`;

    players.forEach(player => {
        container.innerHTML += `
        <div class="card">
            <h3>${player.name}</h3>
            <p>Rating: ${player.rating.toFixed(2)}</p>
        </div>
    `;
    });

}

function selectTeam() {
    const teamId = document.getElementById("teamSelect").value;

    if (!teamId) return;

    const players = teamsData[teamId];

    const container = document.getElementById("playersContainer");

    container.innerHTML = `<h2>⭐ Top Players - ${teamNames[teamId]}</h2>`;

    players.forEach(player => {
        container.innerHTML += `
            <div class="card">
                <h3>${player.name}</h3>
                <p>Rating: ${player.rating.toFixed(2)}</p>
            </div>
        `;
    });
}

window.onload = function () {
    loadTeams();
};
