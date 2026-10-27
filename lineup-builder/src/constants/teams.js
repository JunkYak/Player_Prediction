/**
 * teams.js
 *
 * Official NBA Franchise metadata, IDs, tricodes, brand colors, and CDN logo endpoints.
 * Stable NBA team identifiers matching stats.nba.com and the serving dataset.
 */

export const NBA_TEAMS = {
  1610612737: {
    teamId: 1610612737,
    city: 'Atlanta',
    name: 'Hawks',
    fullName: 'Atlanta Hawks',
    tricode: 'ATL',
    primaryColor: '#C8102E',
    secondaryColor: '#FDB927',
  },
  1610612738: {
    teamId: 1610612738,
    city: 'Boston',
    name: 'Celtics',
    fullName: 'Boston Celtics',
    tricode: 'BOS',
    primaryColor: '#007A33',
    secondaryColor: '#BA9653',
  },
  1610612751: {
    teamId: 1610612751,
    city: 'Brooklyn',
    name: 'Nets',
    fullName: 'Brooklyn Nets',
    tricode: 'BKN',
    primaryColor: '#000000',
    secondaryColor: '#FFFFFF',
  },
  1610612766: {
    teamId: 1610612766,
    city: 'Charlotte',
    name: 'Hornets',
    fullName: 'Charlotte Hornets',
    tricode: 'CHA',
    primaryColor: '#1D1160',
    secondaryColor: '#00788C',
  },
  1610612741: {
    teamId: 1610612741,
    city: 'Chicago',
    name: 'Bulls',
    fullName: 'Chicago Bulls',
    tricode: 'CHI',
    primaryColor: '#CE1141',
    secondaryColor: '#000000',
  },
  1610612739: {
    teamId: 1610612739,
    city: 'Cleveland',
    name: 'Cavaliers',
    fullName: 'Cleveland Cavaliers',
    tricode: 'CLE',
    primaryColor: '#860038',
    secondaryColor: '#041E42',
  },
  1610612742: {
    teamId: 1610612742,
    city: 'Dallas',
    name: 'Mavericks',
    fullName: 'Dallas Mavericks',
    tricode: 'DAL',
    primaryColor: '#00538C',
    secondaryColor: '#002B5E',
  },
  1610612743: {
    teamId: 1610612743,
    city: 'Denver',
    name: 'Nuggets',
    fullName: 'Denver Nuggets',
    tricode: 'DEN',
    primaryColor: '#0E2240',
    secondaryColor: '#FEC524',
  },
  1610612765: {
    teamId: 1610612765,
    city: 'Detroit',
    name: 'Pistons',
    fullName: 'Detroit Pistons',
    tricode: 'DET',
    primaryColor: '#C8102E',
    secondaryColor: '#1D42BA',
  },
  1610612744: {
    teamId: 1610612744,
    city: 'Golden State',
    name: 'Warriors',
    fullName: 'Golden State Warriors',
    tricode: 'GSW',
    primaryColor: '#1D428A',
    secondaryColor: '#FFC72C',
  },
  1610612745: {
    teamId: 1610612745,
    city: 'Houston',
    name: 'Rockets',
    fullName: 'Houston Rockets',
    tricode: 'HOU',
    primaryColor: '#CE1141',
    secondaryColor: '#000000',
  },
  1610612754: {
    teamId: 1610612754,
    city: 'Indiana',
    name: 'Pacers',
    fullName: 'Indiana Pacers',
    tricode: 'IND',
    primaryColor: '#002D62',
    secondaryColor: '#FDBB30',
  },
  1610612746: {
    teamId: 1610612746,
    city: 'LA',
    name: 'Clippers',
    fullName: 'LA Clippers',
    tricode: 'LAC',
    primaryColor: '#C8102E',
    secondaryColor: '#1D428A',
  },
  1610612747: {
    teamId: 1610612747,
    city: 'Los Angeles',
    name: 'Lakers',
    fullName: 'Los Angeles Lakers',
    tricode: 'LAL',
    primaryColor: '#552583',
    secondaryColor: '#FDB927',
  },
  1610612763: {
    teamId: 1610612763,
    city: 'Memphis',
    name: 'Grizzlies',
    fullName: 'Memphis Grizzlies',
    tricode: 'MEM',
    primaryColor: '#5D76A9',
    secondaryColor: '#12173F',
  },
  1610612748: {
    teamId: 1610612748,
    city: 'Miami',
    name: 'Heat',
    fullName: 'Miami Heat',
    tricode: 'MIA',
    primaryColor: '#98002E',
    secondaryColor: '#F9A01B',
  },
  1610612749: {
    teamId: 1610612749,
    city: 'Milwaukee',
    name: 'Bucks',
    fullName: 'Milwaukee Bucks',
    tricode: 'MIL',
    primaryColor: '#00471B',
    secondaryColor: '#EEE1C6',
  },
  1610612750: {
    teamId: 1610612750,
    city: 'Minnesota',
    name: 'Timberwolves',
    fullName: 'Minnesota Timberwolves',
    tricode: 'MIN',
    primaryColor: '#0C2340',
    secondaryColor: '#236192',
  },
  1610612740: {
    teamId: 1610612740,
    city: 'New Orleans',
    name: 'Pelicans',
    fullName: 'New Orleans Pelicans',
    tricode: 'NOP',
    primaryColor: '#0C2340',
    secondaryColor: '#C8102E',
  },
  1610612752: {
    teamId: 1610612752,
    city: 'New York',
    name: 'Knicks',
    fullName: 'New York Knicks',
    tricode: 'NYK',
    primaryColor: '#006BB6',
    secondaryColor: '#F58426',
  },
  1610612760: {
    teamId: 1610612760,
    city: 'Oklahoma City',
    name: 'Thunder',
    fullName: 'Oklahoma City Thunder',
    tricode: 'OKC',
    primaryColor: '#007AC1',
    secondaryColor: '#EF3B24',
  },
  1610612753: {
    teamId: 1610612753,
    city: 'Orlando',
    name: 'Magic',
    fullName: 'Orlando Magic',
    tricode: 'ORL',
    primaryColor: '#0077C0',
    secondaryColor: '#C4CED4',
  },
  1610612755: {
    teamId: 1610612755,
    city: 'Philadelphia',
    name: '76ers',
    fullName: 'Philadelphia 76ers',
    tricode: 'PHI',
    primaryColor: '#006BB6',
    secondaryColor: '#ED174C',
  },
  1610612756: {
    teamId: 1610612756,
    city: 'Phoenix',
    name: 'Suns',
    fullName: 'Phoenix Suns',
    tricode: 'PHX',
    primaryColor: '#1D1160',
    secondaryColor: '#E56020',
  },
  1610612757: {
    teamId: 1610612757,
    city: 'Portland',
    name: 'Trail Blazers',
    fullName: 'Portland Trail Blazers',
    tricode: 'POR',
    primaryColor: '#E03A3E',
    secondaryColor: '#000000',
  },
  1610612758: {
    teamId: 1610612758,
    city: 'Sacramento',
    name: 'Kings',
    fullName: 'Sacramento Kings',
    tricode: 'SAC',
    primaryColor: '#5A2D81',
    secondaryColor: '#63727A',
  },
  1610612759: {
    teamId: 1610612759,
    city: 'San Antonio',
    name: 'Spurs',
    fullName: 'San Antonio Spurs',
    tricode: 'SAS',
    primaryColor: '#C4CED4',
    secondaryColor: '#000000',
  },
  1610612761: {
    teamId: 1610612761,
    city: 'Toronto',
    name: 'Raptors',
    fullName: 'Toronto Raptors',
    tricode: 'TOR',
    primaryColor: '#CE1141',
    secondaryColor: '#000000',
  },
  1610612762: {
    teamId: 1610612762,
    city: 'Utah',
    name: 'Jazz',
    fullName: 'Utah Jazz',
    tricode: 'UTA',
    primaryColor: '#002B5C',
    secondaryColor: '#00471B',
  },
  1610612764: {
    teamId: 1610612764,
    city: 'Washington',
    name: 'Wizards',
    fullName: 'Washington Wizards',
    tricode: 'WAS',
    primaryColor: '#002B5C',
    secondaryColor: '#E31837',
  },
};

/**
 * Returns official CDN vector logo URL for a team ID.
 */
export function getTeamLogoUrl(teamId) {
  if (!teamId) return null;
  return `https://cdn.nba.com/logos/nba/${teamId}/global/L/logo.svg`;
}

/**
 * Resolves full team information given a teamId or teamName.
 */
export function getTeamInfo(teamId, fallbackName = '') {
  const numericId = Number(teamId);
  if (numericId && NBA_TEAMS[numericId]) {
    return NBA_TEAMS[numericId];
  }

  // Fallback by matching team name / tricode if teamId is missing or unknown
  if (fallbackName) {
    const clean = fallbackName.trim().toLowerCase();
    for (const team of Object.values(NBA_TEAMS)) {
      if (
        team.name.toLowerCase() === clean ||
        team.fullName.toLowerCase() === clean ||
        team.tricode.toLowerCase() === clean
      ) {
        return team;
      }
    }
  }

  return {
    teamId: numericId || 0,
    city: '',
    name: fallbackName || 'NBA',
    fullName: fallbackName || 'NBA Team',
    tricode: (fallbackName || 'NBA').slice(0, 3).toUpperCase(),
    primaryColor: '#f97316',
    secondaryColor: '#1e1e1e',
  };
}
