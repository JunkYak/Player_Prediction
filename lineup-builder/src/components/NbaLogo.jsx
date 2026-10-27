import { useState } from 'react';
import { getTeamLogoUrl, getTeamInfo } from '../constants/teams';

/**
 * NbaLogo
 *
 * Renders official vector SVG logo from NBA CDN with an inline SVG badge fallback.
 * Uses stable teamId and handles load errors gracefully without breaking the layout.
 */
export default function NbaLogo({ teamId, teamName, size = 40, className = '', style = {} }) {
  const [hasError, setHasError] = useState(false);
  const info = getTeamInfo(teamId, teamName);
  const logoUrl = getTeamLogoUrl(teamId);

  if (!logoUrl || hasError) {
    return (
      <div
        className={`nba-logo-fallback ${className}`}
        style={{
          width: size,
          height: size,
          minWidth: size,
          minHeight: size,
          borderRadius: 8,
          backgroundColor: '#161616',
          border: '1px solid #262626',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: Math.max(10, Math.floor(size * 0.32)),
          fontWeight: 700,
          color: info.primaryColor || '#f97316',
          letterSpacing: '0.04em',
          userSelect: 'none',
          ...style,
        }}
        title={info.fullName}
        aria-label={`${info.fullName} logo`}
      >
        {info.tricode}
      </div>
    );
  }

  return (
    <img
      src={logoUrl}
      alt={`${info.fullName} logo`}
      className={`nba-team-logo ${className}`}
      width={size}
      height={size}
      loading="lazy"
      onError={() => setHasError(true)}
      style={{
        width: size,
        height: size,
        minWidth: size,
        minHeight: size,
        objectFit: 'contain',
        display: 'block',
        filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.4))',
        ...style,
      }}
    />
  );
}
