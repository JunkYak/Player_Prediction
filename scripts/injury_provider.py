"""
scripts/injury_provider.py

Canonical Injury and Availability Provider for NBA Player Performance System.

Architecture:
    External Source (NBA Injury Report PDF / API)
        ↓
    InjuryProvider.get_injury_report(target_date)
        ↓
    PlayerIdentityResolver (maps names + teams -> canonical personId without surname guessing)
        ↓
    InjuryReportResult (CanonicalInjuryRecord list + status metadata)
        ↓
    Prediction Pipeline (safe merge by personId)
"""

import os
import re
import unicodedata
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import pandas as pd
from nba_api.stats.static import players, teams

logger = logging.getLogger(__name__)


# ==============================
# CANONICAL STATUS DEFINITIONS
# ==============================

class CanonicalStatus:
    ACTIVE = "Active"          # Confirmed available / not listed on valid report
    PROBABLE = "Probable"      # Uncertain (likely to play)
    QUESTIONABLE = "Questionable"  # Uncertain (50/50)
    DOUBTFUL = "Doubtful"      # Uncertain (unlikely to play)
    OUT = "Out"                # Confirmed unavailable (injury, G-League, suspension)
    UNKNOWN = "Unknown"        # Source failed or player unresolved

    STATUS_RANK = {
        ACTIVE: 0,
        "Available": 0,
        PROBABLE: 1,
        QUESTIONABLE: 2,
        DOUBTFUL: 3,
        OUT: 4,
        UNKNOWN: 5,
    }


def normalize_status_string(raw_status: Optional[str]) -> str:
    """Map arbitrary status text from external sources to CanonicalStatus."""
    if not raw_status or not isinstance(raw_status, str):
        return CanonicalStatus.UNKNOWN

    s = raw_status.strip().lower()

    if any(k in s for k in ["out", "g league", "g-league", "suspended", "not with team", "ineligible"]):
        return CanonicalStatus.OUT
    elif "doubtful" in s:
        return CanonicalStatus.DOUBTFUL
    elif "questionable" in s:
        return CanonicalStatus.QUESTIONABLE
    elif "probable" in s:
        return CanonicalStatus.PROBABLE
    elif any(k in s for k in ["available", "active"]):
        return CanonicalStatus.ACTIVE
    else:
        return CanonicalStatus.UNKNOWN


# ==============================
# CANONICAL DATA STRUCTURES
# ==============================

@dataclass
class CanonicalInjuryRecord:
    personId: Optional[int]
    playerName: str
    teamId: Optional[int]
    teamName: str
    status: str
    rawStatus: str
    reason: str
    reportDate: str
    source: str

    def to_dict(self) -> dict:
        return {
            "personId": self.personId,
            "playerName": self.playerName,
            "teamId": self.teamId,
            "teamName": self.teamName,
            "status": self.status,
            "rawStatus": self.rawStatus,
            "reason": self.reason,
            "reportDate": self.reportDate,
            "source": self.source,
        }


@dataclass
class InjuryReportResult:
    records: List[CanonicalInjuryRecord] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    reportDate: Optional[str] = None
    source: str = "nba_official_pdf"

    @property
    def is_empty(self) -> bool:
        return len(self.records) == 0

    def get_player_status_map(self) -> Dict[int, CanonicalInjuryRecord]:
        """Return mapping from personId -> CanonicalInjuryRecord for all resolved players."""
        mapping = {}
        for rec in self.records:
            if rec.personId is not None:
                mapping[rec.personId] = rec
        return mapping


# ==============================
# PLAYER IDENTITY RESOLVER
# ==============================

class PlayerIdentityResolver:
    """
    Safely resolves raw player name and team strings to official NBA personId.
    Strictly forbids surname-only matching.
    """

    def __init__(self):
        self._name_to_players: Dict[str, List[dict]] = {}
        self._team_lookup: Dict[str, int] = {}
        self._init_static_db()

    def _normalize_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        s = str(text).strip().lower()
        # Remove accents
        s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
        # Normalize punctuation / extra spaces
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def _strip_suffix(self, norm_name: str) -> str:
        """Remove common generational suffixes (jr, sr, ii, iii, iv)."""
        tokens = norm_name.split()
        if tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
            return " ".join(tokens[:-1])
        return norm_name

    def _init_static_db(self):
        try:
            all_players = players.get_players()
            for p in all_players:
                norm = self._normalize_text(p["full_name"])
                self._name_to_players.setdefault(norm, []).append(p)
                # Suffix-stripped alias
                stripped = self._strip_suffix(norm)
                if stripped != norm:
                    self._name_to_players.setdefault(stripped, []).append(p)

            all_teams = teams.get_teams()
            for t in all_teams:
                for key in [t["full_name"], t["nickname"], t["abbreviation"], t["city"]]:
                    norm_t = self._normalize_text(key)
                    if norm_t:
                        self._team_lookup[norm_t] = t["id"]
        except Exception as e:
            logger.warning(f"Failed to initialize NBA static player database: {e}")

    def resolve_team_id(self, team_str: Optional[str]) -> Optional[int]:
        if not team_str:
            return None
        norm = self._normalize_text(team_str)
        return self._team_lookup.get(norm)

    def parse_raw_name(self, raw_name: Optional[str]) -> str:
        """Convert 'Last, First' or 'First Last' into 'First Last'."""
        if not raw_name or not isinstance(raw_name, str):
            return ""
        raw_name = raw_name.strip()
        if "," in raw_name:
            parts = raw_name.split(",", 1)
            return f"{parts[1].strip()} {parts[0].strip()}"
        return raw_name

    def resolve_player(
        self,
        raw_name: Optional[str],
        team_str: Optional[str] = None
    ) -> Tuple[Optional[int], str, bool]:
        """
        Resolve a raw player name to (personId, canonical_name, is_ambiguous).

        Rules:
        1. Never match on surname alone.
        2. Full name exact normalized match -> unique ID.
        3. If multiple historical matches exist, prioritize is_active == True.
        4. If still ambiguous, do not guess -> return (None, parsed_name, True).
        """
        parsed_name = self.parse_raw_name(raw_name)
        if not parsed_name:
            return None, "", False

        norm_name = self._normalize_text(parsed_name)
        candidates = self._name_to_players.get(norm_name, [])

        if not candidates:
            # Try stripped suffix
            stripped = self._strip_suffix(norm_name)
            candidates = self._name_to_players.get(stripped, [])

        if not candidates:
            return None, parsed_name, False

        # If exactly 1 match
        if len(candidates) == 1:
            return candidates[0]["id"], candidates[0]["full_name"], False

        # Filter for active players
        active_candidates = [c for c in candidates if c.get("is_active")]
        if len(active_candidates) == 1:
            return active_candidates[0]["id"], active_candidates[0]["full_name"], False

        # Ambiguous match (e.g. multiple active players with exact same full name)
        logger.warning(
            f"Ambiguous player name '{raw_name}' ({parsed_name}) maps to multiple candidates: "
            f"{[c['id'] for c in (active_candidates or candidates)]}. Refusing to guess."
        )
        return None, parsed_name, True


# ==============================
# INJURY PROVIDER IMPLEMENTATION
# ==============================

class InjuryProvider:
    """
    Fetches and normalizes official NBA injury reports.
    Provides resilient failure handling and clean abstraction.
    """

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }

    # Standard NBA injury report release times (Eastern Time)
    RELEASE_HOURS = [17, 13, 18, 20, 8]

    def __init__(self, resolver: Optional[PlayerIdentityResolver] = None):
        self.resolver = resolver or PlayerIdentityResolver()

    def _fetch_raw_report_df(self, timestamp: datetime) -> Optional[pd.DataFrame]:
        """Fetch raw DataFrame from external NBA injury report source."""
        from nbainjuries import injury
        return injury.get_reportdata(
            timestamp,
            return_df=True,
            headerparam=self.DEFAULT_HEADERS
        )

    def get_injury_report(
        self,
        target_date: Optional[datetime] = None,
        days_back_limit: int = 3
    ) -> InjuryReportResult:
        """
        Fetch official injury report for the target date.

        If target_date is not provided, defaults to current system date.
        If no report is found for target_date, looks back up to days_back_limit days.
        If all attempts fail, returns InjuryReportResult with success=False (never raises).
        """
        base_dt = target_date or datetime.now()
        report_df = None
        used_timestamp = None
        last_error = None

        for days in range(days_back_limit + 1):
            cur_date = base_dt - timedelta(days=days)
            for hour in self.RELEASE_HOURS:
                ts = datetime(cur_date.year, cur_date.month, cur_date.day, hour, 30)
                try:
                    df = self._fetch_raw_report_df(ts)
                    if df is not None and not df.empty:
                        report_df = df
                        used_timestamp = ts
                        logger.info(f"Successfully retrieved NBA injury report for timestamp: {ts}")
                        break
                except Exception as e:
                    last_error = e
            if report_df is not None:
                break

        if report_df is None:
            err_msg = f"No valid injury report could be fetched for date {base_dt.strftime('%Y-%m-%d')} (last error: {last_error})"
            logger.warning(err_msg)
            return InjuryReportResult(
                records=[],
                success=False,
                error_message=err_msg,
                reportDate=base_dt.strftime("%Y-%m-%d"),
                source="none"
            )

        # Normalize raw DataFrame into CanonicalInjuryRecords
        records = []
        report_date_str = used_timestamp.strftime("%Y-%m-%d")

        for _, row in report_df.iterrows():
            raw_pname = row.get("Player Name")
            if not isinstance(raw_pname, str) or not raw_pname.strip():
                continue

            raw_team = str(row.get("Team", ""))
            raw_status = str(row.get("Current Status", ""))
            raw_reason = str(row.get("Reason", ""))

            person_id, canonical_name, is_ambiguous = self.resolver.resolve_player(
                raw_pname,
                raw_team
            )
            team_id = self.resolver.resolve_team_id(raw_team)
            status = normalize_status_string(raw_status)

            # If player name is ambiguous or could not be resolved, mark status as Unknown
            if is_ambiguous or person_id is None:
                final_status = CanonicalStatus.UNKNOWN if status != CanonicalStatus.OUT else CanonicalStatus.OUT
            else:
                final_status = status

            rec = CanonicalInjuryRecord(
                personId=person_id,
                playerName=canonical_name or raw_pname,
                teamId=team_id,
                teamName=raw_team,
                status=final_status,
                rawStatus=raw_status,
                reason=raw_reason,
                reportDate=report_date_str,
                source="nba_official_pdf"
            )
            records.append(rec)

        return InjuryReportResult(
            records=records,
            success=True,
            error_message=None,
            reportDate=report_date_str,
            source="nba_official_pdf"
        )
