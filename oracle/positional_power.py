"""
oracle/positional_power.py — Detailed positional power analyzer.

BUSINESS SUMMARY
----------------
This module provides position-by-position breakdowns for all 32 national
teams, including named players with individual ratings. It answers questions
like: "How much weaker does France get if Mbappé is injured?" and "Which
teams rely dangerously on a single player?" Those fragility signals feed
directly into match simulation to make results more realistic.

DEVELOPER NOTES
---------------
Data structure: POSITIONAL_DATA in team_strength.py is the single source of
truth for player ratings. This module wraps that data with analytical
methods — depth scores, dependency indices, and injury impact simulations.

Key metrics:
  - depth_score: ratio of backup to starter rating (0 = no backup, 1 = equal)
  - key_player_dependency: max(starter - backup) / sum(all starters); higher
    means team is more fragile to injury
  - injury_impact: composite score drop modelled as starter_rating replaced
    by backup_rating, re-weighted by position importance

Complexity: O(P) per team where P = 6 positions. All lookups are O(1).
"""

from __future__ import annotations

import logging
from typing import Optional

from config import POSITION_WEIGHTS
from oracle.team_strength import POSITIONAL_DATA

logger = logging.getLogger(__name__)

# Injury replacement quality factors — how much a backup can cover
# in an emergency (accounts for positional versatility and squad depth)
EMERGENCY_COVER_FACTOR: dict[str, float] = {
    "GK": 0.85,   # GK position least covered by non-specialists
    "CB": 0.90,
    "FB": 0.88,
    "CM": 0.87,
    "AM": 0.82,   # creative positions hardest to replace
    "FW": 0.84,
}


class PositionalPowerAnalyzer:
    """
    Analyzes positional quality, squad depth, and injury fragility for
    all 32 2026 World Cup national teams.

    Uses POSITIONAL_DATA from oracle.team_strength as its data source —
    one authoritative dict mapping team → position → {rating, starter, backup}.

    Parameters
    ----------
    None

    Methods
    -------
    get_position_rating(team, position)        → float
    get_team_depth_score(team)                 → float   [0, 1]
    get_key_player_dependency(team)            → float   [0, 1]
    simulate_injury_impact(team, position)     → float   composite score drop
    get_squad_summary(team)                    → dict
    rank_teams_by_depth()                      → list[tuple[str, float]]
    rank_teams_by_dependency()                 → list[tuple[str, float]]
    """

    def __init__(self) -> None:
        self._data = POSITIONAL_DATA

    # ------------------------------------------------------------------
    # Basic lookups
    # ------------------------------------------------------------------

    def get_position_rating(self, team: str, position: str) -> float:
        """
        Return the overall rating for a position group within a team.

        Parameters
        ----------
        team : str
            National team name (must match POSITIONAL_DATA keys).
        position : str
            Position group: GK | CB | FB | CM | AM | FW.

        Returns
        -------
        float
            Position rating on 0–100 scale. Returns 65.0 if data missing.
        """
        team_data = self._data.get(team)
        if team_data is None:
            logger.warning("No positional data for '%s'.", team)
            return 65.0
        pos_data = team_data.get(position)
        if pos_data is None:
            logger.warning("No '%s' data for team '%s'.", position, team)
            return 65.0
        return float(pos_data["rating"])

    def get_starter(self, team: str, position: str) -> str:
        """Return the named starter at a position, e.g. 'Alisson (91)'."""
        return self._data.get(team, {}).get(position, {}).get("starter", "Unknown")

    def get_backup(self, team: str, position: str) -> str:
        """Return the named backup at a position."""
        return self._data.get(team, {}).get(position, {}).get("backup", "Unknown")

    def _parse_rating_from_label(self, label: str) -> float:
        """
        Extract numeric rating from a player label like 'Alisson (91)'.

        Parameters
        ----------
        label : str   e.g. "Alisson (91)"

        Returns
        -------
        float   Extracted rating, or 65.0 if parsing fails.
        """
        try:
            return float(label.split("(")[1].rstrip(")"))
        except (IndexError, ValueError):
            return 65.0

    # ------------------------------------------------------------------
    # Depth analysis
    # ------------------------------------------------------------------

    def get_team_depth_score(self, team: str) -> float:
        """
        Measure overall squad depth as the ratio of backup to starter quality,
        averaged across all six position groups weighted by importance.

        A score of 1.0 means backups are as good as starters (maximum depth).
        A score of 0.0 means the team has no viable backups.

        Algorithm:
          For each position:
            depth_pos = backup_rating / starter_rating
          depth_team = weighted_average(depth_pos, POSITION_WEIGHTS)

        Parameters
        ----------
        team : str

        Returns
        -------
        float
            Weighted depth score in [0, 1].
        """
        team_data = self._data.get(team)
        if team_data is None:
            return 0.60  # league average fallback

        weighted_depth = 0.0
        for pos, weight in POSITION_WEIGHTS.items():
            pos_data = team_data.get(pos, {})
            starter_r = float(pos_data.get("rating", 70.0))
            backup_label = pos_data.get("backup", "Unknown (65)")
            backup_r = self._parse_rating_from_label(backup_label)
            if starter_r > 0:
                depth = min(backup_r / starter_r, 1.0)
            else:
                depth = 0.0
            weighted_depth += weight * depth

        return round(float(weighted_depth), 4)

    # ------------------------------------------------------------------
    # Key-player dependency
    # ------------------------------------------------------------------

    def get_key_player_dependency(self, team: str) -> float:
        """
        Quantify how much the team relies on its single best player.

        High dependency (→ 1.0) means one injury could severely degrade
        the team. Low dependency (→ 0.0) means balanced, resilient squad.

        Algorithm:
          gap_pos = max(0, starter_rating - backup_rating) for each position
          dependency = max(gap_pos × position_weight) / sum(starter_ratings × weight)
          Scaled and normalised to [0, 1].

        Parameters
        ----------
        team : str

        Returns
        -------
        float
            Key-player dependency index in [0, 1].
            Argentina (Messi) and Portugal (Ronaldo) score highest.
        """
        team_data = self._data.get(team)
        if team_data is None:
            return 0.30

        max_gap_weighted = 0.0
        total_weighted_rating = 0.0

        for pos, weight in POSITION_WEIGHTS.items():
            pos_data = team_data.get(pos, {})
            starter_r = float(pos_data.get("rating", 70.0))
            backup_label = pos_data.get("backup", "Unknown (65)")
            backup_r = self._parse_rating_from_label(backup_label)

            gap = max(0.0, starter_r - backup_r)
            max_gap_weighted = max(max_gap_weighted, gap * weight)
            total_weighted_rating += starter_r * weight

        if total_weighted_rating == 0:
            return 0.0

        dependency = max_gap_weighted / (total_weighted_rating / 10.0)
        return round(min(float(dependency), 1.0), 4)

    # ------------------------------------------------------------------
    # Injury simulation
    # ------------------------------------------------------------------

    def simulate_injury_impact(
        self,
        team: str,
        injured_position: str,
        injury_severity: float = 1.0,
    ) -> float:
        """
        Estimate the drop in composite positional power when the starter
        at a given position is unavailable.

        The backup steps in, but at reduced effectiveness modelled by
        EMERGENCY_COVER_FACTOR[position] — a physiological/tactical penalty
        reflecting that even a high-rated backup disrupts team chemistry.

        Parameters
        ----------
        team : str
        injured_position : str
            GK | CB | FB | CM | AM | FW
        injury_severity : float
            1.0 = fully out, 0.5 = limited to 50% capacity.

        Returns
        -------
        float
            Drop in the team's normalised positional power score (positive = worse).
            e.g. 0.032 means the team's composite score falls by ~3.2 points.
        """
        team_data = self._data.get(team)
        if team_data is None:
            return 0.0

        pos_data = team_data.get(injured_position, {})
        starter_r = float(pos_data.get("rating", 70.0))
        backup_label = pos_data.get("backup", "Unknown (65)")
        backup_r = self._parse_rating_from_label(backup_label)

        cover = EMERGENCY_COVER_FACTOR.get(injured_position, 0.87)
        effective_backup = backup_r * cover

        # Rating drop at this position
        rating_drop = max(0.0, starter_r - effective_backup) * injury_severity

        # Scale to [0, 1] normalised positional power
        position_weight = POSITION_WEIGHTS.get(injured_position, 0.15)
        normalised_drop = (rating_drop / 100.0) * position_weight

        return round(float(normalised_drop), 6)

    def simulate_multiple_injuries(
        self, team: str, injured_positions: list[str]
    ) -> float:
        """
        Sum impact of multiple simultaneous injuries (worst-case scenario).

        Parameters
        ----------
        team : str
        injured_positions : list[str]   List of position codes.

        Returns
        -------
        float  Total normalised positional power drop.
        """
        return sum(
            self.simulate_injury_impact(team, pos) for pos in injured_positions
        )

    # ------------------------------------------------------------------
    # Squad summary
    # ------------------------------------------------------------------

    def get_squad_summary(self, team: str) -> dict:
        """
        Return a complete analytical breakdown for a team's squad.

        Parameters
        ----------
        team : str

        Returns
        -------
        dict with keys:
          team, depth_score, key_player_dependency,
          positions (dict of position → {rating, starter, backup, depth}),
          most_fragile_position (str),
          key_player (str — name of most impactful player)
        """
        team_data = self._data.get(team)
        if team_data is None:
            return {"team": team, "error": "No data available"}

        positions_out: dict[str, dict] = {}
        best_impact = 0.0
        key_player = "Unknown"
        most_fragile = "GK"
        max_fragility = 0.0

        for pos in POSITION_WEIGHTS:
            pos_data = team_data.get(pos, {})
            starter_r = float(pos_data.get("rating", 65.0))
            backup_label = pos_data.get("backup", "Unknown (65)")
            backup_r = self._parse_rating_from_label(backup_label)
            depth = round(min(backup_r / max(starter_r, 1.0), 1.0), 3)

            # Track most fragile position
            fragility = (starter_r - backup_r) * POSITION_WEIGHTS[pos]
            if fragility > max_fragility:
                max_fragility = fragility
                most_fragile = pos

            # Track key player (highest starter rating)
            if starter_r > best_impact:
                best_impact = starter_r
                key_player = pos_data.get("starter", "Unknown")

            positions_out[pos] = {
                "rating":  starter_r,
                "starter": pos_data.get("starter", "Unknown"),
                "backup":  backup_label,
                "depth":   depth,
                "injury_impact": self.simulate_injury_impact(team, pos),
            }

        return {
            "team":                 team,
            "depth_score":          self.get_team_depth_score(team),
            "key_player_dependency": self.get_key_player_dependency(team),
            "positions":            positions_out,
            "most_fragile_position": most_fragile,
            "key_player":           key_player,
        }

    # ------------------------------------------------------------------
    # Ranking utilities
    # ------------------------------------------------------------------

    def rank_teams_by_depth(self) -> list[tuple[str, float]]:
        """
        Rank all teams by squad depth score (descending).

        Returns
        -------
        list[tuple[str, float]]  [(team, depth_score), ...] sorted best-first.
        """
        scores = [(team, self.get_team_depth_score(team)) for team in self._data]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def rank_teams_by_dependency(self) -> list[tuple[str, float]]:
        """
        Rank all teams by key-player dependency (descending = most fragile).

        Returns
        -------
        list[tuple[str, float]]  [(team, dependency), ...] sorted highest-first.
        """
        scores = [(team, self.get_key_player_dependency(team)) for team in self._data]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def named_injury_scenario(
        self, team: str, player_name: str, position: str
    ) -> str:
        """
        Generate a human-readable injury impact statement.

        Parameters
        ----------
        team : str
        player_name : str  e.g. "Mbappé"
        position : str

        Returns
        -------
        str  Plain-English impact description.
        """
        impact = self.simulate_injury_impact(team, position)
        team_data = self._data.get(team, {})
        backup = team_data.get(position, {}).get("backup", "a backup")
        current_score = sum(
            team_data.get(p, {}).get("rating", 65.0) * w / 100.0
            for p, w in POSITION_WEIGHTS.items()
        )
        adjusted = max(0.0, current_score - impact)
        pct_drop = (impact / max(current_score, 0.001)) * 100

        return (
            f"If {player_name} ({position}) is injured, {team}'s positional power "
            f"score drops from {current_score:.3f} to {adjusted:.3f} "
            f"(−{pct_drop:.1f}%). {backup} would step in, "
            f"but at reduced effectiveness (emergency cover factor: "
            f"{EMERGENCY_COVER_FACTOR.get(position, 0.87):.0%})."
        )
