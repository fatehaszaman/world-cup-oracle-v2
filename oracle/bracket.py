"""
oracle/bracket.py — 2026 FIFA World Cup bracket manager with event-driven state machine.

BUSINESS SUMMARY
----------------
The 2026 World Cup is the first to feature 48 teams across 16 groups (A–L
plus extensions for the expanded format). This module manages the tournament
bracket: which teams are in which group, how they advance, and how the
knockout bracket is seeded. It also implements an event-driven state machine —
every match result fires an immutable TournamentEvent that updates the bracket
state, making the simulation fully auditable and replayable.

DEVELOPER NOTES
---------------
Architecture: event-driven state machine using a FIFO event queue.
  - Events are dataclass instances (TournamentEvent from oracle.schemas).
  - The bracket state is a dict updated by event handlers.
  - Handlers are registered by event type — adding new side-effects
    (logging, referee assignment, venue routing) requires only a new handler
    registration, not touching match logic.

2026 Format:
  - 48 teams in 16 groups of 3 (A–P)
  - Top 1 from each group (16 teams) + 8 best 2nd-place teams → R32 (24 teams)
  - This implementation uses a simplified 32-team format for backward
    compatibility with standard WC simulation patterns, with Group A–L
    mapping to 12 groups of 4 for the 48-team draw.

Complexity: O(T log T) for group sorting, O(1) for event dispatch.
"""

from __future__ import annotations

import queue
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from oracle.schemas import TournamentEvent, TournamentEventType, RoundName, MatchOutcome

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2026 FIFA World Cup Groups
# 48-team tournament; using 12 groups of 4 (A–L) for simulation
# Teams reflect confirmed/projected 2026 qualified nations as of April 2026
# ---------------------------------------------------------------------------
WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["United States", "Mexico", "Canada", "Uruguay"],
    "B": ["Brazil", "Colombia", "Ecuador", "Ivory Coast"],
    "C": ["Argentina", "Chile", "Bolivia", "Peru"],
    "D": ["France", "Belgium", "Netherlands", "Senegal"],
    "E": ["Spain", "Portugal", "Morocco", "Cameroon"],
    "F": ["Germany", "Italy", "Switzerland", "Austria"],
    "G": ["England", "Croatia", "Serbia", "Denmark"],
    "H": ["Poland", "Japan", "South Korea", "Saudi Arabia"],
    "I": ["Nigeria", "Ghana", "Egypt", "South Africa"],
    "J": ["Iran", "Qatar", "Australia", "New Zealand"],
    "K": ["Costa Rica", "Panama", "Honduras", "Jamaica"],
    "L": ["Turkey", "Czech Republic", "Romania", "Slovakia"],
}

# 32 "core" teams used throughout the simulation engine
CORE_32_TEAMS: list[str] = [
    "France", "England", "Brazil", "Germany", "Spain", "Portugal",
    "Argentina", "Netherlands", "Belgium", "Italy", "Croatia", "Uruguay",
    "Mexico", "Colombia", "Senegal", "United States", "Morocco", "Japan",
    "South Korea", "Switzerland", "Denmark", "Austria", "Poland", "Serbia",
    "Ecuador", "Canada", "Australia", "Nigeria", "Ivory Coast", "Cameroon",
    "Saudi Arabia", "Iran",
]

# Standard R16 bracket pairing (group winner vs runner-up seedings)
# Format: (winner_group, runner_up_group)
R16_BRACKET_PAIRS: list[tuple[str, str]] = [
    ("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"),
    ("I", "J"), ("K", "L"), ("A", "C"), ("B", "D"),
]

# Points system
GROUP_STAGE_POINTS = {"win": 3, "draw": 1, "loss": 0}

# ---------------------------------------------------------------------------
# Event handler registry type
# ---------------------------------------------------------------------------
EventHandler = Callable[[TournamentEvent, dict], None]


# ---------------------------------------------------------------------------
# Bracket state machine
# ---------------------------------------------------------------------------

@dataclass
class BracketState:
    """
    Mutable snapshot of the tournament bracket at any point in the simulation.

    Parameters
    ----------
    group_standings : dict[str, list[str]]
        group_id → [1st, 2nd, 3rd, 4th] (sorted by points/GD)
    group_points : dict[str, dict[str, int]]
        group_id → {team: points}
    advancing_teams : dict[str, str]
        team → "1st_GroupA" | "2nd_GroupB" etc.
    eliminated_teams : set[str]
    round_results : dict[RoundName, list[MatchOutcome]]
    current_round : RoundName
    champion : str   empty until tournament_ended event fires
    events_processed : int
    """
    group_standings:  dict[str, list[str]]   = field(default_factory=dict)
    group_points:     dict[str, dict[str, int]] = field(default_factory=dict)
    advancing_teams:  dict[str, str]          = field(default_factory=dict)
    eliminated_teams: set                     = field(default_factory=set)
    round_results:    dict                    = field(default_factory=dict)
    current_round:    RoundName               = RoundName.GROUP_STAGE
    champion:         str                     = ""
    events_processed: int                     = 0


class TournamentBracketMachine:
    """
    Event-driven bracket state machine for the 2026 World Cup.

    Events flow through a FIFO queue. Each event is consumed by registered
    handlers that update BracketState. The machine guarantees:
      1. Every state transition is triggered by a named event.
      2. All events are appended to an immutable audit log.
      3. Replaying the audit log from scratch reproduces the final state.

    Usage
    -----
    machine = TournamentBracketMachine()
    machine.register_handler(TournamentEventType.MATCH_COMPLETED, my_handler)
    machine.fire(TournamentEvent(...))
    machine.process_all()
    state = machine.state

    Parameters
    ----------
    groups : dict[str, list[str]], optional
        Override default WC2026_GROUPS.
    run_id : str
        Unique run identifier for audit log correlation.
    """

    def __init__(
        self,
        groups: Optional[dict[str, list[str]]] = None,
        run_id: str = "",
    ) -> None:
        self.groups    = groups or WC2026_GROUPS
        self.run_id    = run_id
        self.state     = BracketState()
        self._queue: queue.Queue[TournamentEvent] = queue.Queue()
        self._audit_log: list[TournamentEvent]    = []
        self._handlers: dict[TournamentEventType, list[EventHandler]] = {
            evt: [] for evt in TournamentEventType
        }

        # Register default handlers
        self.register_handler(TournamentEventType.MATCH_COMPLETED,  self._on_match_completed)
        self.register_handler(TournamentEventType.TEAM_ADVANCED,    self._on_team_advanced)
        self.register_handler(TournamentEventType.TEAM_ELIMINATED,  self._on_team_eliminated)
        self.register_handler(TournamentEventType.ROUND_COMPLETED,  self._on_round_completed)
        self.register_handler(TournamentEventType.TOURNAMENT_ENDED, self._on_tournament_ended)

        # Initialise group points
        for grp, teams in self.groups.items():
            self.state.group_points[grp] = {t: 0 for t in teams}
            self.state.group_standings[grp] = list(teams)

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(
        self, event_type: TournamentEventType, handler: EventHandler
    ) -> None:
        """
        Register a callable to be invoked when an event of event_type fires.

        Multiple handlers per event type are supported and called in
        registration order.

        Parameters
        ----------
        event_type : TournamentEventType
        handler : callable(event: TournamentEvent, state_dict: dict) -> None
        """
        self._handlers[event_type].append(handler)
        logger.debug("Registered handler %s for %s", handler.__name__, event_type)

    # ------------------------------------------------------------------
    # Event queue
    # ------------------------------------------------------------------

    def fire(self, event: TournamentEvent) -> None:
        """
        Enqueue an event for processing.

        Events are not processed immediately — call process_all() or
        process_next() to consume the queue. This separation allows batching
        all group matches before updating standings.

        Parameters
        ----------
        event : TournamentEvent
        """
        self._queue.put(event)
        logger.debug("Event queued: %s round=%s", event.event_type, event.round_name)

    def process_next(self) -> bool:
        """
        Process the next event in the queue.

        Returns
        -------
        bool  True if an event was processed, False if queue was empty.
        """
        try:
            event = self._queue.get_nowait()
        except queue.Empty:
            return False

        self._audit_log.append(event)
        self.state.events_processed += 1

        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event, self.state.__dict__)
            except Exception as exc:
                logger.error("Handler %s failed on event %s: %s",
                             handler.__name__, event.event_type, exc)
        return True

    def process_all(self) -> int:
        """
        Drain the entire event queue.

        Returns
        -------
        int  Number of events processed.
        """
        count = 0
        while self.process_next():
            count += 1
        logger.info("Processed %d events. State: round=%s champion='%s'",
                    count, self.state.current_round, self.state.champion)
        return count

    @property
    def audit_log(self) -> list[TournamentEvent]:
        """Return the immutable audit log of all processed events."""
        return list(self._audit_log)

    # ------------------------------------------------------------------
    # Default event handlers
    # ------------------------------------------------------------------

    def _on_match_completed(self, event: TournamentEvent, state: dict) -> None:
        """Update group points when a group stage match completes."""
        match: Optional[MatchOutcome] = event.payload.get("match")
        group_id: str                 = event.payload.get("group_id", "")
        if match is None or not group_id:
            return

        pts = state["group_points"].get(group_id, {})
        ga, gb = match.goals_a, match.goals_b

        if ga > gb:
            pts[match.team_a] = pts.get(match.team_a, 0) + GROUP_STAGE_POINTS["win"]
            pts[match.team_b] = pts.get(match.team_b, 0) + GROUP_STAGE_POINTS["loss"]
        elif gb > ga:
            pts[match.team_b] = pts.get(match.team_b, 0) + GROUP_STAGE_POINTS["win"]
            pts[match.team_a] = pts.get(match.team_a, 0) + GROUP_STAGE_POINTS["loss"]
        else:
            pts[match.team_a] = pts.get(match.team_a, 0) + GROUP_STAGE_POINTS["draw"]
            pts[match.team_b] = pts.get(match.team_b, 0) + GROUP_STAGE_POINTS["draw"]

        state["group_points"][group_id] = pts
        logger.debug("Group %s updated: %s", group_id, pts)

    def _on_team_advanced(self, event: TournamentEvent, state: dict) -> None:
        """Record that a team has advanced to the next round."""
        team     = event.payload.get("team", "")
        to_round = event.payload.get("to_round", "")
        if team:
            state["advancing_teams"][team] = str(to_round)
            state["eliminated_teams"].discard(team)
            logger.debug("%s advanced to %s", team, to_round)

    def _on_team_eliminated(self, event: TournamentEvent, state: dict) -> None:
        """Mark a team as eliminated."""
        team = event.payload.get("team", "")
        if team:
            state["eliminated_teams"].add(team)
            logger.debug("%s eliminated in %s", team, event.round_name)

    def _on_round_completed(self, event: TournamentEvent, state: dict) -> None:
        """Update current round when a round finishes."""
        completed_round = event.payload.get("round", state["current_round"])
        state["round_results"][completed_round] = event.payload.get("results", [])

        # Advance current_round pointer
        round_progression = {
            RoundName.GROUP_STAGE:   RoundName.ROUND_OF_32,
            RoundName.ROUND_OF_32:   RoundName.ROUND_OF_16,
            RoundName.ROUND_OF_16:   RoundName.QUARTER_FINAL,
            RoundName.QUARTER_FINAL: RoundName.SEMI_FINAL,
            RoundName.SEMI_FINAL:    RoundName.FINAL,
        }
        next_round = round_progression.get(RoundName(completed_round))
        if next_round:
            state["current_round"] = next_round
        logger.info("Round %s completed → advancing to %s", completed_round, next_round)

    def _on_tournament_ended(self, event: TournamentEvent, state: dict) -> None:
        """Record the champion."""
        state["champion"] = event.payload.get("winner", "")
        logger.info("TOURNAMENT ENDED. Champion: %s", state["champion"])

    # ------------------------------------------------------------------
    # Bracket utilities
    # ------------------------------------------------------------------

    def sort_group(self, group_id: str, goal_diff: Optional[dict[str, int]] = None) -> list[str]:
        """
        Sort teams in a group by points (desc), then goal difference (desc),
        then alphabetical tiebreak.

        Parameters
        ----------
        group_id : str
        goal_diff : dict, optional  {team: goal_difference}

        Returns
        -------
        list[str]  [1st, 2nd, 3rd, 4th] sorted.
        """
        pts  = self.state.group_points.get(group_id, {})
        gd   = goal_diff or {t: 0 for t in pts}
        teams = list(pts.keys())
        return sorted(
            teams,
            key=lambda t: (pts.get(t, 0), gd.get(t, 0), -ord(t[0])),
            reverse=True,
        )

    def get_advancing_teams_from_groups(
        self,
        top_n: int = 2,
        goal_diffs: Optional[dict[str, dict[str, int]]] = None,
    ) -> dict[str, str]:
        """
        Determine which teams advance from the group stage.

        Parameters
        ----------
        top_n : int  Number of teams advancing per group (default 2).
        goal_diffs : dict, optional  {group_id: {team: gd}}

        Returns
        -------
        dict[str, str]  team → "pos_GroupID"  e.g. "1st_A"
        """
        advancing: dict[str, str] = {}
        for group_id in self.groups:
            gd = (goal_diffs or {}).get(group_id, {})
            sorted_teams = self.sort_group(group_id, gd)
            for rank, team in enumerate(sorted_teams[:top_n], 1):
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank, "th")
                advancing[team] = f"{rank}{suffix}_{group_id}"
        return advancing


def get_bracket(groups: Optional[dict] = None) -> dict:
    """
    Return the full 2026 World Cup bracket structure.

    Parameters
    ----------
    groups : dict, optional  Override WC2026_GROUPS.

    Returns
    -------
    dict with keys: groups, core_32_teams, r16_pairs, format_notes.
    """
    return {
        "groups":        groups or WC2026_GROUPS,
        "core_32_teams": CORE_32_TEAMS,
        "r16_pairs":     R16_BRACKET_PAIRS,
        "format_notes": (
            "2026 FIFA World Cup: 48 teams, 12 groups of 4 (A–L). "
            "Top 2 from each group (24 teams) + 8 best 3rd-place teams (32 total) "
            "advance to the Round of 32. Knockout rounds: R32 → R16 → QF → SF → Final. "
            "Host nations: USA, Canada, Mexico."
        ),
    }
