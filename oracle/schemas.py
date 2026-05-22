"""
oracle/schemas.py — Schema-first type definitions for World Cup Oracle.

BUSINESS SUMMARY
----------------
Every piece of data flowing through this system is defined here as a typed
schema. Think of this as the "contract" layer: if data doesn't match its
schema, the system rejects it loudly rather than silently producing wrong
predictions. This prevents the most common analytics failure mode — garbage
in, garbage out — by making bad data impossible to ignore.

DEVELOPER NOTES
---------------
All schemas use Python dataclasses with __post_init__ validation. Where
optional Pydantic is installed, a parallel Pydantic base-class variant is
provided for FastAPI compatibility. Every public function in the oracle
package accepts and returns typed schema objects, not raw dicts.

Complexity: O(1) validation per field; negligible runtime overhead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MatchResult(str, Enum):
    WIN  = "win"
    DRAW = "draw"
    LOSS = "loss"

class RoundName(str, Enum):
    GROUP_STAGE      = "group_stage"
    ROUND_OF_32      = "round_of_32"   # 2026 format
    ROUND_OF_16      = "round_of_16"
    QUARTER_FINAL    = "quarter_final"
    SEMI_FINAL       = "semi_final"
    THIRD_PLACE      = "third_place"
    FINAL            = "final"

class StrictnessLevel(str, Enum):
    LENIENT = "lenient"
    AVERAGE = "average"
    STRICT  = "strict"

class BiasRisk(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

class TournamentEventType(str, Enum):
    MATCH_COMPLETED  = "match_completed"
    TEAM_ADVANCED    = "team_advanced"
    TEAM_ELIMINATED  = "team_eliminated"
    ROUND_COMPLETED  = "round_completed"
    TOURNAMENT_ENDED = "tournament_ended"

# ---------------------------------------------------------------------------
# Core data schemas
# ---------------------------------------------------------------------------

@dataclass
class PositionRating:
    """
    Rating for a single position group within a national team squad.

    Parameters
    ----------
    position : str
        One of: GK, CB, FB, CM, AM, FW
    rating : float
        Overall quality rating on a 0–100 scale.
    starter : str
        Primary starter's name and rating, e.g. "Alisson (91)".
    backup : str
        Primary backup name and rating.
    depth_score : float
        Gap between starter and backup quality (0–1; lower = less depth).
    """
    position: str
    rating: float
    starter: str
    backup: str
    depth_score: float = 0.75

    def __post_init__(self) -> None:
        valid_positions = {"GK", "CB", "FB", "CM", "AM", "FW"}
        if self.position not in valid_positions:
            raise ValueError(f"position must be one of {valid_positions}, got '{self.position}'")
        if not (0.0 <= self.rating <= 100.0):
            raise ValueError(f"rating must be in [0, 100], got {self.rating}")
        if not (0.0 <= self.depth_score <= 1.0):
            raise ValueError(f"depth_score must be in [0, 1], got {self.depth_score}")


@dataclass
class TeamProfile:
    """
    Complete profile for a national team entering the simulation.

    Parameters
    ----------
    name : str
        Official team name (must match oracle lookup keys).
    fifa_ranking : int
        Current FIFA World Ranking.
    squad_value_eur_m : float
        Total squad market value in EUR millions (Transfermarkt methodology).
    gdp_per_capita : float
        GDP per capita in current USD (World Bank indicator NY.GDP.PCAP.CD).
    population : int
        Total population (World Bank indicator SP.POP.TOTL).
    confederation : str
        FIFA confederation code (e.g., "UEFA", "CONMEBOL").
    positions : list[PositionRating]
        Six position-group ratings for the squad.
    composite_score : float
        Pre-computed composite strength score in [0, 1]. Set after scoring.
    """
    name: str
    fifa_ranking: int
    squad_value_eur_m: float
    gdp_per_capita: float
    population: int
    confederation: str
    positions: list[PositionRating] = field(default_factory=list)
    composite_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Team name must not be empty.")
        if self.fifa_ranking < 1:
            raise ValueError(f"fifa_ranking must be >= 1, got {self.fifa_ranking}")
        if self.squad_value_eur_m < 0:
            raise ValueError("squad_value_eur_m must be non-negative.")
        if self.gdp_per_capita < 0:
            raise ValueError("gdp_per_capita must be non-negative.")
        if self.population < 0:
            raise ValueError("population must be non-negative.")
        if not (0.0 <= self.composite_score <= 1.0):
            if self.composite_score != 0.0:  # 0.0 is the uninitialised default
                raise ValueError(f"composite_score must be in [0, 1], got {self.composite_score}")


@dataclass
class MatchOutcome:
    """
    Result of a single simulated match.

    Parameters
    ----------
    team_a, team_b : str
        Competing teams.
    goals_a, goals_b : int
        Goals scored in 90 minutes (or after extra time for knockouts).
    went_to_extra_time : bool
    went_to_penalties : bool
    winner : str
        Team name of the winner (empty string for group-stage draws).
    referee : str, optional
        Referee assigned to this match.
    penalty_count : int
        Number of penalties awarded during the match.
    yellow_cards : int
        Total yellow cards shown.
    red_cards : int
        Total red cards shown.
    """
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    went_to_extra_time: bool = False
    went_to_penalties: bool = False
    winner: str = ""
    referee: str = ""
    penalty_count: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    round_name: RoundName = RoundName.GROUP_STAGE

    def __post_init__(self) -> None:
        if self.goals_a < 0 or self.goals_b < 0:
            raise ValueError("Goal counts must be non-negative.")
        if self.winner and self.winner not in (self.team_a, self.team_b, ""):
            raise ValueError(f"winner must be team_a or team_b, got '{self.winner}'")


@dataclass
class SimulationConfig:
    """
    Configuration envelope for a Monte Carlo tournament simulation run.

    Parameters
    ----------
    n_runs : int
        Number of full tournament simulations (default: 50,000).
    random_seed : int
        Master seed for reproducibility.  All child seeds are derived
        deterministically from this value.
    use_referee_bias : bool
        Whether to apply RefereeBiasAnalyzer adjustments.
    use_form_weighting : bool
        Whether to incorporate FormAnalyzer momentum scores.
    use_venue_conditions : bool
        Whether to apply VenueConditionsModel altitude/temperature factors.
    use_upset_model : bool
        Whether to use the UpsetDetector logistic-regression overlay.
    float_precision : str
        "float32" or "float64". float32 halves memory at cost of ~1e-7 precision.
    n_workers : int
        ProcessPoolExecutor worker count. 0 = auto (CPU count).
    run_id : str
        Unique identifier for this run (used for cache keys).
    """
    n_runs: int = 50_000
    random_seed: int = 42
    use_referee_bias: bool = True
    use_form_weighting: bool = True
    use_venue_conditions: bool = True
    use_upset_model: bool = True
    float_precision: str = "float32"
    n_workers: int = 0
    run_id: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S"))

    def __post_init__(self) -> None:
        if self.n_runs < 1000:
            raise ValueError(f"n_runs must be >= 1000 for statistical validity, got {self.n_runs}")
        if self.float_precision not in ("float32", "float64"):
            raise ValueError("float_precision must be 'float32' or 'float64'.")
        if self.n_workers < 0:
            raise ValueError("n_workers must be >= 0.")


@dataclass
class TournamentOutcome:
    """
    Aggregated probability distribution from N tournament simulation runs.

    Parameters
    ----------
    team : str
        Team name.
    champion_prob : float
        Probability of winning the tournament.
    finalist_prob : float
        Probability of reaching the final.
    semi_finalist_prob : float
        Probability of reaching the semi-finals.
    quarter_finalist_prob : float
        Probability of reaching the quarter-finals.
    round_of_16_prob : float
        Probability of reaching (at least) the Round of 16.
    not_reaching_r16_prob : float
        Probability of NOT reaching the Round of 16. In a pure 32-team
        format this equals the group-stage exit probability; in the 48-team
        2026 format it also includes Round-of-32 exits. Equals
        1 - round_of_16_prob.
    expected_goals_for : float
        Average goals scored per tournament run.
    expected_goals_against : float
        Average goals conceded per tournament run.
    composite_score : float
        The team's composite strength score used as simulation input.
    """
    team: str
    champion_prob: float
    finalist_prob: float
    semi_finalist_prob: float
    quarter_finalist_prob: float
    round_of_16_prob: float
    not_reaching_r16_prob: float
    expected_goals_for: float = 0.0
    expected_goals_against: float = 0.0
    composite_score: float = 0.0

    def __post_init__(self) -> None:
        probs = [
            self.champion_prob, self.finalist_prob, self.semi_finalist_prob,
            self.quarter_finalist_prob, self.round_of_16_prob, self.not_reaching_r16_prob,
        ]
        for p in probs:
            if not (-1e-6 <= p <= 1.0 + 1e-6):
                raise ValueError(f"All probabilities must be in [0, 1], got {p}")

    @property
    def validation_sum(self) -> float:
        """Cumulative probabilities should roughly sum to 1 across rounds."""
        return (
            self.champion_prob + self.finalist_prob + self.semi_finalist_prob
            + self.quarter_finalist_prob + self.round_of_16_prob + self.not_reaching_r16_prob
        )


@dataclass
class TournamentEvent:
    """
    Immutable event fired by the bracket state machine.

    Events are queued chronologically and consumed by state-update handlers.
    The event-driven design ensures that any side-effect (updating standings,
    triggering referee assignment, logging) is decoupled from match logic.

    Parameters
    ----------
    event_type : TournamentEventType
    timestamp : datetime
        Wall-clock time of event creation (simulation time, not real time).
    round_name : RoundName
    payload : dict
        Event-specific data. Keys vary by event_type:
          MATCH_COMPLETED  → {match: MatchOutcome}
          TEAM_ADVANCED    → {team: str, to_round: RoundName}
          TEAM_ELIMINATED  → {team: str, in_round: RoundName}
          ROUND_COMPLETED  → {round: RoundName, standings: dict}
          TOURNAMENT_ENDED → {winner: str, runner_up: str}
    """
    event_type: TournamentEventType
    round_name: RoundName
    payload: dict
    timestamp: datetime = field(default_factory=datetime.utcnow)
    run_id: str = ""


@dataclass
class VenueProfile:
    """
    Physical characteristics of a 2026 FIFA World Cup venue.

    Parameters
    ----------
    name : str
        Stadium name.
    city : str
        Host city.
    country : str
        Host country (USA, Canada, or Mexico).
    altitude_m : int
        Altitude above sea level in metres.
    avg_game_time_temp_celsius : float
        Historical average temperature at typical kick-off times (local summer).
    capacity : int
        Seating capacity.
    timezone : str
        IANA timezone string.
    """
    name: str
    city: str
    country: str
    altitude_m: int
    avg_game_time_temp_celsius: float
    capacity: int
    timezone: str

    def __post_init__(self) -> None:
        if self.altitude_m < 0:
            raise ValueError("altitude_m must be non-negative.")
        if not (-20.0 <= self.avg_game_time_temp_celsius <= 50.0):
            raise ValueError("avg_game_time_temp_celsius out of plausible range.")


@dataclass
class RefereeAssignment:
    """Links a referee to a specific match."""
    referee_name: str
    match_key: str        # e.g. "Brazil vs France"
    round_name: RoundName
    bias_risk: BiasRisk
    expected_penalties: float
    adjusted_prob_a: float
    adjusted_prob_b: float


@dataclass
class BacktestResult:
    """
    Output of the 2022 World Cup backtesting engine.

    Parameters
    ----------
    brier_score : float
        Mean Brier score across all binary outcome predictions (lower = better).
        Perfect = 0.0, uninformative = 0.25.
    log_loss : float
        Mean log-loss (cross-entropy) across predictions.
    top3_accuracy : float
        Fraction of actual top-3 finishers that the model ranked in top 3.
    calibration_data : list[dict]
        Bin-by-bin calibration curve: predicted_prob_bin vs actual_frequency.
    narrative : str
        Human-readable model performance summary.
    predicted_champion_prob : float
        The model's pre-tournament win probability for the actual 2022 champion.
    actual_champion : str
        Actual 2022 World Cup winner (Argentina).
    model_champion_rank : int
        Where the model ranked the actual champion (1 = model predicted them to win).
    upset_recall : float
        Fraction of actual upsets flagged by the UpsetDetector (sensitivity).
    """
    brier_score: float
    log_loss: float
    top3_accuracy: float
    calibration_data: list[dict]
    narrative: str
    predicted_champion_prob: float
    actual_champion: str
    model_champion_rank: int
    upset_recall: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.brier_score <= 1.0):
            raise ValueError(f"brier_score out of range: {self.brier_score}")
        if self.log_loss < 0:
            raise ValueError(f"log_loss must be non-negative: {self.log_loss}")


@dataclass
class FormRecord:
    """A single match record for recent-form tracking."""
    date: str                  # ISO 8601
    opponent: str
    goals_for: int
    goals_against: int
    result: MatchResult
    venue_type: str            # "home" | "away" | "neutral"
    opponent_strength: float   # [0, 1] composite score of opponent


@dataclass
class HeadToHeadRecord:
    """Aggregated H2H stats between two teams."""
    team_a: str
    team_b: str
    meetings: int
    team_a_wins: int
    team_b_wins: int
    draws: int
    team_a_goals: int
    team_b_goals: int
    last_meeting_date: str
    last_meeting_result: str


@dataclass
class UpsetAlert:
    """A flagged match where upset probability exceeds threshold."""
    underdog: str
    favorite: str
    upset_probability: float
    strength_diff: float
    giant_killer_index: float
    historical_precedents: list[str]
    risk_level: str   # "moderate" | "high" | "extreme"
