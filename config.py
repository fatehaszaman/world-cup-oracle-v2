"""
config.py — World Cup Oracle global configuration.

All model weights, scaling constants, and tunable hyper-parameters live here
so that experiments and ablation studies only touch a single file.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Composite strength score dimension weights (must sum to 1.0)
# ---------------------------------------------------------------------------
# v2 weights — updated after 2022 WC backtest (v1 BPS: 40/64, FAIL)
# Changes vs v1:
#   squad_value      0.30 → 0.26  (-0.04)  over-favoured high-value squads (France > Argentina)
#   positional_power 0.25 → 0.30  (+0.05)  tactical organisation matters more than raw value
#   country_resources 0.15 → 0.13 (-0.02)  GDP/population penalised diaspora-heavy squads (Morocco)
#   historical       0.20 → 0.22  (+0.02)  tournament pedigree slightly underweighted
#   commercial       0.10 → 0.09  (-0.01)  brand value over-inflated Brazil/Germany QF chances
DIMENSION_WEIGHTS: dict[str, float] = {
    "squad_value":        0.26,
    "positional_power":   0.30,
    "country_resources":  0.13,
    "historical":         0.22,
    "commercial":         0.09,
}

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9, \
    "Dimension weights must sum to 1.0"

# ---------------------------------------------------------------------------
# Positional importance weights within positional_power score
# ---------------------------------------------------------------------------
POSITION_WEIGHTS: dict[str, float] = {
    "GK": 0.15,
    "CB": 0.20,
    "FB": 0.10,
    "CM": 0.25,
    "AM": 0.15,
    "FW": 0.15,
}

assert abs(sum(POSITION_WEIGHTS.values()) - 1.0) < 1e-9, \
    "Position weights must sum to 1.0"

# ---------------------------------------------------------------------------
# Historical performance scoring (last 5 World Cups)
# ---------------------------------------------------------------------------
HISTORICAL_POINTS: dict[str, float] = {
    "winner":         3.0,
    "runner_up":      2.0,
    "semi_finalist":  1.0,
    "quarter_finalist": 0.5,
    "round_of_16":    0.1,
    "group_stage":    0.0,
}

# Max theoretical points (5 wins per tournament × 5 tournaments)
HISTORICAL_MAX_POINTS: float = 5 * HISTORICAL_POINTS["winner"]

# ---------------------------------------------------------------------------
# Country resources formula weights
# ---------------------------------------------------------------------------
RESOURCE_WEIGHTS: dict[str, float] = {
    "gdp_per_capita":    0.35,
    "population":        0.20,
    "fifa_budget":       0.30,
    "primary_sport":     0.15,   # 1.0 if football is #1 sport, else penalty
}

# GDP per capita normalization ceiling (USD)
GDP_NORMALIZATION_CEILING: float = 80_000.0

# Population log-scale bounds (log base-10)
POPULATION_LOG_MIN: float = 6.0   # 1 million
POPULATION_LOG_MAX: float = 9.1   # ~1.25 billion

# ---------------------------------------------------------------------------
# Monte Carlo simulation defaults
# ---------------------------------------------------------------------------
MC_DEFAULT_RUNS: int = 50_000
MC_MATCH_SIMULATIONS: int = 10_000
MC_RANDOM_SEED: int = 42

# Poisson goal-rate base values
POISSON_BASE_LAMBDA: float = 1.35   # average goals per team per match
POISSON_STRENGTH_SCALE: float = 0.6  # how much strength diff shifts lambda

# Knockout-match draw handling: after 90 min draw → extra time coin flip weight
EXTRA_TIME_STRONGER_TEAM_BIAS: float = 0.55

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
RAPIDAPI_HOST: str = "api-football-v1.p.rapidapi.com"
API_FOOTBALL_BASE_URL: str = f"https://{RAPIDAPI_HOST}"
API_FOOTBALL_SEASON: int = 2024

WORLD_BANK_BASE_URL: str = "https://api.worldbank.org/v2"
WORLD_BANK_FORMAT: str = "json"

# Local cache TTL in seconds (24 hours)
CACHE_TTL_SECONDS: int = 86_400

# HTTP retry configuration
HTTP_MAX_RETRIES: int = 4
HTTP_BACKOFF_BASE: float = 1.5   # seconds

# ---------------------------------------------------------------------------
# Squad market value normalization ceiling (EUR millions)
# ---------------------------------------------------------------------------
SQUAD_VALUE_CEILING: float = 1_300.0

# ---------------------------------------------------------------------------
# Commercial signal normalization ceilings
# ---------------------------------------------------------------------------
SHIRT_DEAL_CEILING_EUR_M: float = 65.0
KIT_DEAL_CEILING_EUR_M: float = 100.0
FED_REVENUE_CEILING_EUR_M: float = 250.0
SOCIAL_FOLLOWERS_CEILING_M: float = 150.0
FANBASE_INDEX_CEILING: float = 10.0

# ---------------------------------------------------------------------------
# Psychological readiness model weights
# ---------------------------------------------------------------------------
# Weighting: physical (1.5) outweighs psychological (1.0) on a 2.5 total scale.
# Rationale: elite athletes can perform under emotional stress (Brett Favre,
# Isaiah Thomas), but physical conditioning is the harder constraint.
# Psychological state modifies performance at the margin — meaningful but not
# dominant. Sources:
#   - Liverpool University (2018 WC study): negative emotions reduce passing
#     accuracy for 3-9 minutes post-trigger.
#   - TSE (Turner & Slater, 2020): anger/happiness correlate with WC performance
#     on both individual and collective levels.
PSYCH_WEIGHT:    float = 1.0
PHYSICAL_WEIGHT: float = 1.5
READINESS_TOTAL_WEIGHT: float = PSYCH_WEIGHT + PHYSICAL_WEIGHT  # 2.5

# Monte Carlo psychological multiplier formula:
#   psych_multiplier = PSYCH_MC_BASE + PSYCH_MC_SCALE * (readiness / 100)
#   adjusted_score   = base_composite * psych_multiplier
# Range: [PSYCH_MC_BASE, PSYCH_MC_BASE + PSYCH_MC_SCALE] = [0.70, 1.00]
PSYCH_MC_BASE:  float = 0.70
PSYCH_MC_SCALE: float = 0.30

# Psychological score modifier caps
PSYCH_BASELINE:            float = 100.0
PSYCH_MIN_SCORE:           float = 20.0   # floor — even worst-case player still competes
PSYCH_MAX_SCORE:           float = 120.0  # ceiling — uncapped at 100 before normalising

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
TABLE_WIDTH: int = 88
PROBABILITY_DECIMAL_PLACES: int = 1

# ---------------------------------------------------------------------------
# Position-specific psychological sensitivity multipliers
# ---------------------------------------------------------------------------
# How much does psychological state MATTER for each position?
# A CM in emotional distress affects passing, pressing, decision-making — the
# engine of the team (weight 0.25). A GK in distress still has long spells of
# inaction before critical moments; psychological impact is real but narrower.
# A FB's mental state affects overlapping runs and defensive positioning but
# is the least team-critical position (weight 0.10).
#
# These multipliers scale HOW MUCH the emotional score shifts the readiness
# composite for a given position, BEFORE the team-level position weighting.
#
# Formula:
#   position_adjusted_emotional = 100 + (emotional - 100) * PSYCH_SENSITIVITY[pos]
#
# Examples:
#   CM at emotional=70 → 100 + (70-100)*1.30 = 100 - 39 = 61  (hurts more)
#   GK at emotional=70 → 100 + (70-100)*0.80 = 100 - 24 = 76  (hurts less)
#   CM at emotional=115 (revenge mode) → 100 + 15*1.30 = 119.5 (boosts more)
#   GK at emotional=115 → 100 + 15*0.80 = 112.0              (boosts less)
PSYCH_SENSITIVITY: dict[str, float] = {
    "GK": 0.80,   # Critical but episodic — long quiet spells, then one save
    "CB": 1.00,   # Baseline — concentration and communication always on
    "FB": 0.85,   # Less tactically central, overlaps can self-correct
    "CM": 1.30,   # Engine of team — every pass, press, transition affected
    "AM": 1.20,   # Creative hub — creativity/decision-making highly mood-sensitive
    "FW": 1.10,   # Finishing under pressure is mood-sensitive; isolated spells
}
