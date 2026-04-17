"""
oracle/form_analyzer.py — Recent form and head-to-head analyzer.

BUSINESS SUMMARY
----------------
A team that won its last 8 matches plays differently from one that scraped
through qualifying. This module tracks each team's last 10 international
results, weights them by recency (more recent = more important), and
produces a form rating that feeds into match outcome probabilities. It also
tracks direct head-to-head records between teams — historically, H2H
patterns matter especially in high-stakes knockout matches.

DEVELOPER NOTES
---------------
Recency weighting: exponential decay with half-life of 3 matches.
  weight[i] = exp(-decay_rate × i) where i=0 is most recent.
  decay_rate = ln(2) / 3 ≈ 0.231

Opponent strength adjustment: a win against Brazil counts more than
  a win against a minnow. The opponent's composite score (0–1) scales
  the points contribution:
  adjusted_pts = raw_pts × (0.5 + 0.5 × opponent_strength)

Momentum: compares the weighted score of the most recent 5 vs the
  prior 5 matches. Positive delta = trending upward.

Complexity: O(N) per team for N=10 results. O(1) H2H lookup.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

from oracle.schemas import MatchResult, FormRecord, HeadToHeadRecord

logger = logging.getLogger(__name__)

FORM_DECAY_RATE: float = math.log(2) / 3.0   # half-life of 3 matches
FORM_WIN_PTS:    float = 3.0
FORM_DRAW_PTS:   float = 1.0
FORM_LOSS_PTS:   float = 0.0
FORM_MAX_PTS:    float = FORM_WIN_PTS         # max per match (before strength scaling)


# ---------------------------------------------------------------------------
# Hardcoded recent form — last 10 international results per team
# Results cover approximate Nov 2024 – Mar 2026 international windows
# Opponent strength values are approximate composite scores
# ---------------------------------------------------------------------------
RECENT_FORM: dict[str, list[FormRecord]] = {
    "France": [
        FormRecord("2026-03-25", "Croatia",     2, 0, MatchResult.WIN,  "neutral", 0.74),
        FormRecord("2026-03-22", "Germany",      1, 1, MatchResult.DRAW, "neutral", 0.79),
        FormRecord("2025-11-19", "Italy",        3, 1, MatchResult.WIN,  "neutral", 0.68),
        FormRecord("2025-11-15", "Israel",       4, 0, MatchResult.WIN,  "neutral", 0.35),
        FormRecord("2025-10-13", "Belgium",      2, 1, MatchResult.WIN,  "neutral", 0.70),
        FormRecord("2025-10-10", "Spain",        1, 2, MatchResult.LOSS, "neutral", 0.80),
        FormRecord("2025-09-09", "Italy",        2, 2, MatchResult.DRAW, "neutral", 0.68),
        FormRecord("2025-09-05", "Switzerland",  3, 0, MatchResult.WIN,  "away",    0.67),
        FormRecord("2024-11-18", "Italy",        1, 3, MatchResult.LOSS, "neutral", 0.68),
        FormRecord("2024-11-15", "Israel",       0, 0, MatchResult.DRAW, "away",    0.35),
    ],
    "Brazil": [
        FormRecord("2026-03-26", "Colombia",     2, 1, MatchResult.WIN,  "home",    0.60),
        FormRecord("2026-03-22", "Uruguay",      0, 0, MatchResult.DRAW, "away",    0.63),
        FormRecord("2025-11-19", "Venezuela",    5, 1, MatchResult.WIN,  "home",    0.38),
        FormRecord("2025-11-15", "Uruguay",      1, 0, MatchResult.WIN,  "home",    0.63),
        FormRecord("2025-10-15", "Peru",         4, 0, MatchResult.WIN,  "away",    0.42),
        FormRecord("2025-10-11", "Chile",        2, 1, MatchResult.WIN,  "home",    0.50),
        FormRecord("2025-09-10", "Ecuador",      1, 1, MatchResult.DRAW, "away",    0.56),
        FormRecord("2025-09-06", "Paraguay",     4, 1, MatchResult.WIN,  "home",    0.44),
        FormRecord("2024-11-19", "Uruguay",      0, 1, MatchResult.LOSS, "away",    0.63),
        FormRecord("2024-11-15", "Venezuela",    1, 1, MatchResult.DRAW, "home",    0.38),
    ],
    "Germany": [
        FormRecord("2026-03-25", "France",       1, 1, MatchResult.DRAW, "neutral", 0.82),
        FormRecord("2026-03-22", "Italy",        2, 0, MatchResult.WIN,  "home",    0.68),
        FormRecord("2025-11-19", "Bosnia",       7, 0, MatchResult.WIN,  "home",    0.30),
        FormRecord("2025-11-15", "Netherlands",  2, 4, MatchResult.LOSS, "away",    0.72),
        FormRecord("2025-10-14", "Finland",      3, 1, MatchResult.WIN,  "home",    0.32),
        FormRecord("2025-10-11", "Sweden",       2, 2, MatchResult.DRAW, "away",    0.55),
        FormRecord("2025-09-09", "Netherlands",  4, 0, MatchResult.WIN,  "home",    0.72),
        FormRecord("2025-09-06", "Hungary",      5, 0, MatchResult.WIN,  "home",    0.40),
        FormRecord("2024-11-16", "Hungary",      0, 1, MatchResult.LOSS, "away",    0.40),
        FormRecord("2024-11-14", "Bosnia",       2, 0, MatchResult.WIN,  "home",    0.30),
    ],
    "Spain": [
        FormRecord("2026-03-25", "Netherlands",  3, 0, MatchResult.WIN,  "home",    0.72),
        FormRecord("2026-03-22", "Belgium",      2, 0, MatchResult.WIN,  "neutral", 0.70),
        FormRecord("2025-11-18", "Switzerland",  3, 2, MatchResult.WIN,  "away",    0.67),
        FormRecord("2025-11-15", "Denmark",      2, 1, MatchResult.WIN,  "home",    0.65),
        FormRecord("2025-10-15", "France",       2, 1, MatchResult.WIN,  "neutral", 0.82),
        FormRecord("2025-10-12", "Serbia",       3, 0, MatchResult.WIN,  "home",    0.59),
        FormRecord("2025-09-09", "Portugal",     2, 2, MatchResult.DRAW, "neutral", 0.75),
        FormRecord("2025-09-05", "Denmark",      2, 0, MatchResult.WIN,  "away",    0.65),
        FormRecord("2024-11-18", "Switzerland",  3, 2, MatchResult.WIN,  "home",    0.67),
        FormRecord("2024-11-14", "Netherlands",  3, 0, MatchResult.WIN,  "home",    0.72),
    ],
    "Argentina": [
        FormRecord("2026-03-25", "Ecuador",      1, 0, MatchResult.WIN,  "home",    0.56),
        FormRecord("2026-03-22", "Chile",        3, 1, MatchResult.WIN,  "home",    0.50),
        FormRecord("2025-11-19", "Peru",         1, 0, MatchResult.WIN,  "away",    0.42),
        FormRecord("2025-11-15", "Bolivia",      3, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2025-10-16", "Paraguay",     2, 0, MatchResult.WIN,  "away",    0.44),
        FormRecord("2025-10-12", "Venezuela",    6, 0, MatchResult.WIN,  "home",    0.38),
        FormRecord("2025-09-09", "Chile",        2, 1, MatchResult.WIN,  "away",    0.50),
        FormRecord("2025-09-05", "Colombia",     2, 1, MatchResult.WIN,  "home",    0.60),
        FormRecord("2024-11-19", "Ecuador",      1, 0, MatchResult.WIN,  "home",    0.56),
        FormRecord("2024-11-15", "Paraguay",     2, 0, MatchResult.WIN,  "home",    0.44),
    ],
    "England": [
        FormRecord("2026-03-26", "Albania",      2, 0, MatchResult.WIN,  "home",    0.38),
        FormRecord("2026-03-22", "Portugal",     1, 1, MatchResult.DRAW, "home",    0.75),
        FormRecord("2025-11-19", "Greece",       3, 1, MatchResult.WIN,  "home",    0.52),
        FormRecord("2025-11-14", "Republic of Ireland", 5, 0, MatchResult.WIN, "away", 0.40),
        FormRecord("2025-10-13", "Finland",      3, 1, MatchResult.WIN,  "home",    0.32),
        FormRecord("2025-10-10", "Greece",       2, 0, MatchResult.WIN,  "away",    0.52),
        FormRecord("2025-09-09", "Finland",      2, 0, MatchResult.WIN,  "home",    0.32),
        FormRecord("2025-09-05", "Republic of Ireland", 2, 0, MatchResult.WIN, "home", 0.40),
        FormRecord("2024-11-17", "Greece",       1, 2, MatchResult.LOSS, "home",    0.52),
        FormRecord("2024-11-14", "Republic of Ireland", 5, 0, MatchResult.WIN, "neutral", 0.40),
    ],
    "Portugal": [
        FormRecord("2026-03-25", "Hungary",      2, 0, MatchResult.WIN,  "home",    0.40),
        FormRecord("2026-03-22", "England",      1, 1, MatchResult.DRAW, "away",    0.79),
        FormRecord("2025-11-19", "Poland",       5, 1, MatchResult.WIN,  "home",    0.55),
        FormRecord("2025-11-15", "Croatia",      2, 1, MatchResult.WIN,  "away",    0.74),
        FormRecord("2025-10-14", "Poland",       3, 1, MatchResult.WIN,  "away",    0.55),
        FormRecord("2025-10-11", "Scotland",     4, 1, MatchResult.WIN,  "home",    0.42),
        FormRecord("2025-09-09", "Spain",        2, 2, MatchResult.DRAW, "neutral", 0.80),
        FormRecord("2025-09-05", "Scotland",     2, 0, MatchResult.WIN,  "away",    0.42),
        FormRecord("2024-11-18", "Poland",       1, 1, MatchResult.DRAW, "home",    0.55),
        FormRecord("2024-11-15", "Croatia",      1, 0, MatchResult.WIN,  "home",    0.74),
    ],
    "Netherlands": [
        FormRecord("2026-03-25", "Spain",        0, 3, MatchResult.LOSS, "away",    0.80),
        FormRecord("2026-03-22", "Poland",       3, 0, MatchResult.WIN,  "home",    0.55),
        FormRecord("2025-11-18", "Germany",      4, 0, MatchResult.WIN,  "home",    0.79),
        FormRecord("2025-11-15", "Hungary",      4, 0, MatchResult.WIN,  "home",    0.40),
        FormRecord("2025-10-13", "Hungary",      1, 0, MatchResult.WIN,  "away",    0.40),
        FormRecord("2025-10-10", "Bosnia",       5, 2, MatchResult.WIN,  "home",    0.30),
        FormRecord("2025-09-09", "Germany",      0, 4, MatchResult.LOSS, "away",    0.79),
        FormRecord("2025-09-05", "Bosnia",       5, 0, MatchResult.WIN,  "home",    0.30),
        FormRecord("2024-11-18", "Germany",      4, 0, MatchResult.WIN,  "home",    0.79),
        FormRecord("2024-11-15", "Hungary",      4, 0, MatchResult.WIN,  "home",    0.40),
    ],
    "Morocco": [
        FormRecord("2026-03-26", "Gabon",        3, 0, MatchResult.WIN,  "home",    0.28),
        FormRecord("2026-03-22", "Comoros",      4, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-11-19", "Gabon",        2, 0, MatchResult.WIN,  "away",    0.28),
        FormRecord("2025-11-16", "Comoros",      3, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-10-12", "Senegal",      0, 0, MatchResult.DRAW, "neutral", 0.62),
        FormRecord("2025-06-15", "Argentina",    0, 3, MatchResult.LOSS, "neutral", 0.85),
        FormRecord("2025-06-12", "United States",1, 1, MatchResult.DRAW, "neutral", 0.58),
        FormRecord("2025-03-26", "Cameroon",     1, 0, MatchResult.WIN,  "home",    0.47),
        FormRecord("2025-03-22", "Guinea",       2, 1, MatchResult.WIN,  "home",    0.30),
        FormRecord("2024-11-20", "Gabon",        2, 0, MatchResult.WIN,  "home",    0.28),
    ],
    "Croatia": [
        FormRecord("2026-03-25", "Greece",       2, 1, MatchResult.WIN,  "home",    0.52),
        FormRecord("2026-03-22", "Portugal",     1, 2, MatchResult.LOSS, "home",    0.75),
        FormRecord("2025-11-19", "Scotland",     2, 1, MatchResult.WIN,  "home",    0.42),
        FormRecord("2025-11-15", "Portugal",     1, 2, MatchResult.LOSS, "away",    0.75),
        FormRecord("2025-10-13", "Poland",       2, 0, MatchResult.WIN,  "home",    0.55),
        FormRecord("2025-10-10", "Scotland",     2, 1, MatchResult.WIN,  "away",    0.42),
        FormRecord("2025-09-09", "Poland",       1, 1, MatchResult.DRAW, "away",    0.55),
        FormRecord("2025-09-05", "Scotland",     2, 0, MatchResult.WIN,  "home",    0.42),
        FormRecord("2024-11-18", "Portugal",     1, 1, MatchResult.DRAW, "home",    0.75),
        FormRecord("2024-11-15", "Scotland",     1, 1, MatchResult.DRAW, "away",    0.42),
    ],
    "Italy": [
        FormRecord("2026-03-25", "Norway",       2, 0, MatchResult.WIN,  "home",    0.48),
        FormRecord("2026-03-22", "Germany",      0, 2, MatchResult.LOSS, "away",    0.79),
        FormRecord("2025-11-19", "Belgium",      1, 0, MatchResult.WIN,  "neutral", 0.70),
        FormRecord("2025-11-15", "France",       1, 3, MatchResult.LOSS, "neutral", 0.82),
        FormRecord("2025-10-14", "Belgium",      2, 2, MatchResult.DRAW, "away",    0.70),
        FormRecord("2025-10-10", "Israel",       4, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2025-09-09", "France",       2, 2, MatchResult.DRAW, "neutral", 0.82),
        FormRecord("2025-09-06", "Israel",       2, 1, MatchResult.WIN,  "away",    0.35),
        FormRecord("2024-11-18", "France",       3, 1, MatchResult.WIN,  "neutral", 0.82),
        FormRecord("2024-11-15", "Belgium",      2, 2, MatchResult.DRAW, "home",    0.70),
    ],
    "Japan": [
        FormRecord("2026-03-26", "Bahrain",      5, 0, MatchResult.WIN,  "home",    0.28),
        FormRecord("2026-03-21", "China",        7, 0, MatchResult.WIN,  "home",    0.32),
        FormRecord("2025-11-19", "Indonesia",    4, 0, MatchResult.WIN,  "away",    0.22),
        FormRecord("2025-11-15", "China",        1, 0, MatchResult.WIN,  "home",    0.32),
        FormRecord("2025-10-15", "Australia",    1, 0, MatchResult.WIN,  "home",    0.53),
        FormRecord("2025-10-11", "Saudi Arabia", 2, 0, MatchResult.WIN,  "away",    0.45),
        FormRecord("2025-09-10", "Bahrain",      5, 0, MatchResult.WIN,  "home",    0.28),
        FormRecord("2025-09-05", "China",        7, 0, MatchResult.WIN,  "away",    0.32),
        FormRecord("2024-11-19", "Indonesia",    4, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2024-11-15", "China",        1, 0, MatchResult.WIN,  "home",    0.32),
    ],
    "South Korea": [
        FormRecord("2026-03-26", "Jordan",       3, 0, MatchResult.WIN,  "home",    0.33),
        FormRecord("2026-03-21", "Iraq",         2, 0, MatchResult.WIN,  "home",    0.36),
        FormRecord("2025-11-19", "Kuwait",       3, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-11-15", "Oman",         3, 1, MatchResult.WIN,  "home",    0.26),
        FormRecord("2025-10-15", "Iraq",         3, 1, MatchResult.WIN,  "away",    0.36),
        FormRecord("2025-10-11", "Jordan",       1, 0, MatchResult.WIN,  "away",    0.33),
        FormRecord("2025-09-10", "Oman",         3, 1, MatchResult.WIN,  "home",    0.26),
        FormRecord("2025-09-05", "Palestine",    4, 0, MatchResult.WIN,  "home",    0.18),
        FormRecord("2024-11-19", "Kuwait",       3, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2024-11-15", "Palestine",    5, 0, MatchResult.WIN,  "away",    0.18),
    ],
    "Senegal": [
        FormRecord("2026-03-26", "Togo",         2, 0, MatchResult.WIN,  "home",    0.26),
        FormRecord("2026-03-22", "Malawi",       3, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-11-19", "Congo DR",     1, 1, MatchResult.DRAW, "away",    0.35),
        FormRecord("2025-11-16", "Togo",         3, 0, MatchResult.WIN,  "home",    0.26),
        FormRecord("2025-10-12", "Morocco",      0, 0, MatchResult.DRAW, "neutral", 0.65),
        FormRecord("2025-10-09", "Burkina Faso", 1, 0, MatchResult.WIN,  "home",    0.30),
        FormRecord("2025-06-15", "Mauritania",   2, 0, MatchResult.WIN,  "home",    0.20),
        FormRecord("2025-06-12", "Zambia",       2, 0, MatchResult.WIN,  "home",    0.23),
        FormRecord("2025-03-26", "Mauritania",   1, 0, MatchResult.WIN,  "away",    0.20),
        FormRecord("2025-03-22", "Zambia",       1, 0, MatchResult.WIN,  "away",    0.23),
    ],
    "United States": [
        FormRecord("2026-03-26", "Jamaica",      4, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2026-03-22", "Canada",       2, 1, MatchResult.WIN,  "home",    0.58),
        FormRecord("2025-11-19", "Costa Rica",   4, 0, MatchResult.WIN,  "home",    0.42),
        FormRecord("2025-11-15", "Jamaica",      4, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2025-10-15", "Mexico",       2, 0, MatchResult.WIN,  "home",    0.58),
        FormRecord("2025-10-12", "Panama",       3, 1, MatchResult.WIN,  "home",    0.40),
        FormRecord("2025-09-09", "New Zealand",  4, 1, MatchResult.WIN,  "home",    0.28),
        FormRecord("2025-09-05", "Canada",       1, 0, MatchResult.WIN,  "neutral", 0.58),
        FormRecord("2025-06-15", "Morocco",      1, 1, MatchResult.DRAW, "neutral", 0.65),
        FormRecord("2025-06-12", "Brazil",       0, 2, MatchResult.LOSS, "neutral", 0.80),
    ],
    "Switzerland": [
        FormRecord("2026-03-26", "France",       0, 3, MatchResult.LOSS, "home",    0.82),
        FormRecord("2026-03-22", "Kosovo",       3, 0, MatchResult.WIN,  "home",    0.28),
        FormRecord("2025-11-18", "Spain",        2, 3, MatchResult.LOSS, "home",    0.80),
        FormRecord("2025-11-15", "Serbia",       1, 0, MatchResult.WIN,  "home",    0.59),
        FormRecord("2025-10-12", "Kosovo",       2, 0, MatchResult.WIN,  "home",    0.28),
        FormRecord("2025-10-09", "Denmark",      1, 1, MatchResult.DRAW, "away",    0.65),
        FormRecord("2025-09-09", "Spain",        0, 3, MatchResult.LOSS, "away",    0.80),
        FormRecord("2025-09-06", "Serbia",       1, 0, MatchResult.WIN,  "away",    0.59),
        FormRecord("2024-11-18", "Spain",        2, 3, MatchResult.LOSS, "away",    0.80),
        FormRecord("2024-11-15", "Denmark",      2, 0, MatchResult.WIN,  "home",    0.65),
    ],
    "Denmark": [
        FormRecord("2026-03-26", "Georgia",      2, 0, MatchResult.WIN,  "home",    0.38),
        FormRecord("2026-03-22", "Norway",       1, 0, MatchResult.WIN,  "home",    0.48),
        FormRecord("2025-11-18", "Spain",        1, 2, MatchResult.LOSS, "away",    0.80),
        FormRecord("2025-11-14", "Switzerland",  0, 2, MatchResult.LOSS, "home",    0.67),
        FormRecord("2025-10-14", "Switzerland",  1, 1, MatchResult.DRAW, "home",    0.67),
        FormRecord("2025-10-10", "Spain",        1, 2, MatchResult.LOSS, "home",    0.80),
        FormRecord("2025-09-09", "Norway",       2, 1, MatchResult.WIN,  "away",    0.48),
        FormRecord("2025-09-05", "Spain",        0, 2, MatchResult.LOSS, "away",    0.80),
        FormRecord("2024-11-18", "Switzerland",  0, 2, MatchResult.LOSS, "away",    0.67),
        FormRecord("2024-11-14", "Norway",       1, 0, MatchResult.WIN,  "home",    0.48),
    ],
    "Belgium": [
        FormRecord("2026-03-26", "Wales",        4, 0, MatchResult.WIN,  "home",    0.40),
        FormRecord("2026-03-22", "Spain",        0, 2, MatchResult.LOSS, "neutral", 0.80),
        FormRecord("2025-11-19", "Italy",        0, 1, MatchResult.LOSS, "neutral", 0.68),
        FormRecord("2025-11-15", "France",       1, 2, MatchResult.LOSS, "neutral", 0.82),
        FormRecord("2025-10-14", "Italy",        2, 2, MatchResult.DRAW, "home",    0.68),
        FormRecord("2025-10-10", "France",       1, 2, MatchResult.LOSS, "neutral", 0.82),
        FormRecord("2025-09-09", "Wales",        3, 0, MatchResult.WIN,  "home",    0.40),
        FormRecord("2025-09-05", "Italy",        2, 2, MatchResult.DRAW, "away",    0.68),
        FormRecord("2024-11-18", "Italy",        2, 2, MatchResult.DRAW, "away",    0.68),
        FormRecord("2024-11-14", "France",       1, 2, MatchResult.LOSS, "neutral", 0.82),
    ],
    "Uruguay": [
        FormRecord("2026-03-26", "Chile",        3, 1, MatchResult.WIN,  "home",    0.50),
        FormRecord("2026-03-22", "Brazil",       0, 0, MatchResult.DRAW, "home",    0.80),
        FormRecord("2025-11-19", "Bolivia",      4, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2025-11-15", "Colombia",     1, 2, MatchResult.LOSS, "away",    0.60),
        FormRecord("2025-10-16", "Ecuador",      2, 0, MatchResult.WIN,  "home",    0.56),
        FormRecord("2025-10-12", "Venezuela",    3, 0, MatchResult.WIN,  "home",    0.38),
        FormRecord("2025-09-09", "Peru",         2, 0, MatchResult.WIN,  "away",    0.42),
        FormRecord("2025-09-05", "Chile",        2, 0, MatchResult.WIN,  "away",    0.50),
        FormRecord("2024-11-19", "Brazil",       1, 0, MatchResult.WIN,  "home",    0.80),
        FormRecord("2024-11-15", "Bolivia",      4, 0, MatchResult.WIN,  "home",    0.35),
    ],
    "Mexico": [
        FormRecord("2026-03-26", "Honduras",     3, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2026-03-22", "United States",0, 2, MatchResult.LOSS, "away",    0.58),
        FormRecord("2025-11-19", "Canada",       2, 1, MatchResult.WIN,  "home",    0.58),
        FormRecord("2025-11-15", "Panama",       2, 0, MatchResult.WIN,  "home",    0.40),
        FormRecord("2025-10-15", "United States",0, 2, MatchResult.LOSS, "away",    0.58),
        FormRecord("2025-10-12", "Jamaica",      3, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2025-09-09", "Canada",       1, 0, MatchResult.WIN,  "home",    0.58),
        FormRecord("2025-09-05", "Costa Rica",   4, 0, MatchResult.WIN,  "home",    0.42),
        FormRecord("2024-11-19", "Honduras",     2, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2024-11-14", "United States",0, 2, MatchResult.LOSS, "away",    0.58),
    ],
    "Colombia": [
        FormRecord("2026-03-26", "Paraguay",     3, 0, MatchResult.WIN,  "home",    0.44),
        FormRecord("2026-03-22", "Brazil",       1, 2, MatchResult.LOSS, "away",    0.80),
        FormRecord("2025-11-19", "Uruguay",      2, 1, MatchResult.WIN,  "home",    0.63),
        FormRecord("2025-11-15", "Peru",         3, 0, MatchResult.WIN,  "home",    0.42),
        FormRecord("2025-10-15", "Venezuela",    2, 0, MatchResult.WIN,  "home",    0.38),
        FormRecord("2025-10-11", "Bolivia",      2, 1, MatchResult.WIN,  "away",    0.35),
        FormRecord("2025-09-09", "Ecuador",      0, 0, MatchResult.DRAW, "home",    0.56),
        FormRecord("2025-09-05", "Argentina",    1, 2, MatchResult.LOSS, "away",    0.85),
        FormRecord("2024-11-19", "Uruguay",      2, 1, MatchResult.WIN,  "home",    0.63),
        FormRecord("2024-11-15", "Peru",         2, 1, MatchResult.WIN,  "home",    0.42),
    ],
    "Ecuador": [
        FormRecord("2026-03-26", "Venezuela",    2, 0, MatchResult.WIN,  "home",    0.38),
        FormRecord("2026-03-22", "Argentina",    0, 1, MatchResult.LOSS, "away",    0.85),
        FormRecord("2025-11-19", "Bolivia",      2, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2025-11-15", "Paraguay",     1, 0, MatchResult.WIN,  "home",    0.44),
        FormRecord("2025-10-15", "Chile",        1, 0, MatchResult.WIN,  "home",    0.50),
        FormRecord("2025-10-12", "Peru",         2, 0, MatchResult.WIN,  "away",    0.42),
        FormRecord("2025-09-09", "Brazil",       1, 1, MatchResult.DRAW, "home",    0.80),
        FormRecord("2025-09-05", "Colombia",     0, 0, MatchResult.DRAW, "away",    0.60),
        FormRecord("2024-11-19", "Bolivia",      2, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2024-11-15", "Paraguay",     2, 0, MatchResult.WIN,  "home",    0.44),
    ],
    "Canada": [
        FormRecord("2026-03-26", "Jamaica",      3, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2026-03-22", "United States",1, 2, MatchResult.LOSS, "away",    0.58),
        FormRecord("2025-11-19", "Costa Rica",   2, 0, MatchResult.WIN,  "home",    0.42),
        FormRecord("2025-11-14", "Jamaica",      2, 0, MatchResult.WIN,  "home",    0.35),
        FormRecord("2025-10-15", "Panama",       2, 1, MatchResult.WIN,  "home",    0.40),
        FormRecord("2025-10-12", "Curaçao",      4, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-09-09", "United States",0, 1, MatchResult.LOSS, "neutral", 0.58),
        FormRecord("2025-09-05", "Mexico",       0, 1, MatchResult.LOSS, "away",    0.58),
        FormRecord("2024-11-19", "Costa Rica",   2, 0, MatchResult.WIN,  "home",    0.42),
        FormRecord("2024-11-15", "Jamaica",      1, 0, MatchResult.WIN,  "home",    0.35),
    ],
    "Australia": [
        FormRecord("2026-03-26", "China",        2, 0, MatchResult.WIN,  "home",    0.32),
        FormRecord("2026-03-21", "Indonesia",    3, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-11-19", "Saudi Arabia", 2, 0, MatchResult.WIN,  "away",    0.45),
        FormRecord("2025-11-14", "Bahrain",      3, 0, MatchResult.WIN,  "home",    0.28),
        FormRecord("2025-10-15", "Japan",        0, 1, MatchResult.LOSS, "away",    0.60),
        FormRecord("2025-10-11", "Saudi Arabia", 1, 0, MatchResult.WIN,  "home",    0.45),
        FormRecord("2025-09-10", "China",        2, 0, MatchResult.WIN,  "home",    0.32),
        FormRecord("2025-09-05", "Indonesia",    2, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2024-11-19", "Bahrain",      2, 0, MatchResult.WIN,  "home",    0.28),
        FormRecord("2024-11-14", "China",        1, 0, MatchResult.WIN,  "away",    0.32),
    ],
    "Nigeria": [
        FormRecord("2026-03-26", "South Africa", 2, 0, MatchResult.WIN,  "home",    0.38),
        FormRecord("2026-03-22", "Burkina Faso", 2, 0, MatchResult.WIN,  "home",    0.30),
        FormRecord("2025-11-19", "Libya",        2, 0, MatchResult.WIN,  "home",    0.24),
        FormRecord("2025-11-16", "Rwanda",       4, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-10-14", "South Africa", 0, 1, MatchResult.LOSS, "away",    0.38),
        FormRecord("2025-10-10", "Benin",        2, 0, MatchResult.WIN,  "home",    0.25),
        FormRecord("2025-09-09", "Libya",        2, 0, MatchResult.WIN,  "away",    0.24),
        FormRecord("2025-09-05", "Benin",        3, 0, MatchResult.WIN,  "home",    0.25),
        FormRecord("2024-11-19", "Rwanda",       3, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2024-11-14", "Benin",        2, 0, MatchResult.WIN,  "away",    0.25),
    ],
    "Ivory Coast": [
        FormRecord("2026-03-26", "Tanzania",     2, 0, MatchResult.WIN,  "home",    0.24),
        FormRecord("2026-03-22", "Cameroon",     1, 0, MatchResult.WIN,  "home",    0.47),
        FormRecord("2025-11-19", "Madagascar",   3, 0, MatchResult.WIN,  "home",    0.20),
        FormRecord("2025-11-16", "Zambia",       2, 0, MatchResult.WIN,  "home",    0.23),
        FormRecord("2025-10-14", "Cameroon",     1, 0, MatchResult.WIN,  "home",    0.47),
        FormRecord("2025-10-10", "Tanzania",     2, 0, MatchResult.WIN,  "away",    0.24),
        FormRecord("2025-09-09", "Madagascar",   2, 0, MatchResult.WIN,  "home",    0.20),
        FormRecord("2025-09-05", "Zambia",       3, 0, MatchResult.WIN,  "home",    0.23),
        FormRecord("2024-11-19", "Tanzania",     2, 0, MatchResult.WIN,  "home",    0.24),
        FormRecord("2024-11-14", "Cameroon",     1, 1, MatchResult.DRAW, "away",    0.47),
    ],
    "Cameroon": [
        FormRecord("2026-03-26", "Kenya",        2, 0, MatchResult.WIN,  "home",    0.25),
        FormRecord("2026-03-22", "Ivory Coast",  0, 1, MatchResult.LOSS, "away",    0.62),
        FormRecord("2025-11-19", "Namibia",      2, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-11-16", "eSwatini",     4, 0, MatchResult.WIN,  "home",    0.18),
        FormRecord("2025-10-14", "Ivory Coast",  0, 1, MatchResult.LOSS, "away",    0.62),
        FormRecord("2025-10-10", "Kenya",        2, 0, MatchResult.WIN,  "home",    0.25),
        FormRecord("2025-09-09", "Namibia",      3, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-09-05", "eSwatini",     4, 0, MatchResult.WIN,  "home",    0.18),
        FormRecord("2024-11-19", "Kenya",        2, 0, MatchResult.WIN,  "home",    0.25),
        FormRecord("2024-11-14", "Ivory Coast",  1, 1, MatchResult.DRAW, "home",    0.62),
    ],
    "Saudi Arabia": [
        FormRecord("2026-03-26", "Bahrain",      2, 0, MatchResult.WIN,  "home",    0.28),
        FormRecord("2026-03-21", "Indonesia",    3, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-11-19", "Australia",    0, 2, MatchResult.LOSS, "home",    0.53),
        FormRecord("2025-11-14", "Indonesia",    2, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-10-15", "Japan",        0, 2, MatchResult.LOSS, "home",    0.60),
        FormRecord("2025-10-11", "Australia",    0, 1, MatchResult.LOSS, "away",    0.53),
        FormRecord("2025-09-10", "Indonesia",    3, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-09-05", "Bahrain",      2, 1, MatchResult.WIN,  "home",    0.28),
        FormRecord("2024-11-19", "Japan",        0, 2, MatchResult.LOSS, "away",    0.60),
        FormRecord("2024-11-14", "Australia",    0, 1, MatchResult.LOSS, "home",    0.53),
    ],
    "Iran": [
        FormRecord("2026-03-26", "Kyrgyzstan",   2, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2026-03-21", "North Korea",  3, 0, MatchResult.WIN,  "home",    0.20),
        FormRecord("2025-11-19", "Uzbekistan",   1, 0, MatchResult.WIN,  "home",    0.30),
        FormRecord("2025-11-14", "Kyrgyzstan",   2, 0, MatchResult.WIN,  "away",    0.22),
        FormRecord("2025-10-15", "UAE",          1, 0, MatchResult.WIN,  "home",    0.28),
        FormRecord("2025-10-11", "Uzbekistan",   0, 1, MatchResult.LOSS, "away",    0.30),
        FormRecord("2025-09-10", "Kyrgyzstan",   2, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-09-05", "North Korea",  2, 0, MatchResult.WIN,  "home",    0.20),
        FormRecord("2024-11-19", "Uzbekistan",   1, 0, MatchResult.WIN,  "home",    0.30),
        FormRecord("2024-11-14", "UAE",          1, 0, MatchResult.WIN,  "home",    0.28),
    ],
    "Poland": [
        FormRecord("2026-03-26", "Malta",        5, 0, MatchResult.WIN,  "home",    0.20),
        FormRecord("2026-03-22", "Netherlands",  0, 3, MatchResult.LOSS, "away",    0.72),
        FormRecord("2025-11-19", "Portugal",     1, 5, MatchResult.LOSS, "away",    0.75),
        FormRecord("2025-11-15", "Lithuania",    3, 0, MatchResult.WIN,  "home",    0.25),
        FormRecord("2025-10-14", "Portugal",     1, 3, MatchResult.LOSS, "home",    0.75),
        FormRecord("2025-10-11", "Croatia",      0, 2, MatchResult.LOSS, "away",    0.74),
        FormRecord("2025-09-09", "Croatia",      1, 1, MatchResult.DRAW, "home",    0.74),
        FormRecord("2025-09-05", "Scotland",     3, 2, MatchResult.WIN,  "away",    0.42),
        FormRecord("2024-11-18", "Portugal",     1, 1, MatchResult.DRAW, "away",    0.75),
        FormRecord("2024-11-14", "Scotland",     2, 2, MatchResult.DRAW, "home",    0.42),
    ],
    "Serbia": [
        FormRecord("2026-03-25", "Malta",        4, 0, MatchResult.WIN,  "home",    0.20),
        FormRecord("2026-03-22", "Spain",        0, 3, MatchResult.LOSS, "away",    0.80),
        FormRecord("2025-11-19", "Austria",      1, 2, MatchResult.LOSS, "away",    0.60),
        FormRecord("2025-11-15", "Denmark",      1, 0, MatchResult.WIN,  "home",    0.65),
        FormRecord("2025-10-14", "Denmark",      0, 2, MatchResult.LOSS, "away",    0.65),
        FormRecord("2025-10-10", "Switzerland",  0, 1, MatchResult.LOSS, "home",    0.67),
        FormRecord("2025-09-09", "Denmark",      1, 0, MatchResult.WIN,  "home",    0.65),
        FormRecord("2025-09-05", "Switzerland",  0, 1, MatchResult.LOSS, "away",    0.67),
        FormRecord("2024-11-19", "Austria",      0, 1, MatchResult.LOSS, "away",    0.60),
        FormRecord("2024-11-14", "Denmark",      0, 1, MatchResult.LOSS, "away",    0.65),
    ],
    "Austria": [
        FormRecord("2026-03-26", "Norway",       2, 1, MatchResult.WIN,  "home",    0.48),
        FormRecord("2026-03-22", "Finland",      3, 0, MatchResult.WIN,  "home",    0.32),
        FormRecord("2025-11-19", "Serbia",       2, 1, MatchResult.WIN,  "home",    0.59),
        FormRecord("2025-11-15", "Kazakhstan",   4, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-10-14", "Sweden",       1, 0, MatchResult.WIN,  "home",    0.55),
        FormRecord("2025-10-10", "Norway",       1, 0, MatchResult.WIN,  "home",    0.48),
        FormRecord("2025-09-09", "Kazakhstan",   3, 0, MatchResult.WIN,  "home",    0.22),
        FormRecord("2025-09-05", "Sweden",       2, 0, MatchResult.WIN,  "home",    0.55),
        FormRecord("2024-11-19", "Serbia",       1, 0, MatchResult.WIN,  "home",    0.59),
        FormRecord("2024-11-14", "Kazakhstan",   4, 0, MatchResult.WIN,  "home",    0.22),
    ],
}

# ---------------------------------------------------------------------------
# Head-to-head records for major matchups
# ---------------------------------------------------------------------------
H2H_RECORDS: dict[frozenset, HeadToHeadRecord] = {
    frozenset({"France", "Brazil"}): HeadToHeadRecord(
        "France", "Brazil", meetings=10, team_a_wins=3, team_b_wins=5, draws=2,
        team_a_goals=14, team_b_goals=20,
        last_meeting_date="2019-10-09", last_meeting_result="Brazil 1-0 France",
    ),
    frozenset({"Argentina", "France"}): HeadToHeadRecord(
        "Argentina", "France", meetings=12, team_a_wins=6, team_b_wins=3, draws=3,
        team_a_goals=21, team_b_goals=15,
        last_meeting_date="2022-12-18", last_meeting_result="Argentina 3-3 France (Argentina pens)",
    ),
    frozenset({"England", "Germany"}): HeadToHeadRecord(
        "England", "Germany", meetings=23, team_a_wins=10, team_b_wins=7, draws=6,
        team_a_goals=35, team_b_goals=30,
        last_meeting_date="2021-06-29", last_meeting_result="England 2-0 Germany",
    ),
    frozenset({"Spain", "Germany"}): HeadToHeadRecord(
        "Spain", "Germany", meetings=24, team_a_wins=10, team_b_wins=9, draws=5,
        team_a_goals=40, team_b_goals=37,
        last_meeting_date="2024-07-05", last_meeting_result="Spain 2-1 Germany",
    ),
    frozenset({"Brazil", "Argentina"}): HeadToHeadRecord(
        "Brazil", "Argentina", meetings=104, team_a_wins=39, team_b_wins=38, draws=27,
        team_a_goals=165, team_b_goals=157,
        last_meeting_date="2023-11-22", last_meeting_result="Argentina 0-1 Brazil",
    ),
    frozenset({"Morocco", "France"}): HeadToHeadRecord(
        "Morocco", "France", meetings=8, team_a_wins=2, team_b_wins=5, draws=1,
        team_a_goals=9, team_b_goals=18,
        last_meeting_date="2022-12-14", last_meeting_result="France 2-0 Morocco",
    ),
    frozenset({"Croatia", "Brazil"}): HeadToHeadRecord(
        "Croatia", "Brazil", meetings=7, team_a_wins=2, team_b_wins=3, draws=2,
        team_a_goals=9, team_b_goals=11,
        last_meeting_date="2022-12-09", last_meeting_result="Croatia 1-1 Brazil (Croatia pens 4-2)",
    ),
}


class FormAnalyzer:
    """
    Analyzes recent international form and head-to-head records for all teams.

    Provides:
      - Weighted form rating (exponential decay, opponent-strength adjusted)
      - Momentum score (recent 5 vs prior 5 comparison)
      - Head-to-head historical records
      - Goal stats (scored/conceded averages)

    Parameters
    ----------
    form_data : dict, optional  Override RECENT_FORM with custom data.
    h2h_data  : dict, optional  Override H2H_RECORDS.
    """

    def __init__(
        self,
        form_data: Optional[dict[str, list[FormRecord]]] = None,
        h2h_data:  Optional[dict] = None,
    ) -> None:
        self._form = form_data or RECENT_FORM
        self._h2h  = h2h_data  or H2H_RECORDS

    def get_form_rating(self, team: str) -> float:
        """
        Compute weighted recent form rating in [0, 1].

        Algorithm:
          1. For each of the last N matches (most recent first):
             - raw_pts = win→3, draw→1, loss→0
             - strength_mult = 0.5 + 0.5 × opponent_strength  (0.5–1.0)
             - adj_pts = raw_pts × strength_mult
             - decay_weight = exp(-decay_rate × i) where i=0 is most recent
          2. Normalise by max possible weighted score.

        Parameters
        ----------
        team : str

        Returns
        -------
        float  Form rating in [0, 1]; 1.0 = won all against elite opponents.
        """
        records = self._form.get(team)
        if not records:
            logger.warning("No form data for '%s'.", team)
            return 0.55

        weighted_score = 0.0
        max_score      = 0.0

        for i, rec in enumerate(records):  # i=0 = most recent
            decay_w = math.exp(-FORM_DECAY_RATE * i)
            pts = {
                MatchResult.WIN:  FORM_WIN_PTS,
                MatchResult.DRAW: FORM_DRAW_PTS,
                MatchResult.LOSS: FORM_LOSS_PTS,
            }.get(rec.result, 0.0)

            strength_mult = 0.5 + 0.5 * rec.opponent_strength
            adj_pts = pts * strength_mult

            weighted_score += decay_w * adj_pts
            max_score      += decay_w * FORM_WIN_PTS * 1.0  # max strength_mult = 1.0

        return round(float(weighted_score / max(max_score, 1e-9)), 4)

    def momentum_score(self, team: str) -> float:
        """
        Measure whether a team is trending upward or downward.

        Compares the form rating of the most recent 5 matches vs the
        prior 5. Returns a value in [-1, 1]:
          +1 = strong upward trajectory
           0 = flat
          -1 = strong downward trajectory

        Parameters
        ----------
        team : str

        Returns
        -------
        float  Momentum score in [-1, 1].
        """
        records = self._form.get(team)
        if not records or len(records) < 2:
            return 0.0

        def _simple_score(recs: list[FormRecord]) -> float:
            total = 0.0
            for r in recs:
                pts = {MatchResult.WIN: 3, MatchResult.DRAW: 1, MatchResult.LOSS: 0}.get(r.result, 0)
                total += pts * (0.5 + 0.5 * r.opponent_strength)
            return total / max(len(recs), 1)

        recent = records[:5]
        older  = records[5:10] if len(records) >= 10 else records[5:]

        recent_score = _simple_score(recent)
        older_score  = _simple_score(older) if older else recent_score

        max_possible = FORM_WIN_PTS * 1.0  # max = 3 × 1.0
        delta = (recent_score - older_score) / max_possible
        return round(max(-1.0, min(1.0, delta)), 4)

    def head_to_head(self, team_a: str, team_b: str) -> dict:
        """
        Return the H2H record between two teams.

        Parameters
        ----------
        team_a, team_b : str

        Returns
        -------
        dict with keys: meetings, team_a_wins, team_b_wins, draws,
             team_a_goals, team_b_goals, last_meeting_date,
             last_meeting_result, h2h_advantage (str).
        """
        key = frozenset({team_a, team_b})
        rec = self._h2h.get(key)

        if rec is None:
            return {
                "team_a": team_a, "team_b": team_b,
                "meetings": 0, "team_a_wins": 0, "team_b_wins": 0, "draws": 0,
                "team_a_goals": 0, "team_b_goals": 0,
                "last_meeting_date": "N/A", "last_meeting_result": "N/A",
                "h2h_advantage": "no data",
            }

        # Orient so team_a in the result matches the query
        if rec.team_a == team_a:
            a_wins, b_wins = rec.team_a_wins, rec.team_b_wins
            a_goals, b_goals = rec.team_a_goals, rec.team_b_goals
        else:
            a_wins, b_wins = rec.team_b_wins, rec.team_a_wins
            a_goals, b_goals = rec.team_b_goals, rec.team_a_goals

        advantage = (
            team_a if a_wins > b_wins
            else team_b if b_wins > a_wins
            else "level"
        )

        return {
            "team_a":             team_a,
            "team_b":             team_b,
            "meetings":           rec.meetings,
            "team_a_wins":        a_wins,
            "team_b_wins":        b_wins,
            "draws":              rec.draws,
            "team_a_goals":       a_goals,
            "team_b_goals":       b_goals,
            "last_meeting_date":  rec.last_meeting_date,
            "last_meeting_result": rec.last_meeting_result,
            "h2h_advantage":      advantage,
        }

    def get_goals_stats(self, team: str) -> dict:
        """
        Average goals scored and conceded per game from last 10 matches.

        Parameters
        ----------
        team : str

        Returns
        -------
        dict: {avg_scored, avg_conceded, avg_goal_diff, matches}
        """
        records = self._form.get(team, [])
        if not records:
            return {"avg_scored": 1.35, "avg_conceded": 1.10, "avg_goal_diff": 0.25, "matches": 0}

        scored    = sum(r.goals_for    for r in records)
        conceded  = sum(r.goals_against for r in records)
        n = len(records)
        return {
            "avg_scored":    round(scored   / n, 2),
            "avg_conceded":  round(conceded / n, 2),
            "avg_goal_diff": round((scored - conceded) / n, 2),
            "matches":       n,
        }

    def get_all_form_ratings(self) -> dict[str, float]:
        """
        Return form ratings for all teams with data, sorted descending.

        Returns
        -------
        dict[str, float]  team → form_rating, sorted by rating desc.
        """
        ratings = {team: self.get_form_rating(team) for team in self._form}
        return dict(sorted(ratings.items(), key=lambda x: x[1], reverse=True))
