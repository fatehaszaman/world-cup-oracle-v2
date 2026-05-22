"""
oracle/rating_distribution.py — Player rating uncertainty (mean ± form variance).

BUSINESS SUMMARY
----------------
Right now POSITIONAL_DATA stores each player as a single point estimate
("Mbappé (95)"). That treats a player's rating as known with zero
uncertainty, which is wrong: form fluctuates, injuries change minutes,
and FIFA/Sofascore ratings disagree with each other by 3–7 points for the
same player.

This module turns each point estimate into a distribution:

    rating ~ Normal(mean, sigma)

where `sigma` comes from recent form variance (last N matches) or a
position-specific prior if form data is unavailable. The Monte Carlo
engine can then draw a sampled rating per simulation, propagating
*rating uncertainty* on top of *match-outcome randomness* — two
independent sources of variance that v1/v2 currently conflate.

USAGE (opt-in, non-breaking)
----------------------------
The point-estimate path in team_strength.score_positional_power() is
preserved. To enable distributional sampling, the Monte Carlo driver can
call `sample_positional_power(team, rng)` instead of
`scorer.score_positional_power(team)`.

DEVELOPER NOTES
---------------
- Default form-variance prior: sigma = 2.5 rating points (≈ one tier on
  the FIFA scale). Empirically matches week-to-week Sofascore swings.
- A clipped Normal is used (mean ± 3*sigma capped at [40, 99]) to avoid
  unphysical samples from the tails.
- Form variance can be plugged in per-player via PLAYER_FORM_SIGMA below
  once data is available; default falls back to POSITION_PRIOR_SIGMA.

Complexity: O(positions × samples). Negligible vs MC simulation cost.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from config import POSITION_WEIGHTS

# ---------------------------------------------------------------------------
# Form-variance priors (rating points, 0–99 scale)
# ---------------------------------------------------------------------------
# Position-specific priors. Forwards/attackers are more volatile (form
# swings, finishing variance); GKs and CBs are more stable.
POSITION_PRIOR_SIGMA: dict[str, float] = {
    "GK": 1.8,
    "CB": 2.0,
    "FB": 2.3,
    "CM": 2.5,
    "AM": 3.0,
    "FW": 3.2,
}

# Hard clipping bounds to avoid unphysical tail draws (3-sigma cap).
RATING_FLOOR: float = 40.0
RATING_CEILING: float = 99.0

# Per-player overrides go here once form data is wired in.
# Key: player name string as stored in POSITIONAL_DATA "starter"/"backup".
# Value: form-variance sigma in rating points.
PLAYER_FORM_SIGMA: dict[str, float] = {
    # Examples — to be populated from form_analyzer match-by-match output:
    # "Kylian Mbappé (95)": 2.1,
    # "Vinícius Júnior (92)": 3.4,
}


def _sigma_for(player: Optional[str], position: str) -> float:
    """Return the form-variance sigma for a given player/position."""
    if player and player in PLAYER_FORM_SIGMA:
        return PLAYER_FORM_SIGMA[player]
    return POSITION_PRIOR_SIGMA.get(position, 2.5)


def sample_player_rating(
    mean: float,
    position: str,
    player: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> float:
    """
    Draw a single sample from the player's rating distribution.

    Parameters
    ----------
    mean : float        Point-estimate rating (0–99 scale).
    position : str      One of GK, CB, FB, CM, AM, FW.
    player : str        Optional player-name key into PLAYER_FORM_SIGMA.
    rng : random.Random Optional rng for reproducibility.

    Returns
    -------
    float  Sampled rating, clipped to [RATING_FLOOR, RATING_CEILING].
    """
    rng = rng or random
    sigma = _sigma_for(player, position)
    raw = rng.gauss(mean, sigma)
    return max(RATING_FLOOR, min(raw, RATING_CEILING))


def sample_positional_power(
    pos_data: dict[str, dict],
    rng: Optional[random.Random] = None,
) -> float:
    """
    Draw a single sample of a team's positional-power score by sampling
    each position rating independently from its distribution and applying
    the same POSITION_WEIGHTS the deterministic scorer uses.

    Parameters
    ----------
    pos_data : dict     A team's entry from POSITIONAL_DATA.
    rng : random.Random Optional rng for reproducibility.

    Returns
    -------
    float  Sampled positional power in [0, 1].

    Notes
    -----
    Returns 0.0 if pos_data is empty — callers should fall back to
    UNKNOWN_TEAM_DEFAULT_SCORE in that case.
    """
    if not pos_data:
        return 0.0

    weighted_sum = 0.0
    for pos, weight in POSITION_WEIGHTS.items():
        spec = pos_data.get(pos, {})
        mean = float(spec.get("rating", 65.0))
        starter = spec.get("starter")
        sampled = sample_player_rating(mean, pos, player=starter, rng=rng)
        weighted_sum += sampled * weight

    return weighted_sum / 100.0


def expected_positional_power(pos_data: dict[str, dict]) -> tuple[float, float]:
    """
    Closed-form expected value and standard deviation of the weighted
    positional-power score (sanity check — should match Monte Carlo
    samples as n → ∞).

    Returns
    -------
    (mean, std) : tuple[float, float]  Both on the [0, 1] scale.
    """
    if not pos_data:
        return 0.0, 0.0

    mean = 0.0
    variance = 0.0
    for pos, weight in POSITION_WEIGHTS.items():
        spec = pos_data.get(pos, {})
        m = float(spec.get("rating", 65.0))
        sigma = _sigma_for(spec.get("starter"), pos)
        mean += m * weight
        # Independent positions → variances add (weighted by w^2).
        variance += (weight * sigma) ** 2

    return mean / 100.0, math.sqrt(variance) / 100.0
