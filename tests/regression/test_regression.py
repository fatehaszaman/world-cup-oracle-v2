"""
tests/regression/test_regression.py — Regression tests against known 2022 WC outcomes.

Each test asserts that the model produces a probability within an expected range
for a specific 2022 World Cup match-up.  This guards against regressions where
refactoring accidentally changes calibration.

Expected probability ranges are based on the 2022 backtest calibration with 50k runs.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from backtest.wc2022_backtest import (
    WC2022Backtest,
    _simulate_match,
    _TEAM_STRENGTH_2022,
)
import numpy as np


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(seed=123)


@pytest.fixture(scope="module")
def bt():
    b = WC2022Backtest(n_simulations=10_000, seed=42)
    b.run()
    return b


# ---------------------------------------------------------------------------
# Match-level regression tests (win probability ranges)
# ---------------------------------------------------------------------------

def _win_prob(team_a: str, team_b: str, n: int = 5000) -> float:
    """Estimate win probability for team_a over n simulations."""
    rng = np.random.default_rng(42)
    wins = sum(
        1 for _ in range(n)
        if _simulate_match(team_a, team_b, _TEAM_STRENGTH_2022, rng, allow_draw=False)[0] == team_a
    )
    return wins / n


class TestRegressionMatchProbabilities:
    """
    10 regression cases. Each asserts that the model's win probability
    falls within a plausible empirical range.
    """

    def test_argentina_beats_qatar_group(self):
        """Argentina (0.87) vs Qatar (0.38): model should assign >70% to Argentina."""
        p = _win_prob("Argentina", "Qatar")
        assert p > 0.70, f"Argentina P(win) vs Qatar = {p:.2f}, expected >0.70"

    def test_france_beats_australia_group(self):
        """France vs Australia: France P(win) should be > 0.65."""
        p = _win_prob("France", "Australia")
        assert p > 0.65, f"France P(win) vs Australia = {p:.2f}, expected >0.65"

    def test_brazil_beats_south_korea_r16(self):
        """Brazil vs South Korea (R16): Brazil P(win) should be > 0.65."""
        p = _win_prob("Brazil", "South Korea")
        assert p > 0.65, f"Brazil P(win) = {p:.2f}, expected >0.65"

    def test_netherlands_beats_usa_r16(self):
        """Netherlands vs USA (R16): Netherlands P(win) should be > 0.55."""
        p = _win_prob("Netherlands", "USA")
        assert p > 0.55, f"Netherlands P(win) = {p:.2f}, expected >0.55"

    def test_portugal_beats_switzerland_r16(self):
        """Portugal 6-1 Switzerland: Portugal P(win) should be > 0.60."""
        p = _win_prob("Portugal", "Switzerland")
        assert p > 0.60, f"Portugal P(win) = {p:.2f}, expected >0.60"

    def test_argentina_beats_france_final(self):
        """Argentina vs France (Final): should be very close, both >35%."""
        p_arg = _win_prob("Argentina", "France", n=10_000)
        p_fra = _win_prob("France", "Argentina", n=10_000)
        assert p_arg > 0.35, f"Argentina P(win Final) = {p_arg:.2f}, expected >0.35"
        assert p_fra > 0.35, f"France P(win Final) = {p_fra:.2f}, expected >0.35"

    def test_upset_saudi_vs_argentina(self):
        """Saudi Arabia vs Argentina: underdog P(win) should be in [0.05, 0.25]."""
        p = _win_prob("Saudi Arabia", "Argentina")
        assert 0.05 <= p <= 0.35, \
            f"Saudi Arabia P(win) = {p:.2f}, expected [0.05, 0.35]"

    def test_upset_japan_vs_germany(self):
        """Japan vs Germany: underdog P(win) should be in [0.10, 0.40]."""
        p = _win_prob("Japan", "Germany")
        assert 0.10 <= p <= 0.45, \
            f"Japan P(win vs Germany) = {p:.2f}, expected [0.10, 0.45]"

    def test_upset_morocco_vs_spain(self):
        """Morocco vs Spain: Morocco P(win) should be in [0.15, 0.45]."""
        p = _win_prob("Morocco", "Spain")
        assert 0.15 <= p <= 0.50, \
            f"Morocco P(win vs Spain) = {p:.2f}, expected [0.15, 0.50]"

    def test_close_match_croatia_vs_brazil(self):
        """Croatia vs Brazil (QF): Brazil slight favourite P(win) > 0.45."""
        p = _win_prob("Brazil", "Croatia")
        assert p > 0.45, f"Brazil P(win vs Croatia) = {p:.2f}, expected >0.45"


# ---------------------------------------------------------------------------
# Tournament-level regression tests
# ---------------------------------------------------------------------------

class TestTournamentRegression:
    def test_argentina_winner_in_top2(self, bt):
        """Argentina should be in the top 2 predicted champion probabilities."""
        probs = bt._results["champion_probs"]
        top2 = sorted(probs, key=lambda t: -probs[t])[:2]
        assert "Argentina" in top2, \
            f"Argentina not in top 2 champion probs; top2={top2}"

    def test_france_finalist_in_top2(self, bt):
        """France should be in the top 2 finalist probabilities."""
        probs = bt._results["finalist_probs"]
        top2 = sorted(probs, key=lambda t: -probs[t])[:2]
        assert "France" in top2, \
            f"France not in top 2 finalist probs; top2={top2}"

    def test_qatar_lowest_champion_prob(self, bt):
        """Qatar should be one of the 3 lowest champion probabilities."""
        probs = bt._results["champion_probs"]
        bottom3 = sorted(probs, key=lambda t: probs[t])[:3]
        assert "Qatar" in bottom3, \
            f"Qatar not in bottom 3 champion probs; bottom3={bottom3}"
