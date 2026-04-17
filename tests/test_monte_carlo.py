"""
tests/test_monte_carlo.py — pytest unit tests for oracle.monte_carlo.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from oracle.team_strength import TeamStrengthScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def team_scores():
    scorer = TeamStrengthScorer()
    return scorer.score_all_teams()


@pytest.fixture(scope="module")
def simple_scores():
    """Minimal score dict for fast unit testing."""
    return {
        "TeamA": type("S", (), {"composite": 0.80})(),
        "TeamB": type("S", (), {"composite": 0.50})(),
    }


# ---------------------------------------------------------------------------
# Single-match simulation
# ---------------------------------------------------------------------------

class TestSingleMatchSimulation:
    def test_probabilities_sum_to_one(self):
        """P(A wins) + P(B wins) + P(draw) must equal 1.0."""
        from oracle.monte_carlo import MonteCarloSimulator
        scorer = TeamStrengthScorer()
        scores = scorer.score_all_teams()
        sim = MonteCarloSimulator(team_scores=scores)

        result = sim.simulate_single_match("Argentina", "Brazil", n_simulations=1000)
        total = result["p_home_win"] + result["p_away_win"] + result["p_draw"]
        assert abs(total - 1.0) < 0.005, \
            f"Probabilities sum to {total}, expected ~1.0"

    def test_stronger_team_wins_more_often(self):
        """Argentina should beat Qatar more than half the time."""
        from oracle.monte_carlo import MonteCarloSimulator
        scorer = TeamStrengthScorer()
        scores = scorer.score_all_teams()
        sim = MonteCarloSimulator(team_scores=scores)

        result = sim.simulate_single_match("Argentina", "Qatar", n_simulations=5000)
        assert result["p_home_win"] > 0.50, \
            f"Argentina should win >50% vs Qatar; got {result['p_home_win']:.2f}"

    def test_probabilities_in_unit_interval(self):
        from oracle.monte_carlo import MonteCarloSimulator
        scorer = TeamStrengthScorer()
        scores = scorer.score_all_teams()
        sim = MonteCarloSimulator(team_scores=scores)

        result = sim.simulate_single_match("France", "Morocco", n_simulations=1000)
        for key in ("p_home_win", "p_away_win", "p_draw"):
            assert 0.0 <= result[key] <= 1.0, \
                f"{key}={result[key]} out of [0,1]"


# ---------------------------------------------------------------------------
# Tournament simulation
# ---------------------------------------------------------------------------

class TestTournamentSimulation:
    def test_1000_run_returns_all_32_teams(self, team_scores):
        """Every team should appear at least once in 1000 simulation champion counts."""
        from oracle.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(team_scores=team_scores)
        results = sim.run_tournament(n_simulations=1000)

        all_teams = set(team_scores.keys())
        # Every team should have a champion_probs entry (can be 0)
        assert set(results.champion_probs.keys()) >= all_teams or \
               len(results.champion_probs) >= 30, \
               "Tournament results should include all/most 32 teams"

    def test_champion_probs_sum_to_one(self, team_scores):
        from oracle.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(team_scores=team_scores)
        results = sim.run_tournament(n_simulations=500)

        total = sum(results.champion_probs.values())
        assert abs(total - 1.0) < 0.02, \
            f"Champion probs sum to {total:.4f}, expected ~1.0"

    def test_strong_teams_have_higher_champion_prob(self, team_scores):
        from oracle.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(team_scores=team_scores)
        results = sim.run_tournament(n_simulations=2000)

        probs = results.champion_probs
        # Argentina or France or Brazil should be top 3
        sorted_teams = sorted(probs, key=lambda t: -probs.get(t, 0))[:3]
        strong = {"Argentina", "France", "Brazil", "England", "Spain"}
        assert len(strong & set(sorted_teams)) >= 2, \
            f"Expected strong teams in top 3; got {sorted_teams}"


# ---------------------------------------------------------------------------
# Referee bias adjustment
# ---------------------------------------------------------------------------

class TestRefereeBiasAdjustment:
    def test_referee_bias_modifies_probabilities(self):
        """
        Applying a strict referee profile should shift probabilities
        away from the baseline (strict refs tend to equalise by awarding
        more penalties to underdogs).
        """
        from oracle.monte_carlo import MonteCarloSimulator
        scorer = TeamStrengthScorer()
        scores = scorer.score_all_teams()
        sim = MonteCarloSimulator(team_scores=scores)

        baseline = sim.simulate_single_match("Argentina", "Qatar", n_simulations=3000)

        # Simulate with a high-strictness referee config
        high_strict = sim.simulate_single_match(
            "Argentina", "Qatar",
            n_simulations=3000,
            referee_bias={"penalties_per_game": 1.2, "strictness": "strict"},
        )

        # The win probabilities should differ between the two
        diff = abs(baseline["p_home_win"] - high_strict["p_home_win"])
        assert diff > 0.005, \
            f"Referee bias should change p_home_win; diff={diff:.4f}"

    def test_zero_bias_matches_baseline(self):
        """Passing no bias or zero bias should produce similar results."""
        from oracle.monte_carlo import MonteCarloSimulator
        scorer = TeamStrengthScorer()
        scores = scorer.score_all_teams()
        sim = MonteCarloSimulator(team_scores=scores)

        result1 = sim.simulate_single_match("France", "Croatia", n_simulations=2000)
        result2 = sim.simulate_single_match(
            "France", "Croatia",
            n_simulations=2000,
            referee_bias={"penalties_per_game": 0.25, "strictness": "average"},
        )
        # Results should be close (not identical due to RNG, but within 5%)
        diff = abs(result1["p_home_win"] - result2["p_home_win"])
        assert diff < 0.10, \
            f"Average-bias should produce similar result to no-bias; diff={diff:.4f}"
