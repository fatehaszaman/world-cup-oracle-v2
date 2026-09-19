"""
tests/test_team_strength.py — pytest unit tests for oracle.team_strength.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from oracle.team_strength import TeamStrengthScorer, SQUAD_MARKET_VALUES_EUR_M


@pytest.fixture(scope="module")
def scorer() -> TeamStrengthScorer:
    return TeamStrengthScorer()


@pytest.fixture(scope="module")
def all_scores(scorer: TeamStrengthScorer):
    return scorer.score_all_teams()


# ---------------------------------------------------------------------------
# Composite score range
# ---------------------------------------------------------------------------

class TestCompositeScoreRange:
    def test_returns_float(self, all_scores):
        for team, score in all_scores.items():
            assert isinstance(score, float), \
                f"{team}: composite should be float, got {type(score)}"

    def test_in_unit_interval(self, all_scores):
        for team, score in all_scores.items():
            assert 0.0 <= score <= 1.0, \
                f"{team}: composite={score} out of [0,1]"

    def test_argentina_composite_not_none(self, all_scores):
        assert "Argentina" in all_scores
        assert all_scores["Argentina"] > 0.0

    def test_all_32_teams_present(self, all_scores):
        required = {
            "Argentina", "France", "Brazil", "England", "Spain", "Germany",
            "Portugal", "Netherlands", "Croatia", "Morocco", "Japan", "United States",
        }
        for team in required:
            assert team in all_scores, f"{team} missing from scores"
        assert set(all_scores) == set(SQUAD_MARKET_VALUES_EUR_M)
        assert len(all_scores) == 32


# ---------------------------------------------------------------------------
# Top-team ordering
# ---------------------------------------------------------------------------

class TestTopTeamOrdering:
    TOP5_EXPECTED = {"Argentina", "France", "Brazil", "England", "Spain"}

    def test_argentina_in_top5(self, all_scores):
        sorted_teams = sorted(all_scores, key=lambda t: -all_scores[t])
        top5 = set(sorted_teams[:5])
        assert "Argentina" in top5, \
            f"Argentina not in top 5; top5={top5}"

    def test_brazil_in_top5(self, all_scores):
        sorted_teams = sorted(all_scores, key=lambda t: -all_scores[t])
        top5 = set(sorted_teams[:5])
        assert "Brazil" in top5, \
            f"Brazil not in top 5; top5={top5}"

    def test_france_in_top5(self, all_scores):
        sorted_teams = sorted(all_scores, key=lambda t: -all_scores[t])
        top5 = set(sorted_teams[:5])
        assert "France" in top5, \
            f"France not in top 5; top5={top5}"

    def test_qatar_near_bottom(self, all_scores):
        assert "Qatar" in all_scores, "Legacy expectation requires Qatar; input table does not include it"
        sorted_teams = sorted(all_scores, key=lambda t: -all_scores[t])
        qatar_rank = sorted_teams.index("Qatar") + 1
        n = len(sorted_teams)
        assert qatar_rank > n * 0.75, \
            f"Qatar ranked {qatar_rank}/{n}, expected bottom 25%"

    def test_strong_teams_above_weak(self, all_scores):
        """Argentina and Brazil should score higher than Qatar."""
        assert "Qatar" in all_scores, "Legacy expectation requires Qatar; input table does not include it"
        assert all_scores["Argentina"] > all_scores["Qatar"]
        assert all_scores["Brazil"] > all_scores["Qatar"]


# ---------------------------------------------------------------------------
# Squad value normalisation
# ---------------------------------------------------------------------------

class TestSquadValueNormalization:
    def test_squad_value_in_unit_interval(self, scorer):
        """squad_value sub-score should be in [0,1]."""
        scores = scorer.score_all_teams()
        for team, score in scores.items():
            sv = scorer.score_squad_value(team)
            assert 0.0 <= sv <= 1.0, \
                f"{team}: squad_value_score={sv} out of [0,1]"

    def test_england_high_squad_value(self, all_scores, scorer):
        """England should have a squad value score in the top 5."""
        sorted_teams = sorted(all_scores, key=lambda t: -scorer.score_squad_value(t))
        top5 = sorted_teams[:5]
        assert "England" in top5, f"England squad value not top 5; got {top5}"

    def test_normalised_values_spread(self, all_scores):
        """Retain the original minimum composite spread expectation of 0.3."""
        values = list(all_scores.values())
        spread = max(values) - min(values)
        assert spread >= 0.3, f"Composite score spread too narrow: {spread:.3f}"


# ---------------------------------------------------------------------------
# Historical score
# ---------------------------------------------------------------------------

class TestHistoricalScore:
    def test_historical_score_range(self, all_scores, scorer):
        for team, score in all_scores.items():
            assert 0.0 <= scorer.score_historical_performance(team) <= 1.0, \
                f"{team}: historical_score out of [0,1]"

    def test_brazil_high_historical(self, all_scores, scorer):
        """Retain the legacy top-three hypothesis; this scorer uses recent editions."""
        sorted_teams = sorted(all_scores, key=lambda t: -scorer.score_historical_performance(t))
        brazil_rank = sorted_teams.index("Brazil") + 1
        assert brazil_rank <= 3, \
            f"Brazil historical rank={brazil_rank}, expected top 3"
