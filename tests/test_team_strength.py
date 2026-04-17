"""
tests/test_team_strength.py — pytest unit tests for oracle.team_strength.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from oracle.team_strength import TeamStrengthScorer


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
            assert isinstance(score.composite, float), \
                f"{team}: composite should be float, got {type(score.composite)}"

    def test_in_unit_interval(self, all_scores):
        for team, score in all_scores.items():
            assert 0.0 <= score.composite <= 1.0, \
                f"{team}: composite={score.composite} out of [0,1]"

    def test_argentina_composite_not_none(self, all_scores):
        assert "Argentina" in all_scores
        assert all_scores["Argentina"].composite > 0.0

    def test_all_32_teams_present(self, all_scores):
        required = {
            "Argentina", "France", "Brazil", "England", "Spain", "Germany",
            "Portugal", "Netherlands", "Croatia", "Morocco", "Japan", "USA",
        }
        for team in required:
            assert team in all_scores, f"{team} missing from scores"


# ---------------------------------------------------------------------------
# Top-team ordering
# ---------------------------------------------------------------------------

class TestTopTeamOrdering:
    TOP5_EXPECTED = {"Argentina", "France", "Brazil", "England", "Spain"}

    def test_argentina_in_top5(self, all_scores):
        sorted_teams = sorted(all_scores, key=lambda t: -all_scores[t].composite)
        top5 = set(sorted_teams[:5])
        assert "Argentina" in top5, \
            f"Argentina not in top 5; top5={top5}"

    def test_brazil_in_top5(self, all_scores):
        sorted_teams = sorted(all_scores, key=lambda t: -all_scores[t].composite)
        top5 = set(sorted_teams[:5])
        assert "Brazil" in top5, \
            f"Brazil not in top 5; top5={top5}"

    def test_france_in_top5(self, all_scores):
        sorted_teams = sorted(all_scores, key=lambda t: -all_scores[t].composite)
        top5 = set(sorted_teams[:5])
        assert "France" in top5, \
            f"France not in top 5; top5={top5}"

    def test_qatar_near_bottom(self, all_scores):
        sorted_teams = sorted(all_scores, key=lambda t: -all_scores[t].composite)
        qatar_rank = sorted_teams.index("Qatar") + 1
        n = len(sorted_teams)
        assert qatar_rank > n * 0.75, \
            f"Qatar ranked {qatar_rank}/{n}, expected bottom 25%"

    def test_strong_teams_above_weak(self, all_scores):
        """Argentina and Brazil should score higher than Qatar."""
        assert all_scores["Argentina"].composite > all_scores["Qatar"].composite
        assert all_scores["Brazil"].composite > all_scores["Qatar"].composite


# ---------------------------------------------------------------------------
# Squad value normalisation
# ---------------------------------------------------------------------------

class TestSquadValueNormalization:
    def test_squad_value_in_unit_interval(self, scorer):
        """squad_value sub-score should be in [0,1]."""
        scores = scorer.score_all_teams()
        for team, score in scores.items():
            sv = score.squad_value_score
            assert 0.0 <= sv <= 1.0, \
                f"{team}: squad_value_score={sv} out of [0,1]"

    def test_england_high_squad_value(self, all_scores):
        """England should have a squad value score in the top 5."""
        sorted_teams = sorted(all_scores, key=lambda t: -all_scores[t].squad_value_score)
        top5 = sorted_teams[:5]
        assert "England" in top5, f"England squad value not top 5; got {top5}"

    def test_normalised_values_spread(self, all_scores):
        """Scores should not be bunched — range should be at least 0.5."""
        values = [s.composite for s in all_scores.values()]
        spread = max(values) - min(values)
        assert spread >= 0.3, f"Composite score spread too narrow: {spread:.3f}"


# ---------------------------------------------------------------------------
# Historical score
# ---------------------------------------------------------------------------

class TestHistoricalScore:
    def test_historical_score_range(self, all_scores):
        for team, score in all_scores.items():
            assert 0.0 <= score.historical_score <= 1.0, \
                f"{team}: historical_score out of [0,1]"

    def test_brazil_high_historical(self, all_scores):
        """Brazil (5 titles) should have the highest or near-highest historical score."""
        sorted_teams = sorted(all_scores, key=lambda t: -all_scores[t].historical_score)
        brazil_rank = sorted_teams.index("Brazil") + 1
        assert brazil_rank <= 3, \
            f"Brazil historical rank={brazil_rank}, expected top 3"
