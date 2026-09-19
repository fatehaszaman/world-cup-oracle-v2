"""Tests of the implemented simulator API, retaining legacy model hypotheses.

Technical checks are not evidence of predictive calibration. Seeded referee
fixtures isolate the adjustment from random sampling and real-world narratives.
"""
import numpy as np
import pytest
from oracle.monte_carlo import TournamentSimulator
from oracle.team_strength import TeamStrengthScorer
from oracle.bracket import WC2026_GROUPS


@pytest.fixture(scope="module")
def team_scores():
    return TeamStrengthScorer().score_all_teams()


def match(scores, a="Argentina", b="Brazil", n=1000, delta=None):
    sim = TournamentSimulator()
    if delta is not None:
        class Bias:
            def get_match_bias_factor(self, *args, base_prob_a, **kwargs):
                p = base_prob_a + delta
                return {"adjusted_prob_a": p, "adjusted_prob_b": 1 - p,
                        "bias_magnitude": abs(delta)}
        sim._referee_bias_analyzer = Bias()
    return sim.simulate_match(a, b, scores, n_simulations=n,
                              referee="fixture" if delta is not None else None,
                              rng=np.random.default_rng(42))


class TestSingleMatchSimulation:
    def test_probabilities_sum_to_one(self, team_scores):
        r = match(team_scores)
        assert sum(r[k] for k in ("win_prob_a", "win_prob_b", "draw_prob")) == pytest.approx(1, abs=2e-6)

    def test_stronger_team_wins_more_often(self, team_scores):
        # Original >50% expectation is retained, not relaxed. The absent
        # Qatar input is surfaced rather than silently testing a fallback.
        assert "Qatar" in team_scores, "Qatar is absent from the scorer input table"
        assert match(team_scores, "Argentina", "Qatar", 5000)["win_prob_a"] > 0.50

    def test_probabilities_in_unit_interval(self, team_scores):
        r = match(team_scores, "France", "Morocco")
        for k in ("win_prob_a", "win_prob_b", "draw_prob"):
            assert 0 <= r[k] <= 1


class TestTournamentSimulation:
    def test_1000_run_returns_all_scenario_teams(self, team_scores):
        df = TournamentSimulator().run_tournament(team_scores, n_runs=1000)
        assert set(df["team"]) == {t for g in WC2026_GROUPS.values() for t in g}
        assert len(df) == 48

    def test_champion_probs_sum_to_one(self, team_scores):
        df = TournamentSimulator().run_tournament(team_scores, n_runs=500)
        assert abs(df["champion_prob"].sum() - 1) < 0.02

    def test_strong_teams_have_higher_champion_prob(self, team_scores):
        df = TournamentSimulator().run_tournament(team_scores, n_runs=2000)
        top3 = set(df.nlargest(3, "champion_prob")["team"])
        strong = {"Argentina", "France", "Brazil", "England", "Spain"}
        assert len(strong & top3) >= 2, f"Expected strong teams in top 3; got {top3}"


class TestRefereeBiasAdjustment:
    def test_referee_bias_modifies_probabilities(self, team_scores):
        baseline = match(team_scores, n=3000)
        biased = match(team_scores, n=3000, delta=0.1)
        assert biased["referee_adjusted"]
        assert abs(baseline["win_prob_a"] - biased["win_prob_a"]) > 0.005
        assert biased["draw_prob"] == baseline["draw_prob"]
        assert sum(biased[k] for k in ("win_prob_a", "win_prob_b", "draw_prob")) == pytest.approx(1, abs=2e-6)

    def test_zero_bias_matches_baseline(self, team_scores):
        baseline = match(team_scores, n=2000)
        neutral = match(team_scores, n=2000, delta=0)
        assert neutral["referee_adjusted"]
        for k in ("win_prob_a", "win_prob_b", "draw_prob"):
            assert neutral[k] == baseline[k]


@pytest.mark.parametrize("n", [2, 3, 8, 16, 24, 32])
def test_knockout_eliminates_every_nonchampion_once(n):
    sim = TournamentSimulator()
    teams = {f"Team{i}": "1st" for i in range(n)}
    scores = {team: 0.5 for team in teams}
    matches = []
    def fixture(a, b, scores, rng):
        assert a != b
        matches.append((a, b))
        return b
    sim._simulate_ko_match = fixture
    result = sim.simulate_knockout(teams, scores, np.random.default_rng(42))
    assert len(matches) == n - 1
    assert len({a for a, b in matches}) == n - 1  # fixture always eliminates A
    assert set(result) == set(teams)
    assert [t for t, stage in result.items() if stage == "winner"] == [matches[-1][1]]
    assert sum(stage in ("final", "winner") for stage in result.values()) == 2


def test_eight_team_bracket_records_reaching_not_winning_round():
    sim = TournamentSimulator()
    teams = {f"Team{i}": "1st" for i in range(8)}
    r = sim.simulate_knockout(teams, {}, np.random.default_rng(42))
    assert sum(stage == "quarter_final" for stage in r.values()) == 4
    assert sum(stage == "semi_final" for stage in r.values()) == 2
    assert sum(stage == "final" for stage in r.values()) == 1
    assert sum(stage == "winner" for stage in r.values()) == 1
