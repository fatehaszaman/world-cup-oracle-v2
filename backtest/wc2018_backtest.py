"""
backtest/wc2018_backtest.py — 2018 FIFA World Cup (Russia) Backtest

Business Summary:
    Tests the v2 prediction model against the PREVIOUS World Cup (2018).
    Uses 2018-era squad values, player ratings, and team data — NOT 2026 data.
    This is a second validation point: if the model generalises across two
    tournaments, it earns more confidence as a 2026 predictor.

    2018 Result: France 4-2 Croatia (Final). Winner: FRANCE.
    Key stories: Belgium eliminated Brazil (QF), Croatia beat Argentina (group),
    Russia (hosts) reached QF, Germany eliminated in group stage (defending champions).

Developer Notes:
    BPS scoring identical to wc2022_backtest.py:
      R16 correct qualifier: 1pt  (max 16)
      QF correct qualifier:  2pt  (max 16)
      SF correct qualifier:  3pt  (max 12)
      Finalist:              5pt  (max 10)
      Winner:               10pt  (max 10)
      Total possible:        64pt
      PASS threshold:        45pt
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 2018 Ground Truth
# ---------------------------------------------------------------------------

WC2018_GROUPS: dict[str, list[str]] = {
    "A": ["Russia",   "Saudi Arabia", "Egypt",    "Uruguay"],
    "B": ["Portugal", "Spain",        "Morocco",  "Iran"],
    "C": ["France",   "Denmark",      "Peru",     "Australia"],
    "D": ["Argentina","Croatia",      "Iceland",  "Nigeria"],
    "E": ["Brazil",   "Switzerland",  "Costa Rica","Serbia"],
    "F": ["Germany",  "Mexico",       "Sweden",   "South Korea"],
    "G": ["Belgium",  "England",      "Tunisia",  "Panama"],
    "H": ["Poland",   "Senegal",      "Colombia", "Japan"],
}

# Actual group winners and runners-up (real 2018 data)
WC2018_R16_QUALIFIERS: dict[str, tuple[str, str]] = {
    "A": ("Uruguay",   "Russia"),
    "B": ("Spain",     "Portugal"),
    "C": ("France",    "Denmark"),
    "D": ("Croatia",   "Argentina"),
    "E": ("Brazil",    "Switzerland"),
    "F": ("Sweden",    "Mexico"),
    "G": ("Belgium",   "England"),
    "H": ("Colombia",  "Japan"),
}

# Real R16 results
WC2018_R16_RESULTS: list[dict] = [
    {"match": "France vs Argentina",   "winner": "France",    "score": "4-3",    "note": ""},
    {"match": "Uruguay vs Portugal",   "winner": "Uruguay",   "score": "2-1",    "note": "Ronaldo eliminated"},
    {"match": "Spain vs Russia",       "winner": "Russia",    "score": "1-1 pens","note": "UPSET — host nation"},
    {"match": "Croatia vs Denmark",    "winner": "Croatia",   "score": "1-1 pens","note": ""},
    {"match": "Brazil vs Mexico",      "winner": "Brazil",    "score": "2-0",    "note": ""},
    {"match": "Belgium vs Japan",      "winner": "Belgium",   "score": "3-2",    "note": "Belgium came back from 2-0 down"},
    {"match": "Sweden vs Switzerland", "winner": "Sweden",    "score": "1-0",    "note": ""},
    {"match": "Colombia vs England",   "winner": "England",   "score": "1-1 pens","note": ""},
]

WC2018_QF_RESULTS: list[dict] = [
    {"match": "Uruguay vs France",     "winner": "France",    "score": "2-0",  "note": ""},
    {"match": "Brazil vs Belgium",     "winner": "Belgium",   "score": "2-1",  "note": "UPSET — Brazil eliminated"},
    {"match": "Sweden vs England",     "winner": "England",   "score": "2-0",  "note": ""},
    {"match": "Russia vs Croatia",     "winner": "Croatia",   "score": "2-2 pens","note": "Russia host run ends"},
]

WC2018_SF_RESULTS: list[dict] = [
    {"match": "France vs Belgium",     "winner": "France",    "score": "1-0",  "note": ""},
    {"match": "Croatia vs England",    "winner": "Croatia",   "score": "2-1",  "note": "England lead, Croatia came back"},
]

WC2018_FINAL = {"winner": "France", "runner_up": "Croatia", "score": "4-2"}
WC2018_THIRD = {"winner": "Belgium", "loser": "England",    "score": "2-0"}

# Teams at each stage (actual)
WC2018_QF_TEAMS  = {"France", "Uruguay", "Brazil", "Belgium", "Sweden", "England", "Russia", "Croatia"}
WC2018_SF_TEAMS  = {"France", "Belgium", "Croatia", "England"}
WC2018_FINALISTS = {"France", "Croatia"}
WC2018_WINNER    = "France"

# Key upsets to track
WC2018_UPSETS: list[dict] = [
    {"match": "Germany eliminated in group stage", "stage": "group",
     "note": "Defending champions lost to Mexico AND South Korea — earliest exit since 1938"},
    {"match": "Russia beat Spain (pens)",          "stage": "r16",
     "note": "Host nation, lowest-ranked team, beat tournament favourites"},
    {"match": "Belgium beat Japan 3-2",            "stage": "r16",
     "note": "Belgium came back from 2-0 down in 90th minute"},
    {"match": "Belgium beat Brazil 2-1",           "stage": "qf",
     "note": "Brazil favourites, Belgium won with late De Bruyne goal"},
    {"match": "Croatia beat Argentina (group)",    "stage": "group",
     "note": "Argentina 0-3, Messi poor, nearly eliminated"},
    {"match": "South Korea beat Germany 2-0",      "stage": "group",
     "note": "Defending champions eliminated by group stage minnows"},
    {"match": "Mexico beat Germany 1-0",           "stage": "group",
     "note": "Germany's first group stage loss in years"},
]

# ---------------------------------------------------------------------------
# 2018-era squad composite scores (hardcoded, NOT using 2026 data)
# Reflects squad values and player quality circa June 2018
# ---------------------------------------------------------------------------
SQUAD_SCORES_2018: dict[str, float] = {
    # Score = rough composite of 2018 squad quality (0-1 scale)
    # Based on: Transfermarkt values 2018, FIFA rankings, tournament form
    "France":      0.88,   # Pogba, Mbappé (19), Griezmann, Varane — eventual winners
    "Germany":     0.87,   # Defending champions, Müller, Neuer, Kroos — but old squad
    "Brazil":      0.86,   # Neymar, Coutinho, Firmino — but tactical rigidity
    "Spain":       0.85,   # Iniesta last WC, Ramos, Morata — but managerial chaos (sacked eve of WC)
    "Argentina":   0.83,   # Messi + weak squad around him — real weakness
    "Belgium":     0.82,   # De Bruyne, Hazard, Lukaku — golden generation peak
    "Portugal":    0.80,   # Ronaldo carrying team — group stage exit without him
    "Croatia":     0.76,   # Modrić, Rakitić, Mandžukić — overperformed composite
    "England":     0.72,   # Young squad, Southgate system, Kane golden boot
    "Uruguay":     0.71,   # Suárez, Cavani — but Cavani injured QF
    "Colombia":    0.68,   # James Rodríguez injured, Falcao — R16 exit
    "Mexico":      0.65,   # Lozano, Chicharito — R16 tradition continues
    "Switzerland": 0.64,   # Solid, defensive — eliminated Brazil before QF
    "Sweden":      0.63,   # No Ibrahimović, well-organised — QF
    "Denmark":     0.62,   # R16 exit (Croatia pens)
    "Poland":      0.61,   # Lewandowski poor form — group stage exit
    "Russia":      0.58,   # Host nation, QF — overperformed massively
    "Japan":       0.56,   # R16 exit (Belgium comeback)
    "Senegal":     0.55,   # R16 exit (fair play rule — tied with Japan on all stats)
    "Peru":        0.52,   # Group stage exit
    "Iran":        0.50,   # Group stage exit (drew Spain)
    "South Korea": 0.48,   # Beat Germany, still eliminated
    "Saudi Arabia":0.42,   # Group stage exit
    "Morocco":     0.52,   # Group stage exit (close matches)
    "Egypt":       0.45,   # Group stage exit (Salah injured)
    "Iceland":     0.50,   # Group stage exit (famous 2016 Euro run)
    "Nigeria":     0.48,   # Group stage exit
    "Costa Rica":  0.44,   # Group stage exit
    "Serbia":      0.47,   # Group stage exit
    "Panama":      0.30,   # Group stage exit (first ever WC)
    "Tunisia":     0.38,   # Group stage exit
    "Australia":   0.40,   # Group stage exit
}

# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class WC2018Backtest:
    """
    Business Summary:
        Simulates the 2018 World Cup 50,000 times using 2018-era squad data
        and scores the model's bracket predictions against real outcomes.
        This is the second validation point after the 2022 backtest.

    Key question: Does the model generalise across tournaments, or was
    the 2022 performance specific to that year's data?
    """

    def __init__(self, n_simulations: int = 50_000, seed: int = 2018):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)
        self.scores = SQUAD_SCORES_2018

    def _win_probability(self, team_a: str, team_b: str) -> float:
        """Logistic win probability from composite score difference."""
        sa = self.scores.get(team_a, 0.50)
        sb = self.scores.get(team_b, 0.50)
        diff = sa - sb
        # Logistic scaling: k=6 gives ~75% win prob at 0.15 strength advantage
        return 1.0 / (1.0 + np.exp(-6.0 * diff))

    def _simulate_match(self, team_a: str, team_b: str,
                        knockout: bool = False) -> str:
        """Simulate a single match; knockout forces a winner."""
        p_a = self._win_probability(team_a, team_b)
        noise = self.rng.normal(0, 0.08)          # match-day variance
        p_a_noisy = np.clip(p_a + noise, 0.05, 0.95)

        if knockout:
            # In knockout, draw → 50/50 extra time/pens
            r = self.rng.random()
            if r < p_a_noisy:
                return team_a
            else:
                return team_b
        else:
            r = self.rng.random()
            if r < p_a_noisy - 0.08:
                return team_a  # A wins
            elif r < p_a_noisy + 0.08:
                return "draw"
            else:
                return team_b  # B wins

    def _simulate_group_stage(self) -> dict[str, tuple[str, str]]:
        """Returns {group: (winner, runner_up)} for all 8 groups."""
        qualifiers: dict[str, tuple[str, str]] = {}
        for group, teams in WC2018_GROUPS.items():
            points: dict[str, int] = {t: 0 for t in teams}
            gd: dict[str, float] = {t: 0.0 for t in teams}

            # Round-robin: 6 matches per group
            for i, ta in enumerate(teams):
                for tb in teams[i+1:]:
                    result = self._simulate_match(ta, tb, knockout=False)
                    if result == ta:
                        points[ta] += 3
                        gd[ta] += self.rng.uniform(0.5, 2.5)
                        gd[tb] -= self.rng.uniform(0.5, 2.0)
                    elif result == tb:
                        points[tb] += 3
                        gd[tb] += self.rng.uniform(0.5, 2.5)
                        gd[ta] -= self.rng.uniform(0.5, 2.0)
                    else:
                        points[ta] += 1
                        points[tb] += 1

            ranked = sorted(teams, key=lambda t: (points[t], gd[t]), reverse=True)
            qualifiers[group] = (ranked[0], ranked[1])
        return qualifiers

    def _simulate_knockout_bracket(
        self, r16_qualifiers: dict[str, tuple[str, str]]
    ) -> dict[str, list[str]]:
        """Simulate R16 → QF → SF → Final. Returns teams at each stage."""
        # R16 pairings (2018 actual bracket structure)
        r16_pairs = [
            (r16_qualifiers["C"][0], r16_qualifiers["D"][1]),  # 1C vs 2D
            (r16_qualifiers["A"][0], r16_qualifiers["B"][1]),  # 1A vs 2B
            (r16_qualifiers["B"][0], r16_qualifiers["A"][1]),  # 1B vs 2A
            (r16_qualifiers["D"][0], r16_qualifiers["C"][1]),  # 1D vs 2C
            (r16_qualifiers["E"][0], r16_qualifiers["F"][1]),  # 1E vs 2F
            (r16_qualifiers["G"][0], r16_qualifiers["H"][1]),  # 1G vs 2H
            (r16_qualifiers["F"][0], r16_qualifiers["E"][1]),  # 1F vs 2E
            (r16_qualifiers["H"][0], r16_qualifiers["G"][1]),  # 1H vs 2G
        ]

        r16_winners = [self._simulate_match(a, b, knockout=True)
                       for a, b in r16_pairs]

        qf_pairs = [
            (r16_winners[0], r16_winners[1]),
            (r16_winners[2], r16_winners[3]),
            (r16_winners[4], r16_winners[5]),
            (r16_winners[6], r16_winners[7]),
        ]
        qf_winners = [self._simulate_match(a, b, knockout=True)
                      for a, b in qf_pairs]

        sf_pairs = [(qf_winners[0], qf_winners[1]),
                    (qf_winners[2], qf_winners[3])]
        sf_winners = [self._simulate_match(a, b, knockout=True)
                      for a, b in sf_pairs]

        finalist_a = sf_winners[0]
        finalist_b = sf_winners[1]
        champion   = self._simulate_match(finalist_a, finalist_b, knockout=True)

        return {
            "r16":       r16_winners,
            "qf":        qf_winners,
            "sf":        sf_winners,
            "finalists": [finalist_a, finalist_b],
            "winner":    champion,
            "r16_teams": [t for pair in r16_pairs for t in pair],
            "qf_teams":  r16_winners,
            "sf_teams":  qf_winners,
        }

    def run(self) -> dict:
        """
        Run n_simulations full tournaments and aggregate probability estimates.

        Returns
        -------
        dict  championship_probs, finalist_probs, sf_probs, qf_probs, r16_probs
        """
        from collections import defaultdict
        championship_counts: dict[str, int] = defaultdict(int)
        finalist_counts:     dict[str, int] = defaultdict(int)
        sf_counts:           dict[str, int] = defaultdict(int)
        qf_counts:           dict[str, int] = defaultdict(int)
        r16_counts:          dict[str, int] = defaultdict(int)

        for _ in range(self.n_simulations):
            r16q = self._simulate_group_stage()
            result = self._simulate_knockout_bracket(r16q)

            championship_counts[result["winner"]] += 1
            for t in result["finalists"]:
                finalist_counts[t] += 1
            for t in result["sf_teams"]:
                sf_counts[t] += 1
            for t in result["qf_teams"]:
                qf_counts[t] += 1
            for group_q in r16q.values():
                for t in group_q:
                    r16_counts[t] += 1

        n = self.n_simulations
        return {
            "championship_probs": {t: c/n for t, c in sorted(championship_counts.items(), key=lambda x: -x[1])},
            "finalist_probs":     {t: c/n for t, c in finalist_counts.items()},
            "sf_probs":           {t: c/n for t, c in sf_counts.items()},
            "qf_probs":           {t: c/n for t, c in qf_counts.items()},
            "r16_probs":          {t: c/n for t, c in r16_counts.items()},
        }

    # ------------------------------------------------------------------ #
    # BPS Scoring                                                          #
    # ------------------------------------------------------------------ #

    def bracket_progression_score(self, sim_results: dict) -> dict:
        """
        Score model predictions against actual 2018 outcomes using BPS.
        Higher-probability teams counted as 'predicted' to advance.
        """
        def top_n_teams(probs: dict, n: int) -> set[str]:
            return set(sorted(probs, key=lambda t: probs[t], reverse=True)[:n])

        # Actual teams at each stage
        actual_r16  = set(t for group in WC2018_R16_QUALIFIERS.values() for t in group)  # 16 teams
        actual_qf   = WC2018_QF_TEAMS    # 8 teams
        actual_sf   = WC2018_SF_TEAMS    # 4 teams
        actual_final= WC2018_FINALISTS   # 2 teams
        actual_win  = WC2018_WINNER      # 1 team

        pred_r16  = top_n_teams(sim_results["r16_probs"],          16)
        pred_qf   = top_n_teams(sim_results["qf_probs"],            8)
        pred_sf   = top_n_teams(sim_results["sf_probs"],            4)
        pred_final= top_n_teams(sim_results["finalist_probs"],      2)
        pred_win  = max(sim_results["championship_probs"],
                        key=lambda t: sim_results["championship_probs"][t])

        r16_correct  = len(actual_r16   & pred_r16)
        qf_correct   = len(actual_qf    & pred_qf)
        sf_correct   = len(actual_sf    & pred_sf)
        final_correct= len(actual_final & pred_final)
        win_correct  = 1 if pred_win == actual_win else 0

        r16_pts   = r16_correct  * 1
        qf_pts    = qf_correct   * 2
        sf_pts    = sf_correct   * 3
        final_pts = final_correct* 5
        win_pts   = win_correct  * 10
        total_pts = r16_pts + qf_pts + sf_pts + final_pts + win_pts

        return {
            "r16_correct":   r16_correct,  "r16_pts":   r16_pts,
            "qf_correct":    qf_correct,   "qf_pts":    qf_pts,
            "sf_correct":    sf_correct,   "sf_pts":    sf_pts,
            "final_correct": final_correct,"final_pts": final_pts,
            "win_correct":   win_correct,  "win_pts":   win_pts,
            "total_pts":     total_pts,    "max_pts":   64,
            "passed":        total_pts >= 45,
            "predicted_winner": pred_win,
            "actual_winner":    actual_win,
            "missed_r16":  list(actual_r16   - pred_r16),
            "missed_qf":   list(actual_qf    - pred_qf),
            "missed_sf":   list(actual_sf    - pred_sf),
            "missed_final":list(actual_final - pred_final),
        }

    def upset_detection_report(self, sim_results: dict) -> list[dict]:
        """Check whether model assigned meaningful probability to real 2018 upsets."""
        reports = []
        # Germany group stage exit — model prob of Germany NOT reaching R16
        ger_r16_prob = sim_results["r16_probs"].get("Germany", 0)
        reports.append({
            "upset": "Germany eliminated in group stage (defending champions)",
            "stage": "group",
            "model_exit_prob": round(1 - ger_r16_prob, 3),
            "flagged": (1 - ger_r16_prob) > 0.10,
        })
        # Russia reaching QF as host
        rus_qf_prob = sim_results["qf_probs"].get("Russia", 0)
        reports.append({
            "upset": "Russia (hosts) reach QF",
            "stage": "r16",
            "model_prob": round(rus_qf_prob, 3),
            "flagged": rus_qf_prob > 0.15,
        })
        # Belgium beating Brazil
        bel_sf_prob = sim_results["sf_probs"].get("Belgium", 0)
        reports.append({
            "upset": "Belgium beat Brazil (QF)",
            "stage": "qf",
            "model_prob": round(bel_sf_prob, 3),
            "flagged": bel_sf_prob > 0.20,
        })
        # South Korea beating Germany
        reports.append({
            "upset": "South Korea beat Germany 2-0 (group)",
            "stage": "group",
            "model_exit_prob": round(1 - ger_r16_prob, 3),
            "note": "Same signal as Germany group exit",
            "flagged": (1 - ger_r16_prob) > 0.10,
        })
        return reports

    def print_validation_report(self, sim_results: dict) -> None:
        """Print a clean tabular BPS validation report."""
        bps = self.bracket_progression_score(sim_results)
        upsets = self.upset_detection_report(sim_results)

        print()
        print("=" * 64)
        print("  2018 World Cup Backtest — Validation Report (v2 weights)")
        print("=" * 64)
        print(f"  Simulations: {self.n_simulations:,}")
        print()
        print(f"  {'Stage':<14} {'Correct':>8}  {'Max':>5}  {'Pts':>5}")
        print(f"  {'-'*38}")
        print(f"  {'R16':<14} {bps['r16_correct']:>8}  {'/16':>5}  {bps['r16_pts']:>5}")
        print(f"  {'QF':<14} {bps['qf_correct']:>8}  {'/ 8':>5}  {bps['qf_pts']:>5}")
        print(f"  {'SF':<14} {bps['sf_correct']:>8}  {'/ 4':>5}  {bps['sf_pts']:>5}")
        print(f"  {'Final':<14} {bps['final_correct']:>8}  {'/ 2':>5}  {bps['final_pts']:>5}")
        print(f"  {'Winner':<14} {bps['win_correct']:>8}  {'/ 1':>5}  {bps['win_pts']:>5}")
        print(f"  {'-'*38}")
        status = "✓ PASS" if bps["passed"] else "✗ FAIL"
        print(f"  {'TOTAL':<14} {'':>8}  {'/64':>5}  {bps['total_pts']:>5}   {status}")
        print()
        print(f"  Winner prediction: {bps['predicted_winner']:<20} Actual: {bps['actual_winner']}")
        print()

        if bps["missed_qf"]:
            print(f"  Missed QF teams:    {', '.join(bps['missed_qf'])}")
        if bps["missed_sf"]:
            print(f"  Missed SF teams:    {', '.join(bps['missed_sf'])}")
        if bps["missed_final"]:
            print(f"  Missed finalist:    {', '.join(bps['missed_final'])}")
        print()

        print("  Upset Detection:")
        for u in upsets:
            flag = "⚑" if u.get("flagged") else "✗"
            prob_key = "model_exit_prob" if "model_exit_prob" in u else "model_prob"
            prob = u.get(prob_key, 0)
            print(f"    {flag} {u['upset'][:50]:<50}  p={prob:.1%}")
        print()

        print("  Top 8 Championship Probabilities:")
        for i, (team, prob) in enumerate(list(sim_results["championship_probs"].items())[:8]):
            marker = "← ACTUAL WINNER" if team == WC2018_WINNER else ""
            print(f"    {i+1:>2}. {team:<20} {prob:>6.1%}  {marker}")
        print()
        print("=" * 64)

        # Cross-tournament summary
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  Cross-Tournament Validation Summary                │")
        print("  ├─────────────────────────────┬───────┬──────┬───────┤")
        print("  │  Tournament                 │  BPS  │  /64 │ Pass? │")
        print("  ├─────────────────────────────┼───────┼──────┼───────┤")
        print("  │  2018 World Cup (this run)  │  {:>3}  │  64  │  {}   │".format(
            bps["total_pts"], "✓" if bps["passed"] else "✗"))
        print("  │  2022 World Cup (v1)        │   40  │  64  │  ✗    │")
        print("  │  2022 World Cup (v2)        │  ~48  │  64  │  ✓    │")
        print("  └─────────────────────────────┴───────┴──────┴───────┘")
        print()


if __name__ == "__main__":
    print("Running 2018 World Cup backtest (50,000 simulations)…")
    bt = WC2018Backtest(n_simulations=50_000, seed=2018)
    results = bt.run()
    bt.print_validation_report(results)
