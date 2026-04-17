"""
backtest/wc2022_backtest.py — Full 2022 FIFA World Cup backtest.

Uses exact 2022 tournament data and 2022-era squad strength estimates
(NOT 2026 projections) to validate the oracle model's predictive power.

EXACT 2022 RESULTS USED
------------------------
Groups:
  A: Netherlands, Senegal, Ecuador, Qatar
  B: England, USA, Iran, Wales
  C: Argentina, Saudi Arabia, Mexico, Poland
  D: France, Australia, Denmark, Tunisia
  E: Japan, Spain, Germany, Costa Rica
  F: Morocco, Croatia, Belgium, Canada
  G: Brazil, Switzerland, Cameroon, Serbia
  H: Portugal, South Korea, Uruguay, Ghana

Round of 16:
  Netherlands 3-1 USA
  Argentina 2-1 Australia
  France 3-1 Poland
  England 3-0 Senegal
  Croatia 1-1 Japan (Croatia win pens 3-1)
  Brazil 4-1 South Korea
  Morocco 0-0 Spain (Morocco win pens 3-0)
  Portugal 6-1 Switzerland

Quarter-finals:
  Croatia 1-1 Brazil (Croatia win pens 4-2)
  Argentina 2-2 Netherlands (Argentina win pens 4-3)
  Morocco 1-0 Portugal
  France 2-1 England

Semi-finals:
  Argentina 3-0 Croatia
  France 2-0 Morocco

Final:
  Argentina 3-3 France (Argentina win pens 4-2)  ← ARGENTINA WON

3rd place:
  Croatia 2-1 Morocco

Key upsets:
  Saudi Arabia 2-1 Argentina (Group C)
  Japan 2-1 Germany (Group E)
  Japan 2-1 Spain (Group E)
  Morocco eliminated Spain (R16 pens), Belgium (Group), Portugal (QF)
  Croatia eliminated Brazil (QF pens)

Bracket Prediction Score (BPS) methodology:
  R16 qualifiers: 1 pt each (max 16)
  QF qualifiers:  2 pt each (max 16)
  SF qualifiers:  3 pt each (max 12)
  Finalists:      5 pt each (max 10)
  Winner:         10 pt     (max 10)
  Total maximum:  64 pts
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2022 ground-truth tournament data
# ---------------------------------------------------------------------------

WC2022_GROUPS: dict[str, list[str]] = {
    "A": ["Netherlands", "Senegal", "Ecuador", "Qatar"],
    "B": ["England", "USA", "Iran", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Japan", "Spain", "Germany", "Costa Rica"],
    "F": ["Morocco", "Croatia", "Belgium", "Canada"],
    "G": ["Brazil", "Switzerland", "Cameroon", "Serbia"],
    "H": ["Portugal", "South Korea", "Uruguay", "Ghana"],
}

# Actual R16 qualifiers (top 2 from each group)
WC2022_R16_QUALIFIERS: set[str] = {
    # Group A
    "Netherlands", "Senegal",
    # Group B
    "England", "USA",
    # Group C
    "Argentina", "Poland",
    # Group D
    "France", "Australia",
    # Group E
    "Japan", "Spain",
    # Group F
    "Morocco", "Croatia",
    # Group G
    "Brazil", "Switzerland",
    # Group H
    "Portugal", "South Korea",
}

# Actual R16 results
WC2022_R16: list[tuple[str, str, str]] = [
    ("Netherlands", "USA",          "Netherlands"),
    ("Argentina",   "Australia",    "Argentina"),
    ("France",      "Poland",       "France"),
    ("England",     "Senegal",      "England"),
    ("Croatia",     "Japan",        "Croatia"),
    ("Brazil",      "South Korea",  "Brazil"),
    ("Morocco",     "Spain",        "Morocco"),
    ("Portugal",    "Switzerland",  "Portugal"),
]

WC2022_QF: list[tuple[str, str, str]] = [
    ("Croatia",    "Brazil",       "Croatia"),
    ("Argentina",  "Netherlands",  "Argentina"),
    ("Morocco",    "Portugal",     "Morocco"),
    ("France",     "England",      "France"),
]

WC2022_SF: list[tuple[str, str, str]] = [
    ("Argentina", "Croatia", "Argentina"),
    ("France",    "Morocco", "France"),
]

WC2022_FINAL: tuple[str, str, str] = ("Argentina", "France", "Argentina")
WC2022_THIRD: tuple[str, str, str] = ("Croatia", "Morocco", "Croatia")

WC2022_WINNER = "Argentina"

# Real major upsets
WC2022_UPSETS: list[dict] = [
    {"underdog": "Saudi Arabia", "favorite": "Argentina", "stage": "group",
     "result": "Saudi Arabia won 2-1"},
    {"underdog": "Japan",        "favorite": "Germany",   "stage": "group",
     "result": "Japan won 2-1"},
    {"underdog": "Japan",        "favorite": "Spain",     "stage": "group",
     "result": "Japan won 2-1"},
    {"underdog": "Morocco",      "favorite": "Belgium",   "stage": "group",
     "result": "Morocco won 2-0"},
    {"underdog": "Morocco",      "favorite": "Spain",     "stage": "r16",
     "result": "Morocco won pens 3-0 after 0-0"},
    {"underdog": "Morocco",      "favorite": "Portugal",  "stage": "qf",
     "result": "Morocco won 1-0"},
    {"underdog": "Croatia",      "favorite": "Brazil",    "stage": "qf",
     "result": "Croatia won pens 4-2 after 1-1"},
    {"underdog": "Australia",    "favorite": "Denmark",   "stage": "group",
     "result": "Australia through on GD"},
]

# ---------------------------------------------------------------------------
# 2022-era team strength estimates (scaled 0-1, NOT 2026 values)
# ---------------------------------------------------------------------------

_TEAM_STRENGTH_2022: dict[str, float] = {
    "Argentina":    0.87,
    "France":       0.88,
    "Brazil":       0.86,
    "England":      0.82,
    "Spain":        0.80,
    "Germany":      0.79,
    "Portugal":     0.81,
    "Netherlands":  0.77,
    "Belgium":      0.78,
    "Croatia":      0.73,
    "Denmark":      0.71,
    "Switzerland":  0.68,
    "USA":          0.62,
    "Mexico":       0.64,
    "Uruguay":      0.66,
    "Poland":       0.65,
    "Japan":        0.63,
    "South Korea":  0.60,
    "Morocco":      0.61,
    "Senegal":      0.62,
    "Ecuador":      0.58,
    "Australia":    0.55,
    "Serbia":       0.58,
    "Canada":       0.53,
    "Cameroon":     0.52,
    "Ghana":        0.51,
    "Iran":         0.50,
    "Tunisia":      0.52,
    "Saudi Arabia": 0.49,
    "Wales":        0.56,
    "Qatar":        0.38,
    "Costa Rica":   0.44,
}

ALL_2022_TEAMS = list(_TEAM_STRENGTH_2022.keys())


# ---------------------------------------------------------------------------
# Match simulation helpers
# ---------------------------------------------------------------------------

def _simulate_match(
    team_a: str,
    team_b: str,
    scores: dict[str, float],
    rng: np.random.Generator,
    n: int = 1,
    allow_draw: bool = True,
) -> list[str]:
    """
    Simulate *n* matches between team_a and team_b.
    Returns list of winners (length n).
    """
    sa = scores.get(team_a, 0.5)
    sb = scores.get(team_b, 0.5)
    total = sa + sb

    p_a = sa / total
    p_b = sb / total

    if allow_draw:
        draw_boost = 0.15
        p_a *= (1 - draw_boost)
        p_b *= (1 - draw_boost)

    winners = []
    for _ in range(n):
        r = rng.random()
        if r < p_a:
            winners.append(team_a)
        elif r < p_a + p_b:
            winners.append(team_b)
        else:
            # Draw → extra time / penalties: equal probability
            winners.append(team_a if rng.random() < 0.5 else team_b)
    return winners


def _simulate_group(
    group_teams: list[str],
    scores: dict[str, float],
    rng: np.random.Generator,
) -> list[str]:
    """Simulate a 4-team group; return top 2 qualifiers."""
    points: dict[str, int] = {t: 0 for t in group_teams}
    gd: dict[str, int] = {t: 0 for t in group_teams}

    pairs = [(group_teams[i], group_teams[j])
             for i in range(len(group_teams))
             for j in range(i + 1, len(group_teams))]

    for ta, tb in pairs:
        sa = scores.get(ta, 0.5)
        sb = scores.get(tb, 0.5)
        total = sa + sb
        r = rng.random()
        thresh_a = (sa / total) * 0.65
        thresh_d = thresh_a + 0.25
        if r < thresh_a:
            points[ta] += 3
            gd[ta] += 1
            gd[tb] -= 1
        elif r < thresh_d:
            points[ta] += 1
            points[tb] += 1
        else:
            points[tb] += 3
            gd[tb] += 1
            gd[ta] -= 1

    ranked = sorted(group_teams, key=lambda t: (points[t], gd[t]), reverse=True)
    return ranked[:2]


# ---------------------------------------------------------------------------
# Main backtest class
# ---------------------------------------------------------------------------

@dataclass
class WC2022Backtest:
    """
    Full 2022 World Cup backtest using oracle simulation with 2022-era inputs.

    Parameters
    ----------
    n_simulations:
        Number of Monte Carlo runs. Default 50,000.
    seed:
        Random seed for reproducibility.
    """

    n_simulations: int = 50_000
    seed: int = 42
    _results: Optional[dict] = field(default=None, repr=False)

    def run(self) -> dict:
        """
        Run n_simulations full tournament simulations using 2022 squad data.

        Returns
        -------
        dict with keys:
          champion_counts, finalist_counts, sf_counts, qf_counts, r16_counts
        """
        rng = np.random.default_rng(self.seed)
        scores = _TEAM_STRENGTH_2022.copy()

        champion_counts: dict[str, int] = {t: 0 for t in ALL_2022_TEAMS}
        finalist_counts: dict[str, int] = {t: 0 for t in ALL_2022_TEAMS}
        sf_counts:       dict[str, int] = {t: 0 for t in ALL_2022_TEAMS}
        qf_counts:       dict[str, int] = {t: 0 for t in ALL_2022_TEAMS}
        r16_counts:      dict[str, int] = {t: 0 for t in ALL_2022_TEAMS}

        n = self.n_simulations

        for _ in range(n):
            # Group stage
            r16: list[str] = []
            group_order: list[list[str]] = []
            for group_teams in WC2022_GROUPS.values():
                qualifiers = _simulate_group(group_teams, scores, rng)
                r16.extend(qualifiers)
                group_order.append(qualifiers)

            for t in r16:
                r16_counts[t] += 1

            # Build R16 bracket (8 matches from 16 qualifiers)
            # Pair: 1st group A vs 2nd group B, etc. (simplified pairing)
            def sim_round(bracket: list[str], allow_draw: bool = False) -> list[str]:
                winners = []
                for i in range(0, len(bracket), 2):
                    ta, tb = bracket[i], bracket[i + 1]
                    w = _simulate_match(ta, tb, scores, rng, allow_draw=allow_draw)[0]
                    winners.append(w)
                return winners

            # R16
            r16_bracket = []
            for i, grp in enumerate(group_order):
                # 1st of group i paired with 2nd of another (simplified sequential)
                r16_bracket.append(grp[0])
            for i, grp in enumerate(group_order):
                r16_bracket.append(grp[1])
            # Interleave: A1 vs B2, B1 vs A2, C1 vs D2, D1 vs C2, ...
            final_r16 = []
            ng = len(group_order)
            for i in range(0, ng, 2):
                final_r16.append(group_order[i][0])
                final_r16.append(group_order[i + 1][1] if i + 1 < ng else group_order[0][1])
                final_r16.append(group_order[i + 1][0] if i + 1 < ng else group_order[1][0])
                final_r16.append(group_order[i][1])

            qf = sim_round(final_r16[:16], allow_draw=True)
            for t in qf:
                qf_counts[t] += 1

            sf = sim_round(qf, allow_draw=True)
            for t in sf:
                sf_counts[t] += 1

            finalists = sim_round(sf, allow_draw=True)
            for t in finalists:
                finalist_counts[t] += 1

            champion = sim_round(finalists, allow_draw=True)[0]
            champion_counts[champion] += 1

        self._results = {
            "n_simulations":   n,
            "champion_probs":  {t: c / n for t, c in champion_counts.items()},
            "finalist_probs":  {t: c / n for t, c in finalist_counts.items()},
            "sf_probs":        {t: c / n for t, c in sf_counts.items()},
            "qf_probs":        {t: c / n for t, c in qf_counts.items()},
            "r16_probs":       {t: c / n for t, c in r16_counts.items()},
        }
        return self._results

    def bracket_progression_score(self) -> dict:
        """
        Compute Bracket Prediction Score (BPS) against actual 2022 results.

        Scoring:
          R16 qualifier predicted:   1 pt each (max 16)
          QF qualifier predicted:    2 pt each (max 16)
          SF qualifier predicted:    3 pt each (max 12)
          Finalist predicted:        5 pt each (max 10)
          Winner predicted:         10 pt      (max 10)
          Total maximum:            64 pts
        """
        if self._results is None:
            self.run()

        results = self._results  # type: ignore[union-attr]

        # Predict top N teams by probability at each stage
        def top_n(prob_dict: dict[str, float], n: int) -> set[str]:
            return set(sorted(prob_dict, key=lambda t: -prob_dict[t])[:n])

        pred_r16 = top_n(results["r16_probs"],      16)
        pred_qf  = top_n(results["qf_probs"],        8)
        pred_sf  = top_n(results["sf_probs"],         4)
        pred_fin = top_n(results["finalist_probs"],   2)
        pred_win = max(results["champion_probs"], key=lambda t: results["champion_probs"][t])

        # Actual participants
        actual_qf  = {r[2] for r in WC2022_R16}   # R16 winners = QF participants
        actual_sf  = {r[2] for r in WC2022_QF}
        actual_fin = {WC2022_FINAL[0], WC2022_FINAL[1]}

        r16_correct = len(pred_r16 & WC2022_R16_QUALIFIERS)
        qf_correct  = len(pred_qf  & actual_qf)
        sf_correct  = len(pred_sf  & actual_sf)
        fin_correct = len(pred_fin & actual_fin)
        win_correct = 1 if pred_win == WC2022_WINNER else 0

        bps_r16 = r16_correct * 1
        bps_qf  = qf_correct  * 2
        bps_sf  = sf_correct  * 3
        bps_fin = fin_correct * 5
        bps_win = win_correct * 10
        total   = bps_r16 + bps_qf + bps_sf + bps_fin + bps_win

        return {
            "r16": {"correct": r16_correct, "max": 16, "pts": bps_r16},
            "qf":  {"correct": qf_correct,  "max": 8,  "pts": bps_qf},
            "sf":  {"correct": sf_correct,  "max": 4,  "pts": bps_sf},
            "fin": {"correct": fin_correct, "max": 2,  "pts": bps_fin},
            "win": {"correct": win_correct, "max": 1,  "pts": bps_win,
                    "predicted": pred_win, "actual": WC2022_WINNER},
            "total": {"pts": total, "max": 64},
            "pass": total >= 45,
        }

    def upset_detection_report(self) -> dict:
        """
        For each real 2022 upset, report the probability the model assigned
        to the underdog winning.
        """
        if self._results is None:
            self.run()

        results = self._results  # type: ignore[union-attr]
        report = []

        for upset in WC2022_UPSETS:
            underdog = upset["underdog"]
            favorite = upset["favorite"]
            # Use champion probability as proxy for overall strength
            p_underdog = results["champion_probs"].get(underdog, 0.0)
            p_favorite = results["champion_probs"].get(favorite, 0.0)
            total = p_underdog + p_favorite
            match_prob = p_underdog / total if total > 0 else 0.5

            report.append({
                "underdog":       underdog,
                "favorite":       favorite,
                "stage":          upset["stage"],
                "result":         upset["result"],
                "upset_prob_pct": round(match_prob * 100, 1),
                "flagged":        match_prob >= 0.20,  # flag if >20% chance
            })

        flagged_count = sum(1 for r in report if r["flagged"])
        return {
            "upsets": report,
            "flagged": flagged_count,
            "total":   len(report),
        }

    def print_validation_report(self) -> None:
        """Print a formatted validation report to stdout."""
        bps = self.bracket_progression_score()
        upsets = self.upset_detection_report()

        print("\n" + "=" * 64)
        print("  2022 World Cup Backtest — Validation Report")
        print("=" * 64)

        print(f"\n{'Stage':<12} {'Correct':>8} {'Max':>5} {'Pts':>6}")
        print("-" * 36)
        for stage_key, label in [("r16","R16"), ("qf","QF"), ("sf","SF"),
                                   ("fin","Final"), ("win","Winner")]:
            s = bps[stage_key]
            print(f"  {label:<10} {s['correct']:>8} {s['max']:>5} {s['pts']:>6}")
        print("-" * 36)
        t = bps["total"]
        status = "✓ PASS" if bps["pass"] else "✗ FAIL"
        print(f"  {'TOTAL':<10} {'':>8} {'64':>5} {t['pts']:>6}   {status}")

        print(f"\nWinner prediction: {bps['win']['predicted']:20s}  "
              f"Actual: {bps['win']['actual']}")

        print(f"\nUpset detection: {upsets['flagged']}/{upsets['total']} flagged correctly")
        for u in upsets["upsets"]:
            flag = "⚑" if u["flagged"] else " "
            print(f"  {flag} {u['underdog']:15s} vs {u['favorite']:15s} "
                  f"({u['stage']:6s})  model: {u['upset_prob_pct']:5.1f}%")

        print("\n" + "=" * 64 + "\n")
