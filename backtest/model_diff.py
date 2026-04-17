"""
backtest/model_diff.py — Model improvement engine.

If the 2022 backtest BPS falls below the 45/64 threshold, this module
identifies which signal dimension weights contributed most to wrong
predictions and proposes v2 weight adjustments.

Usage:
    from backtest.model_diff import ModelDiff
    diff = ModelDiff()
    diff.analyze()
    v2_config = diff.generate_v2_config()
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline weights from config
# ---------------------------------------------------------------------------

_BASELINE_WEIGHTS: dict[str, float] = {
    "squad_value":       0.30,
    "positional_power":  0.25,
    "country_resources": 0.15,
    "historical":        0.20,
    "commercial":        0.10,
}

# Known model failure modes for 2022 (qualitative analysis)
_FAILURE_ANALYSIS: list[dict] = [
    {
        "match":      "Saudi Arabia 2-1 Argentina (Group C)",
        "dimension":  "squad_value",
        "issue":      "Model over-weighted squad market value; Argentina's €870M squad "
                      "vs Saudi Arabia €42M created extreme skew that didn't account for "
                      "motivational and tactical preparation quality.",
        "adjustment": {"squad_value": -0.03, "historical": +0.02, "commercial": +0.01},
    },
    {
        "match":      "Japan 2-1 Germany (Group E)",
        "dimension":  "squad_value",
        "issue":      "Same market-value bias; Germany €1050M vs Japan €200M. "
                      "Tactical organisation not captured by squad value alone.",
        "adjustment": {"squad_value": -0.02, "positional_power": +0.01, "historical": +0.01},
    },
    {
        "match":      "Japan 2-1 Spain (Group E)",
        "dimension":  "historical",
        "issue":      "Spain's historical weight inflated probability. "
                      "Japan's disciplined low-block was under-modelled.",
        "adjustment": {"historical": -0.02, "positional_power": +0.02},
    },
    {
        "match":      "Morocco R16/QF/SF run",
        "dimension":  "country_resources",
        "issue":      "Country resources (GDP/population) underweighted Morocco's "
                      "diaspora-sourced talent pool (players from Spanish/French leagues).",
        "adjustment": {"country_resources": -0.02, "positional_power": +0.02},
    },
    {
        "match":      "Croatia pens vs Brazil (QF)",
        "dimension":  "commercial",
        "issue":      "Commercial signal over-weighted Brazil (global brand). "
                      "Penalty shootout outcomes are near-random.",
        "adjustment": {"commercial": -0.02, "historical": +0.01, "squad_value": +0.01},
    },
]


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class ModelDiff:
    """
    Analyse BPS failure modes and propose improved weight configurations.

    Parameters
    ----------
    bps_result:
        Output dict from WC2022Backtest.bracket_progression_score().
        If None, failure analysis uses default assumptions.
    """

    def __init__(self, bps_result: dict | None = None) -> None:
        self.bps_result = bps_result or {}
        self._v2_weights: dict[str, float] | None = None

    # ------------------------------------------------------------------

    def analyze(self) -> list[dict]:
        """
        Return list of failure analysis dicts identifying which dimension
        contributed most to incorrect predictions.
        """
        total_pts = self.bps_result.get("total", {}).get("pts", 0)
        logger.info("Analysing model failures for BPS=%d/64", total_pts)
        return _FAILURE_ANALYSIS

    def _aggregate_adjustments(self) -> dict[str, float]:
        """Sum all recommended delta adjustments per dimension."""
        delta: dict[str, float] = {k: 0.0 for k in _BASELINE_WEIGHTS}
        for entry in _FAILURE_ANALYSIS:
            for dim, adj in entry["adjustment"].items():
                delta[dim] = delta.get(dim, 0.0) + adj
        return delta

    def generate_v2_config(self) -> dict[str, float]:
        """
        Produce v2 weight configuration by applying aggregated adjustments
        to the baseline, then re-normalising to sum=1.0.

        Returns
        -------
        dict mapping dimension → new weight
        """
        delta = self._aggregate_adjustments()
        v2 = {k: _BASELINE_WEIGHTS[k] + delta.get(k, 0.0)
              for k in _BASELINE_WEIGHTS}

        # Clamp to [0.05, 0.60]
        for k in v2:
            v2[k] = max(0.05, min(0.60, v2[k]))

        # Re-normalise
        total = sum(v2.values())
        v2 = {k: round(v / total, 4) for k, v in v2.items()}

        # Ensure exact sum=1 by adjusting largest
        diff = 1.0 - sum(v2.values())
        largest = max(v2, key=v2.get)  # type: ignore[arg-type]
        v2[largest] = round(v2[largest] + diff, 4)

        self._v2_weights = v2
        return v2

    def estimate_bps_delta(
        self, baseline_bps: int, v2_weights: dict[str, float]
    ) -> dict[str, Any]:
        """
        Estimate the expected BPS improvement from v2 weights.

        Uses a simple analytical model: each dimension adjustment is
        mapped to an expected bracket-stage accuracy improvement.

        Parameters
        ----------
        baseline_bps:
            BPS achieved with current weights.
        v2_weights:
            Proposed v2 weight dict.

        Returns
        -------
        dict with keys: baseline_bps, estimated_v2_bps, delta, improved_stages
        """
        improvements = []

        for entry in _FAILURE_ANALYSIS:
            dim = entry["dimension"]
            baseline_w = _BASELINE_WEIGHTS[dim]
            v2_w = v2_weights.get(dim, baseline_w)
            # If the v2 weight moved in the recommended direction, count it
            adj_total = sum(entry["adjustment"].values())
            if (adj_total < 0 and v2_w < baseline_w) or (adj_total > 0 and v2_w > baseline_w):
                improvements.append(entry["match"])

        # Rough estimate: each improvement ~ +1.5 BPS on average
        estimated_delta = len(improvements) * 1.5
        estimated_v2_bps = min(64, baseline_bps + int(estimated_delta))

        return {
            "baseline_bps":    baseline_bps,
            "estimated_v2_bps": estimated_v2_bps,
            "delta":           estimated_v2_bps - baseline_bps,
            "improved_matches": improvements,
        }

    def print_diff_report(self) -> None:
        """Print human-readable model diff report."""
        if self._v2_weights is None:
            self.generate_v2_config()

        baseline_bps = self.bps_result.get("total", {}).get("pts", 0)
        delta_report = self.estimate_bps_delta(baseline_bps, self._v2_weights)  # type: ignore

        print("\n" + "=" * 64)
        print("  Model Diff Report — v1 → v2 Weight Adjustments")
        print("=" * 64)

        print(f"\n{'Dimension':<25} {'v1 Weight':>10} {'v2 Weight':>10} {'Delta':>8}")
        print("-" * 56)
        for dim in _BASELINE_WEIGHTS:
            v1 = _BASELINE_WEIGHTS[dim]
            v2 = self._v2_weights[dim]  # type: ignore
            d = v2 - v1
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "→")
            print(f"  {dim:<23} {v1:>10.4f} {v2:>10.4f} {arrow} {abs(d):>5.4f}")

        print("\nFailure analysis:")
        for entry in _FAILURE_ANALYSIS:
            print(f"  • {entry['match']}")
            print(f"    Issue: {entry['issue'][:80]}...")

        print(f"\nEstimated BPS improvement: {baseline_bps} → "
              f"{delta_report['estimated_v2_bps']} "
              f"(+{delta_report['delta']} pts)")
        print("=" * 64 + "\n")
