"""
oracle/hyperparameter_tuner.py — Grid-search weight optimiser.

Searches over all valid dimension weight combinations (step=0.05, sum=1.0
constraint) to minimise Brier score on the 2022 World Cup backtest.

Results are cached to .cache/hyperparameter_search.json to avoid re-running
expensive searches on every startup.

Usage:
    from oracle.hyperparameter_tuner import HyperparameterTuner
    tuner = HyperparameterTuner()
    best_weights = tuner.optimize_weights()
    report_df = tuner.sensitivity_report()
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "hyperparameter_search.json"

DIMENSIONS = ["squad_value", "positional_power", "country_resources",
              "historical", "commercial"]

# Baseline 2022 team strengths used in Brier score computation
_TEAM_STRENGTHS_2022: dict[str, float] = {
    "Argentina":    0.87, "France":       0.88, "Brazil":       0.86,
    "England":      0.82, "Spain":        0.80, "Germany":      0.79,
    "Portugal":     0.81, "Netherlands":  0.77, "Belgium":      0.78,
    "Croatia":      0.73, "Denmark":      0.71, "Switzerland":  0.68,
    "USA":          0.62, "Mexico":       0.64, "Uruguay":      0.66,
    "Poland":       0.65, "Japan":        0.63, "South Korea":  0.60,
    "Morocco":      0.61, "Senegal":      0.62, "Ecuador":      0.58,
    "Australia":    0.55, "Serbia":       0.58, "Canada":       0.53,
    "Cameroon":     0.52, "Ghana":        0.51, "Iran":         0.50,
    "Tunisia":      0.52, "Saudi Arabia": 0.49, "Wales":        0.56,
    "Qatar":        0.38, "Costa Rica":   0.44,
}

# 2022 group stage outcomes encoded as (winner, loser, winner_strength - loser_strength)
# 1 = favourite won, 0 = upset
_WC2022_GROUP_OUTCOMES: list[tuple[str, str, int]] = [
    ("Netherlands", "Qatar",        1),
    ("England",     "Iran",         1),
    ("Argentina",   "Saudi Arabia", 0),  # upset
    ("France",      "Australia",    1),
    ("Japan",       "Germany",      0),  # upset
    ("Morocco",     "Croatia",      1),  # draw/Morocco edge
    ("Brazil",      "Serbia",       1),
    ("Portugal",    "Ghana",        1),
    ("Netherlands", "Ecuador",      1),
    ("England",     "USA",          1),  # draw, England favoured
    ("France",      "Denmark",      1),
    ("Japan",       "Spain",        0),  # upset
    ("Morocco",     "Belgium",      0),  # upset
    ("Brazil",      "Switzerland",  1),
    ("Portugal",    "Uruguay",      1),
]

# R16 outcomes
_WC2022_KO_OUTCOMES: list[tuple[str, str, int]] = [
    ("Netherlands", "USA",         1),
    ("Argentina",   "Australia",   1),
    ("France",      "Poland",      1),
    ("England",     "Senegal",     1),
    ("Croatia",     "Japan",       0),   # upset (pens)
    ("Brazil",      "South Korea", 1),
    ("Morocco",     "Spain",       0),   # upset (pens)
    ("Portugal",    "Switzerland", 1),
    # QF
    ("Croatia",     "Brazil",      0),   # upset
    ("Argentina",   "Netherlands", 1),
    ("Morocco",     "Portugal",    0),   # upset
    ("France",      "England",     1),
    # SF
    ("Argentina",   "Croatia",     1),
    ("France",      "Morocco",     1),
    # Final
    ("Argentina",   "France",      1),   # Argentina won
]

ALL_OUTCOMES = _WC2022_GROUP_OUTCOMES + _WC2022_KO_OUTCOMES


# ---------------------------------------------------------------------------
# Brier score computation
# ---------------------------------------------------------------------------

def _match_win_prob(team_a: str, team_b: str, weights: dict[str, float]) -> float:
    """
    Compute P(team_a wins) given dimension weights.
    Simplified model: weighted strength ratio with noise component.
    """
    # Under this simplified model each dimension contributes equally to strength
    # (since we only have one composite score in backtest mode)
    sa = _TEAM_STRENGTHS_2022.get(team_a, 0.5)
    sb = _TEAM_STRENGTHS_2022.get(team_b, 0.5)

    # Apply weight sensitivity: higher squad_value weight → stronger teams dominate more
    amplify = 1.0 + (weights.get("squad_value", 0.3) - 0.3) * 2.0
    sa_adj = sa ** amplify
    sb_adj = sb ** amplify
    total = sa_adj + sb_adj
    return sa_adj / total if total > 0 else 0.5


def _compute_brier_score(weights: dict[str, float]) -> float:
    """Compute mean Brier score across all 2022 outcomes under given weights."""
    total_bs = 0.0
    for team_a, team_b, actual in ALL_OUTCOMES:
        p = _match_win_prob(team_a, team_b, weights)
        total_bs += (p - actual) ** 2
    return total_bs / len(ALL_OUTCOMES)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def _generate_weight_combinations(step: float = 0.05) -> list[dict[str, float]]:
    """
    Generate all 5-dimension weight combinations that sum to 1.0.

    With step=0.05 this produces C(24,4) = 10,626 candidates approximately.
    """
    values = [round(v, 2) for v in np.arange(step, 1.0, step)]
    combos = []

    for combo in itertools.combinations_with_replacement(values, len(DIMENSIONS)):
        if abs(sum(combo) - 1.0) < 1e-6:
            combos.append(dict(zip(DIMENSIONS, combo)))

    # Also generate by random resampling for coverage
    rng = np.random.default_rng(42)
    for _ in range(500):
        raw = rng.dirichlet(np.ones(len(DIMENSIONS)))
        quantised = np.round(raw / step) * step
        if quantised.sum() > 0:
            quantised = quantised / quantised.sum()
        combos.append(dict(zip(DIMENSIONS, quantised.tolist())))

    return combos


# ---------------------------------------------------------------------------
# HyperparameterTuner
# ---------------------------------------------------------------------------

class HyperparameterTuner:
    """
    Grid-search optimiser for oracle dimension weights.

    Minimises Brier score on the 2022 World Cup backtest data.
    Results are cached to avoid redundant computation.
    """

    def __init__(self) -> None:
        self._results: Optional[list[dict]] = None
        self._best: Optional[dict[str, float]] = None

    def _load_cache(self) -> bool:
        """Load previously computed grid search results. Returns True if loaded."""
        if not CACHE_FILE.exists():
            return False
        try:
            data = json.loads(CACHE_FILE.read_text())
            self._results = data.get("results")
            self._best    = data.get("best")
            logger.info("Loaded hyperparameter search cache (%d results)", len(self._results or []))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache load failed: %s", exc)
            return False

    def _save_cache(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        payload = {
            "results":    self._results,
            "best":       self._best,
            "computed_at": time.time(),
        }
        CACHE_FILE.write_text(json.dumps(payload, indent=2))

    def optimize_weights(
        self,
        step: float = 0.05,
        force_recompute: bool = False,
    ) -> dict[str, float]:
        """
        Run grid search to find dimension weights minimising Brier score.

        Parameters
        ----------
        step:
            Grid step size for weight values. Default 0.05.
        force_recompute:
            Skip cache and recompute from scratch.

        Returns
        -------
        dict mapping dimension → optimal weight
        """
        if not force_recompute and self._load_cache():
            return self._best  # type: ignore[return-value]

        logger.info("Running hyperparameter grid search (step=%.2f)…", step)
        combos = _generate_weight_combinations(step)
        logger.info("Evaluating %d weight combinations", len(combos))

        results = []
        for weights in combos:
            bs = _compute_brier_score(weights)
            results.append({"weights": weights, "brier_score": round(bs, 6)})

        results.sort(key=lambda x: x["brier_score"])
        self._results = results
        self._best    = results[0]["weights"]
        self._save_cache()

        logger.info(
            "Best weights: %s  →  Brier score %.4f",
            self._best, results[0]["brier_score"]
        )
        return self._best  # type: ignore[return-value]

    def sensitivity_report(self) -> pd.DataFrame:
        """
        Compute sensitivity of Brier score to each dimension weight.

        Returns
        -------
        pd.DataFrame with columns: dimension, weight_range, brier_mean,
        brier_std, sensitivity (higher = more influential)
        """
        if self._results is None:
            self.optimize_weights()

        records = []
        for dim in DIMENSIONS:
            dim_vals = [r["weights"][dim] for r in self._results]  # type: ignore
            bs_vals  = [r["brier_score"]  for r in self._results]  # type: ignore

            bins = np.linspace(0, 1, 6)
            dim_arr = np.array(dim_vals)
            bs_arr  = np.array(bs_vals)

            bin_means = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (dim_arr >= lo) & (dim_arr < hi)
                if mask.sum() > 0:
                    bin_means.append(bs_arr[mask].mean())

            sensitivity = (max(bin_means) - min(bin_means)) if len(bin_means) > 1 else 0.0

            records.append({
                "dimension":    dim,
                "weight_range": f"{dim_arr.min():.2f}–{dim_arr.max():.2f}",
                "brier_mean":   round(float(bs_arr.mean()), 4),
                "brier_std":    round(float(bs_arr.std()), 4),
                "sensitivity":  round(sensitivity, 4),
            })

        df = pd.DataFrame(records).sort_values("sensitivity", ascending=False)
        return df.reset_index(drop=True)
