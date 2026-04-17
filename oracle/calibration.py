"""
oracle/calibration.py — Probability calibration analysis for tournament predictions.

Provides:
  - 10-bin reliability diagram data
  - ECE (Expected Calibration Error)
  - MCE (Max Calibration Error)
  - Isotonic regression post-hoc calibration
  - CalibrationAnalyzer.plot_calibration_data() → dict for charting

A well-calibrated model produces probabilities that match empirical frequencies:
when it says "30% chance of winning", roughly 30% of such events should occur.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBin:
    """One bin of the reliability diagram."""
    lower: float
    upper: float
    mean_predicted_prob: float
    empirical_frequency: float
    count: int

    @property
    def calibration_error(self) -> float:
        return abs(self.empirical_frequency - self.mean_predicted_prob)


@dataclass
class CalibrationResult:
    bins: list[CalibrationBin]
    ece: float          # Expected Calibration Error
    mce: float          # Max Calibration Error
    overconfidence: float   # mean(predicted - actual) when predicted > 0.5
    underconfidence: float  # mean(actual - predicted) when predicted < 0.5
    n_samples: int


# ---------------------------------------------------------------------------
# CalibrationAnalyzer
# ---------------------------------------------------------------------------

class CalibrationAnalyzer:
    """
    Analyse and correct probability calibration for tournament predictions.

    Parameters
    ----------
    n_bins:
        Number of bins for reliability diagram. Default 10.

    Usage
    -----
    >>> analyzer = CalibrationAnalyzer()
    >>> y_pred = [0.7, 0.3, 0.8, 0.5, ...]   # model probabilities
    >>> y_true = [1,   0,   1,   0, ...]        # actual outcomes (1=correct)
    >>> result = analyzer.compute(y_pred, y_true)
    >>> print(f"ECE: {result.ece:.4f}")
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self._isotonic_params: Optional[dict] = None

    # ------------------------------------------------------------------
    # Core calibration computation
    # ------------------------------------------------------------------

    def compute(
        self,
        y_pred: list[float],
        y_true: list[int],
    ) -> CalibrationResult:
        """
        Compute reliability diagram data and calibration metrics.

        Parameters
        ----------
        y_pred:
            Model-predicted probabilities in [0, 1].
        y_true:
            Binary actual outcomes (1 = event occurred, 0 = did not).

        Returns
        -------
        CalibrationResult
        """
        y_pred_arr = np.array(y_pred, dtype=float)
        y_true_arr = np.array(y_true, dtype=float)

        if len(y_pred_arr) != len(y_true_arr):
            raise ValueError("y_pred and y_true must have the same length")

        n = len(y_pred_arr)
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bins: list[CalibrationBin] = []

        for i in range(self.n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (y_pred_arr >= lo) & (y_pred_arr < hi)
            if i == self.n_bins - 1:
                mask = mask | (y_pred_arr == 1.0)

            count = int(mask.sum())
            if count == 0:
                bins.append(CalibrationBin(
                    lower=lo, upper=hi,
                    mean_predicted_prob=(lo + hi) / 2,
                    empirical_frequency=0.0,
                    count=0,
                ))
                continue

            mean_pred = float(y_pred_arr[mask].mean())
            emp_freq  = float(y_true_arr[mask].mean())
            bins.append(CalibrationBin(
                lower=lo, upper=hi,
                mean_predicted_prob=mean_pred,
                empirical_frequency=emp_freq,
                count=count,
            ))

        # ECE: weighted average calibration error
        ece = sum(
            (b.count / n) * b.calibration_error
            for b in bins
        )

        # MCE: maximum calibration error across bins
        mce = max((b.calibration_error for b in bins if b.count > 0), default=0.0)

        # Overconfidence / underconfidence
        high_mask = y_pred_arr > 0.5
        low_mask  = y_pred_arr < 0.5
        overconf  = float((y_pred_arr[high_mask] - y_true_arr[high_mask]).mean()) \
                    if high_mask.any() else 0.0
        underconf = float((y_true_arr[low_mask]  - y_pred_arr[low_mask]).mean()) \
                    if low_mask.any() else 0.0

        return CalibrationResult(
            bins=bins,
            ece=round(ece, 6),
            mce=round(mce, 6),
            overconfidence=round(overconf, 6),
            underconfidence=round(underconf, 6),
            n_samples=n,
        )

    # ------------------------------------------------------------------
    # Isotonic regression calibration
    # ------------------------------------------------------------------

    def fit_isotonic(
        self,
        y_pred: list[float],
        y_true: list[int],
    ) -> None:
        """
        Fit isotonic regression for post-hoc probability calibration.

        After fitting, use calibrate() to transform raw probabilities.
        """
        from scipy.interpolate import PchipInterpolator  # type: ignore

        y_pred_arr = np.array(y_pred, dtype=float)
        y_true_arr = np.array(y_true, dtype=float)

        # Simple pool-adjacent-violators (PAV) algorithm
        n = len(y_pred_arr)
        sort_idx = np.argsort(y_pred_arr)
        y_s = y_true_arr[sort_idx]
        x_s = y_pred_arr[sort_idx]

        # PAV
        calibrated = y_s.copy()
        change = True
        while change:
            change = False
            i = 0
            while i < n - 1:
                if calibrated[i] > calibrated[i + 1]:
                    # Pool
                    pool_mean = calibrated[i:i + 2].mean()
                    calibrated[i:i + 2] = pool_mean
                    change = True
                i += 1

        self._isotonic_params = {
            "x": x_s,
            "y": calibrated,
        }
        logger.debug("Isotonic regression fitted on %d samples", n)

    def calibrate(self, raw_prob: float) -> float:
        """
        Apply fitted isotonic calibration to a raw probability.

        Raises RuntimeError if fit_isotonic() has not been called.
        """
        if self._isotonic_params is None:
            raise RuntimeError("Call fit_isotonic() before calibrate()")

        x = self._isotonic_params["x"]
        y = self._isotonic_params["y"]

        # Linear interpolation
        return float(np.interp(raw_prob, x, y))

    # ------------------------------------------------------------------
    # Chart data export
    # ------------------------------------------------------------------

    def plot_calibration_data(
        self,
        y_pred: list[float],
        y_true: list[int],
    ) -> dict:
        """
        Compute all data needed to render a reliability (calibration) diagram.

        Returns
        -------
        dict with keys:
          bins          — list of bin dicts (lower, upper, pred, actual, count)
          ece           — float
          mce           — float
          perfect_line  — [[0,0],[1,1]]
          overconfidence, underconfidence
          n_samples
        """
        result = self.compute(y_pred, y_true)

        bin_data = [
            {
                "lower":           b.lower,
                "upper":           b.upper,
                "mean_pred":       round(b.mean_predicted_prob, 4),
                "empirical_freq":  round(b.empirical_frequency, 4),
                "count":           b.count,
                "error":           round(b.calibration_error, 4),
            }
            for b in result.bins
        ]

        return {
            "bins":            bin_data,
            "ece":             result.ece,
            "mce":             result.mce,
            "overconfidence":  result.overconfidence,
            "underconfidence": result.underconfidence,
            "perfect_line":    [[0.0, 0.0], [1.0, 1.0]],
            "n_samples":       result.n_samples,
        }

    # ------------------------------------------------------------------
    # Synthetic validation helper
    # ------------------------------------------------------------------

    @staticmethod
    def synthetic_wc_calibration_data() -> tuple[list[float], list[int]]:
        """
        Generate synthetic calibration data based on historical WC predictions.

        Returns predicted probabilities and binary outcomes for ~256 match
        predictions across 2010–2022 World Cups (approximated).
        """
        rng = np.random.default_rng(42)

        # Simulate predictions: model predicts p, actual outcome has slight noise
        probs = rng.uniform(0.05, 0.95, size=256)
        # Outcomes: 1 if event occurred; slightly miscalibrated model (overconfident)
        overconf_factor = 0.85
        outcomes = (rng.random(size=256) < probs * overconf_factor).astype(int)

        return list(probs.round(3)), list(outcomes)
