"""
examples/run_backtest.py — 2022 World Cup backtest demo.

Runs the full WC2022 backtest, prints the validation report, and if BPS < 45
runs model_diff to propose weight improvements.

Usage:
    python examples/run_backtest.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.wc2022_backtest import WC2022Backtest
from backtest.model_diff import ModelDiff


def main() -> None:
    print("Running 2022 World Cup backtest (50,000 simulations)…")
    bt = WC2022Backtest(n_simulations=50_000)
    bt.run()
    bt.print_validation_report()

    bps = bt.bracket_progression_score()
    total_pts = bps["total"]["pts"]
    passed    = bps["pass"]

    if not passed:
        print(f"\nBPS {total_pts}/64 is below threshold (45). Running model_diff…\n")
        diff = ModelDiff(bps_result=bps)
        diff.analyze()
        v2 = diff.generate_v2_config()
        diff.print_diff_report()

        print("Re-running backtest with v2 weights…")
        # For demonstration, we show the v2 config — a full re-run would
        # require injecting the new weights into TeamStrengthScorer.
        print("v2 weights:", v2)
        print("(Full v2 re-run not shown in demo mode)")
    else:
        print(f"BPS {total_pts}/64 — PASS. Model meets validation threshold.")


if __name__ == "__main__":
    main()
