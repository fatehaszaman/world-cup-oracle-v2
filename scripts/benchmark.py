"""
scripts/benchmark.py — Performance benchmark for the oracle Monte Carlo engine.

Times a 50,000-simulation tournament run and reports:
  - Elapsed wall-clock time (seconds)
  - Simulations per second
  - Peak memory usage (MB)
  - PASS/FAIL against the 30-second SLA

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import sys
import os
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

N_SIMULATIONS = 50_000
MAX_SECONDS   = 30.0   # SLA


def run_benchmark() -> None:
    from oracle.team_strength import TeamStrengthScorer
    from oracle.monte_carlo import MonteCarloSimulator

    print(f"world-cup-oracle Benchmark")
    print(f"  Simulations: {N_SIMULATIONS:,}")
    print(f"  SLA target:  < {MAX_SECONDS:.0f} seconds")
    print()

    # --- Score teams ---
    scorer = TeamStrengthScorer()
    scores = scorer.score_all_teams()

    # --- Start tracking ---
    tracemalloc.start()
    t0 = time.perf_counter()

    sim = MonteCarloSimulator(team_scores=scores)
    results = sim.run_tournament(n_simulations=N_SIMULATIONS)

    elapsed = time.perf_counter() - t0
    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_kb / 1024.0
    sims_per_sec = N_SIMULATIONS / elapsed

    # --- Report ---
    print(f"Results:")
    print(f"  Elapsed time:      {elapsed:.2f}s")
    print(f"  Simulations/sec:   {sims_per_sec:,.0f}")
    print(f"  Peak memory:       {peak_mb:.1f} MB")
    print()

    status = "✓ PASS" if elapsed < MAX_SECONDS else "✗ FAIL"
    print(f"  30-second SLA:     {status}  ({elapsed:.2f}s / {MAX_SECONDS:.0f}s)")
    print()

    # Print top 5 predicted champions
    probs = results.champion_probs
    top5 = sorted(probs, key=lambda t: -probs.get(t, 0))[:5]
    print("Top 5 predicted champions:")
    for t in top5:
        print(f"  {t:<20s}  {probs[t]*100:.1f}%")

    # Assert SLA
    assert elapsed < MAX_SECONDS, (
        f"Performance SLA violated: {elapsed:.2f}s > {MAX_SECONDS:.0f}s. "
        f"Consider enabling multiprocessing or reducing simulation count."
    )


if __name__ == "__main__":
    run_benchmark()
