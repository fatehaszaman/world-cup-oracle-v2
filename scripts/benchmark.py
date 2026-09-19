"""Measure the generic engine locally; runtime is not a predictive quality test."""
from pathlib import Path
import argparse
import sys
import time
import tracemalloc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.team_strength import TeamStrengthScorer
from oracle.monte_carlo import TournamentSimulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Optional local performance threshold; no universal SLA.")
    args = parser.parse_args()
    if args.simulations <= 0:
        parser.error("--simulations must be positive")
    scores = TeamStrengthScorer().score_all_teams()
    tracemalloc.start()
    start = time.perf_counter()
    results = TournamentSimulator().run_tournament(scores, n_runs=args.simulations)
    elapsed = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Simulations: {args.simulations}; elapsed: {elapsed:.3f}s")
    print(f"Runs/second: {args.simulations / elapsed:.1f}")
    print(f"Peak traced Python allocations: {peak_bytes / 1024**2:.2f} MiB (not full process RSS)")
    print(results[["team", "champion_prob"]].head().to_string(index=False))
    if args.max_seconds is not None and elapsed > args.max_seconds:
        raise SystemExit(f"Local runtime threshold exceeded: {elapsed:.3f}s > {args.max_seconds}s")


if __name__ == "__main__":
    main()
