"""Run the generic static scenario; no hardcoded forecast or invented interval."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.team_strength import TeamStrengthScorer
from oracle.monte_carlo import TournamentSimulator
from oracle.bracket import WC2026_GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=1000)
    args = parser.parse_args()
    if args.simulations <= 0:
        parser.error("--simulations must be positive")
    scores = TeamStrengthScorer().score_all_teams()
    teams = {t for g in WC2026_GROUPS.values() for t in g}
    print("STATIC GENERIC SCENARIO: not a live or validated forecast.")
    print("Legacy format: top two per group, 24-team knockout with byes.")
    print("Missing team strengths use engine defaults:", ", ".join(sorted(teams - scores.keys())))
    print(f"Actual tournament simulations: {args.simulations}")
    print(TournamentSimulator().run_tournament(scores, n_runs=args.simulations).to_string(index=False))


if __name__ == "__main__":
    main()
