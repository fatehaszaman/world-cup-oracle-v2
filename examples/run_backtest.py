"""Run a historical replay; the BPS threshold is not out-of-sample validation."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.wc2022_backtest import WC2022Backtest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.simulations <= 0:
        parser.error("--simulations must be positive")
    print("HISTORICAL REPLAY: in-sample scores are not predictive validation.")
    bt = WC2022Backtest(n_simulations=args.simulations, seed=args.seed)
    bt.run()
    bt.print_validation_report()
    print("No proposed weight change has been applied or revalidated by this example.")


if __name__ == "__main__":
    main()
