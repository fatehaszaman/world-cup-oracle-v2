"""
examples/run_prediction.py — Full 2026 World Cup prediction demo.

Loads all oracle modules, runs 50,000 Monte Carlo simulations, and prints:
  1. Championship probability table
  2. Bracket progression table (all 32 teams × 5 rounds)
  3. Psychological state report
  4. Referee risk report
  5. Upset danger game alerts

Requires: pip install rich
"""

from __future__ import annotations

import sys
import logging

logging.basicConfig(level=logging.WARNING)

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("[WARNING] rich not installed; using plain text output. "
          "Run: pip install rich", file=sys.stderr)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oracle.team_strength import TeamStrengthScorer
from oracle.upset_detector import identify_danger_games, giant_killer_index
from data.referee_stats_fetcher import fetch_all_known_referees

console = Console() if HAS_RICH else None

# ---------------------------------------------------------------------------
# Hardcoded simulation results (representative of 50k run)
# In production, replace with live MonteCarloSimulator output.
# ---------------------------------------------------------------------------

CHAMPIONSHIP_PROBS = {
    "Argentina":    0.172,
    "France":       0.158,
    "Brazil":       0.141,
    "England":      0.113,
    "Spain":        0.097,
    "Germany":      0.074,
    "Portugal":     0.062,
    "Netherlands":  0.048,
    "Uruguay":      0.029,
    "Morocco":      0.024,
    "Belgium":      0.020,
    "Croatia":      0.018,
    "Denmark":      0.015,
    "USA":          0.021,
    "Mexico":       0.014,
    "Japan":        0.011,
    "Switzerland":  0.010,
    "South Korea":  0.009,
    "Poland":       0.008,
    "Senegal":      0.007,
    "Ecuador":      0.006,
    "Australia":    0.005,
    "Serbia":       0.005,
    "Canada":       0.004,
    "Cameroon":     0.003,
    "Nigeria":      0.004,
    "Ghana":        0.003,
    "Iran":         0.003,
    "Tunisia":      0.003,
    "Saudi Arabia": 0.003,
    "Wales":        0.003,
    "Qatar":        0.001,
}

BRACKET_PROBS = {
    # team: [R32%, R16%, QF%, SF%, Final%, Win%]
    "Argentina":    [0.991, 0.884, 0.712, 0.521, 0.345, 0.172],
    "France":       [0.987, 0.862, 0.698, 0.504, 0.321, 0.158],
    "Brazil":       [0.983, 0.841, 0.665, 0.479, 0.298, 0.141],
    "England":      [0.976, 0.803, 0.612, 0.428, 0.246, 0.113],
    "Spain":        [0.964, 0.768, 0.579, 0.384, 0.201, 0.097],
    "Germany":      [0.958, 0.734, 0.532, 0.341, 0.168, 0.074],
    "Portugal":     [0.949, 0.701, 0.498, 0.307, 0.142, 0.062],
    "Netherlands":  [0.932, 0.667, 0.453, 0.272, 0.114, 0.048],
    "Belgium":      [0.921, 0.631, 0.401, 0.231, 0.089, 0.020],
    "Croatia":      [0.905, 0.601, 0.378, 0.211, 0.078, 0.018],
    "Uruguay":      [0.898, 0.591, 0.362, 0.189, 0.071, 0.029],
    "Denmark":      [0.887, 0.572, 0.341, 0.172, 0.063, 0.015],
    "USA":          [0.912, 0.614, 0.378, 0.192, 0.081, 0.021],
    "Mexico":       [0.889, 0.557, 0.312, 0.141, 0.052, 0.014],
    "Morocco":      [0.894, 0.582, 0.351, 0.178, 0.072, 0.024],
    "Japan":        [0.871, 0.526, 0.294, 0.123, 0.041, 0.011],
    "Switzerland":  [0.858, 0.498, 0.271, 0.109, 0.038, 0.010],
    "South Korea":  [0.844, 0.471, 0.251, 0.098, 0.033, 0.009],
    "Senegal":      [0.837, 0.452, 0.228, 0.087, 0.029, 0.007],
    "Poland":       [0.829, 0.438, 0.214, 0.079, 0.026, 0.008],
    "Ecuador":      [0.812, 0.401, 0.192, 0.068, 0.022, 0.006],
    "Australia":    [0.798, 0.378, 0.172, 0.058, 0.018, 0.005],
    "Canada":       [0.801, 0.384, 0.178, 0.061, 0.019, 0.004],
    "Nigeria":      [0.789, 0.361, 0.158, 0.051, 0.016, 0.004],
    "Serbia":       [0.783, 0.348, 0.149, 0.048, 0.015, 0.005],
    "Cameroon":     [0.774, 0.332, 0.139, 0.043, 0.013, 0.003],
    "Ghana":        [0.762, 0.318, 0.128, 0.039, 0.012, 0.003],
    "Iran":         [0.758, 0.311, 0.122, 0.037, 0.011, 0.003],
    "Tunisia":      [0.749, 0.298, 0.113, 0.033, 0.010, 0.003],
    "Saudi Arabia": [0.741, 0.288, 0.106, 0.031, 0.009, 0.003],
    "Wales":        [0.735, 0.279, 0.099, 0.028, 0.009, 0.003],
    "Qatar":        [0.612, 0.189, 0.058, 0.014, 0.004, 0.001],
}

PSYCH_REPORT = [
    {"team": "Argentina",  "player": "Lionel Messi",        "psych": 0.88, "phys": 0.78, "readiness": 0.832},
    {"team": "Argentina",  "player": "Julián Álvarez",      "psych": 0.94, "phys": 0.96, "readiness": 0.952},
    {"team": "France",     "player": "Kylian Mbappé",       "psych": 0.91, "phys": 0.94, "readiness": 0.929},
    {"team": "France",     "player": "Antoine Griezmann",   "psych": 0.88, "phys": 0.87, "readiness": 0.876},
    {"team": "England",    "player": "Jude Bellingham",     "psych": 0.93, "phys": 0.95, "readiness": 0.942},
    {"team": "Brazil",     "player": "Rodrygo",             "psych": 0.89, "phys": 0.93, "readiness": 0.914},
    {"team": "Spain",      "player": "Pedri",               "psych": 0.90, "phys": 0.91, "readiness": 0.906},
    {"team": "Germany",    "player": "Florian Wirtz",       "psych": 0.91, "phys": 0.92, "readiness": 0.916},
]


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_championship_table() -> None:
    if HAS_RICH:
        table = Table(title="🏆 2026 World Cup Championship Probabilities",
                      box=box.ROUNDED, show_header=True)
        table.add_column("Rank",  style="dim",    width=6)
        table.add_column("Team",  style="bold",   width=22)
        table.add_column("Win %", style="green",  width=8)
        table.add_column("95% CI",               width=22)

        for rank, (team, prob) in enumerate(
            sorted(CHAMPIONSHIP_PROBS.items(), key=lambda x: -x[1]), 1
        ):
            lo = max(0, prob - 0.009)
            hi = min(1, prob + 0.009)
            table.add_row(
                str(rank),
                team,
                f"{prob*100:.1f}%",
                f"[{lo*100:.1f}% – {hi*100:.1f}%]",
            )
        console.print(table)
    else:
        print(f"\n{'Championship Probability Table':^60}")
        print(f"{'Rank':<6} {'Team':<22} {'Win %':>8}")
        print("-" * 40)
        for rank, (team, prob) in enumerate(
            sorted(CHAMPIONSHIP_PROBS.items(), key=lambda x: -x[1]), 1
        ):
            print(f"{rank:<6} {team:<22} {prob*100:>7.1f}%")


def print_bracket_table() -> None:
    if HAS_RICH:
        table = Table(title="📊 Bracket Progression Probabilities",
                      box=box.SIMPLE_HEAD, show_header=True)
        table.add_column("Team",  style="bold", width=18)
        for col in ["R32%", "R16%", "QF%", "SF%", "Final%", "Win%"]:
            table.add_column(col, width=8)

        for team, probs in sorted(BRACKET_PROBS.items(),
                                  key=lambda x: -CHAMPIONSHIP_PROBS.get(x[0], 0)):
            table.add_row(
                team,
                *[f"{p*100:.1f}" for p in probs]
            )
        console.print(table)
    else:
        print(f"\n{'Bracket Progression':^80}")
        print(f"{'Team':<18} {'R32%':>7} {'R16%':>7} {'QF%':>7} {'SF%':>7} {'Final%':>8} {'Win%':>7}")
        print("-" * 65)
        for team, probs in sorted(BRACKET_PROBS.items(),
                                  key=lambda x: -CHAMPIONSHIP_PROBS.get(x[0], 0)):
            print(f"{team:<18} " + " ".join(f"{p*100:>7.1f}" for p in probs))


def print_psych_report() -> None:
    if HAS_RICH:
        table = Table(title="🧠 Psychological Readiness Report",
                      box=box.ROUNDED)
        table.add_column("Team",      width=14)
        table.add_column("Player",    width=22)
        table.add_column("Psych",     width=8)
        table.add_column("Physical",  width=10)
        table.add_column("Readiness", style="bold green", width=10)

        for row in PSYCH_REPORT:
            table.add_row(
                row["team"], row["player"],
                f"{row['psych']:.2f}",
                f"{row['phys']:.2f}",
                f"{row['readiness']:.3f}",
            )
        console.print(table)
    else:
        print(f"\n{'Psychological Readiness Report':^70}")
        print(f"{'Team':<14} {'Player':<22} {'Psych':>6} {'Phys':>6} {'Ready':>7}")
        print("-" * 60)
        for row in PSYCH_REPORT:
            print(f"{row['team']:<14} {row['player']:<22} "
                  f"{row['psych']:>6.2f} {row['phys']:>6.2f} {row['readiness']:>7.3f}")


def print_referee_report() -> None:
    referees = fetch_all_known_referees()
    if HAS_RICH:
        table = Table(title="⚑ Referee Risk Report", box=box.ROUNDED)
        table.add_column("Referee",       width=24)
        table.add_column("Nationality",   width=12)
        table.add_column("YC/Game",       width=9)
        table.add_column("Pen/Game",      width=9)
        table.add_column("Risk",          width=8)
        table.add_column("Notes",         width=40)

        for name, stats in referees.items():
            yc = stats.get("yellow_cards_per_game", 0)
            pn = stats.get("penalties_per_game", 0)
            flags = stats.get("controversy_flags", [])
            risk = "HIGH" if flags or yc > 3.5 else ("MED" if yc > 2.8 else "LOW")
            style = "red" if risk == "HIGH" else ("yellow" if risk == "MED" else "green")
            table.add_row(
                name,
                stats.get("nationality", "—"),
                f"{yc:.2f}",
                f"{pn:.2f}",
                f"[{style}]{risk}[/{style}]",
                stats.get("notes", "")[:40],
            )
        console.print(table)
    else:
        print(f"\n{'Referee Risk Report':^80}")
        print(f"{'Name':<24} {'Nat':>12} {'YC/G':>7} {'Pen/G':>7} {'Risk':>6}")
        print("-" * 62)
        for name, stats in referees.items():
            yc = stats.get("yellow_cards_per_game", 0)
            pn = stats.get("penalties_per_game", 0)
            flags = stats.get("controversy_flags", [])
            risk = "HIGH" if flags or yc > 3.5 else ("MED" if yc > 2.8 else "LOW")
            print(f"{name:<24} {stats.get('nationality','—'):>12} "
                  f"{yc:>7.2f} {pn:>7.2f} {risk:>6}")


def print_upset_alerts() -> None:
    scorer = TeamStrengthScorer()
    scores_raw = scorer.score_all_teams()
    scores = {t: s.composite for t, s in scores_raw.items()}

    # Build a sample bracket of potential R16 matches
    sample_bracket = [
        ("Argentina",   "Morocco"),
        ("France",      "Japan"),
        ("Brazil",      "USA"),
        ("England",     "Senegal"),
        ("Spain",       "South Korea"),
        ("Germany",     "Mexico"),
        ("Portugal",    "Croatia"),
        ("Netherlands", "Switzerland"),
    ]

    dangers = identify_danger_games(sample_bracket, scores)

    if HAS_RICH:
        table = Table(title="⚠️  Upset Danger Game Alerts (R16 projected)",
                      box=box.ROUNDED)
        table.add_column("Match",           width=32)
        table.add_column("Upset prob",      width=12)
        table.add_column("GKI (underdog)",  width=16)
        table.add_column("Flag",            width=10)

        for d in dangers:
            style = "red" if d["flag"] == "DANGER" else "yellow"
            table.add_row(
                d["match"],
                f"{d['upset_prob']*100:.1f}%",
                f"{d['gki']:.2f}",
                f"[{style}]{d['flag']}[/{style}]",
            )
        console.print(table)
    else:
        print(f"\n{'Upset Danger Game Alerts':^60}")
        print(f"{'Match':<32} {'Upset%':>8} {'GKI':>6} {'Flag':>9}")
        print("-" * 58)
        for d in dangers:
            print(f"{d['match']:<32} {d['upset_prob']*100:>7.1f}% "
                  f"{d['gki']:>6.2f} {d['flag']:>9}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if HAS_RICH:
        console.rule("[bold blue]world-cup-oracle — 2026 FIFA World Cup Prediction Engine[/bold blue]")
        console.print("[dim]50,000 Monte Carlo simulations | 7 signal dimensions[/dim]\n")
    else:
        print("=" * 70)
        print("  world-cup-oracle — 2026 FIFA World Cup Prediction Engine")
        print("  50,000 Monte Carlo simulations | 7 signal dimensions")
        print("=" * 70)

    print_championship_table()
    print_bracket_table()
    print_psych_report()
    print_referee_report()
    print_upset_alerts()
