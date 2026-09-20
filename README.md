# World Cup Oracle: v2 Experiment

[Algorithm guide: pseudocode, time complexity, and memory](docs/ALGORITHM_GUIDE.md).

This repository proposes dimension reweighting after an unsuccessful baseline.
Its scored 2022 replay still uses the same fixed team-strength table as v1:
the proposed reweighting is not connected to that evaluation.

## Verified results

| Replay | Simulations | Seed | BPS | Historical threshold | Predicted champion |
|---|---:|---:|---:|---|---|
| 2022 | 50,000 | 42 | 40/64 | Below 45/64 | France |
| 2018 | 50,000 | 2018 | 25/64 | Below 45/64 | Germany |

The 2022 stage points are 12 + 12 + 6 + 10 + 0 = 40.
The 2018 stage points are 14 + 8 + 3 + 0 + 0 = 25.
These are historical model replays, not held-out forecasting accuracy.

## Acknowledging my mistake

I previously published a 48/64 PASS claim, a separate table summing to 57,
and a hardcoded `~48 PASS` report without verifying them against executable
results. Those claims are withdrawn. The actual 2022 result is 40/64, not
48 or 57. The report now computes the 2022 row rather than printing a placeholder.

The proposed outcome table is removed from the current README because
hand-derived targets should not sit beside actual results as if comparable.
The earlier claims remain visible in Git history. Reweighting requires
period-appropriate features and a leakage-controlled evaluation before any
improvement can be claimed.

## Run

```bash
git clone https://github.com/fatehaszaman/world-cup-oracle-v2.git
cd world-cup-oracle-v2
python -m pip install -r requirements.txt
python -m backtest.wc2022_backtest
python -m backtest.wc2018_backtest
```

The full suite is not green. The 2026-09-19 follow-up on Python 3.14 recorded
34 passes and 8 failures using `python -m pytest -q`, with no collection
errors or skipped tests. Imports, compilation, and all three script smoke
checks pass. Remaining failures concern missing Qatar scorer inputs,
Brazil's historical rank, and four unchanged probability hypotheses.

Shared generic-engine repairs now complete the knockout, report reached
rounds correctly, and preserve referee draw mass. Examples compute output
rather than print fabricated forecasts; the benchmark uses the real API and
correct memory units. The generic 48-team scenario advances only 24 teams
with byes, not an official 2026 bracket. These repairs do not change the
separate historical replay results above.

## Current development

The latest bracket/market/xG evaluation and optional referee-dampening work
is in [world-cup-oracle-trials](https://github.com/fatehaszaman/world-cup-oracle-trials).
Its [audit](https://github.com/fatehaszaman/world-cup-oracle-trials/blob/main/AUDIT.md)
documents the current results, hindsight limitations, xG tie correction,
and separate simulation paths. v1/v2 remain baselines, not silently upgraded
copies of trials.

Maintained by [fatehaszaman](https://github.com/fatehaszaman).
