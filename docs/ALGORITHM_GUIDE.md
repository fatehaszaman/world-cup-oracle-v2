# Algorithm guide

These cards cover v2's generic simulator. They do not imply that proposed
reweighting was applied to every replay or that this experiment is validated.
Read the README's distinction between the experiment and scored backtests.

## Simulate a match

Implementation: [`TournamentSimulator.simulate_match`](../oracle/monte_carlo.py).

```text
# Match Monte Carlo / Vectorized Poisson Counts
# Input: team pair, score map, P samples
# Output: win/draw probabilities and mean goals
# Time: expected O(P), excluding optional referee work
# Memory: O(P) temporary arrays; O(1) returned summary

LOOK UP scores with missing-score fallback 0.50
CLIP strength difference; construct positive Poisson goal rates
DRAW two P-element goal arrays
COUNT wins and draws; divide by P
OPTIONALLY adjust decisive probabilities without deleting draw probability
RETURN rounded probabilities, goal means, and adjustment metadata
```

Costs assume bounded goal-rate parameters and fixed-width arithmetic.
P must be positive; the method does not turn missing scores into evidence.
Vectorization does not make P independent trials constant-time.

## Aggregate repeated tournaments

Implementation: [`TournamentSimulator.run_tournament`](../oracle/monte_carlo.py).

```text
# Tournament Summary / Reach Tensor
# Input: R runs, T teams, K=6 stage flags
# Output: team probability table
# Time: O(R*(C_run + T*K) + T*log T)
# Memory: O(R*T*K + T*K), plus simulator and one-run workspace
# C_run = the delegated one-tournament simulation cost

SORT unique teams; allocate reach[R, T, K] as float32
FOR each run:
    SIMULATE tournament with a reproducible child generator
    RECORD every team's stage flags
AVERAGE flags across runs
BUILD rows and SORT by champion probability
RETURN table
```

Retained indicator storage alone is `4*R*T*K` bytes. The configured-team
Cholesky matrix requires O(T_config²) persistent memory and O(T_config³)
factorization during initialization. The method named `memory_usage_mb`
counts only that factor, not the reach tensor or process memory.

The generic simulator remains a simplified tournament path with byes; it
is not the separate trials repository's official-format scenario engine.
Runs are sequential. Sample counts, elapsed-time logs, and complexity notes
do not establish forecasting skill or a runtime service-level guarantee.
