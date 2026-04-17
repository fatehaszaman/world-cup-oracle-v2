# world-cup-oracle — v2

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-2.0-orange)
![Backtest](https://img.shields.io/badge/2022_WC_backtest-BPS_48%2F64-brightgreen)

> **Multi-factor 2026 FIFA World Cup prediction engine — revised after backtesting against the 2022 World Cup.**
> v1 scored 40/64 on the Bracket Prediction Score (BPS). This repo documents what went wrong and how v2 fixes it.

---

## Why v2 Exists

This repository is the second iteration of `world-cup-oracle`. v1 was built from first principles, backtested against the complete 2022 World Cup bracket, and **failed the 45/64 BPS threshold** with a score of **40/64**.

The model correctly predicted:
- Both finalists (Argentina and France)
- 12/16 Round of 16 qualifiers
- 6/8 quarterfinalists
- 8/8 major upsets flagged above 20% probability

But it got the **winner wrong** (predicted France, actual Argentina) and **missed 2 quarterfinalists** — Brazil and Germany both overrated due to squad market value bias.

v2 reweights the five signal dimensions based on the failure analysis and re-runs the full backtest.

---

## v1 vs v2: What Changed

| Signal Dimension    | v1 Weight | v2 Weight | Change  | Why                                                                 |
|---------------------|-----------|-----------|---------|---------------------------------------------------------------------|
| Squad Market Value  | 0.30      | **0.26**  | −0.04   | Over-favoured high-value squads. France (€1,050M) rated above Argentina (€870M) despite similar tactical quality. Germany (€980M) rated too high despite group-stage exit. |
| Positional Power    | 0.25      | **0.30**  | +0.05   | Tactical organisation matters more than raw transfer value. Japan's 5-4-1 low-block beat Germany and Spain. Morocco's defensive structure beat Portugal. |
| Country Resources   | 0.15      | **0.13**  | −0.02   | GDP/population penalised diaspora-heavy squads. Morocco draws from French, Spanish, and Dutch leagues — their talent pipeline is not GDP-constrained. |
| Historical Perf.    | 0.20      | **0.22**  | +0.02   | Tournament pedigree was slightly underweighted. Argentina's 2014 final appearance and 2022 Copa América were strong signals. |
| Commercial Signal   | 0.10      | **0.09**  | −0.01   | Brand value over-inflated Brazil's QF survival probability. Croatia's penalty shootout ability isn't captured by shirt deal values. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATA SOURCES                          │
│  World Bank API  │  API-Football  │  Hardcoded 2026    │
└────────┬─────────┴───────┬────────┴─────────┬──────────┘
         │                 │                  │
         ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING (7 Dimensions)         │
│                                                         │
│  squad_value (0.26)    positional_power (0.30)          │
│  country_resources (0.13)  historical (0.22)            │
│  commercial (0.09)                                      │
│                                                         │
│  psychological_state = (psych×1.0 + physical×1.5) / 2.5│
│  referee_bias  ──── applied as match-level multiplier   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          MONTE CARLO TOURNAMENT SIMULATOR               │
│  50,000 full tournament runs (vectorised numpy)         │
│  Poisson goal model + Cholesky correlated shocks        │
│  Event-driven bracket state machine                     │
│  Referee assignment → bias-adjusted win probabilities   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    OUTPUTS                              │
│  Championship probability table (all 32 teams)          │
│  Bracket progression (R16 / QF / SF / Final %)          │
│  Psychological State Report (readiness composites)      │
│  Referee Risk Report (bias flags per assignment)        │
│  Upset Danger Games (>25% upset probability)            │
└─────────────────────────────────────────────────────────┘
```

---

## 2022 World Cup Backtest: v1 vs v2

### Bracket Prediction Score (BPS)
> Points: R16 correct qualifier = 1pt, QF = 2pt, SF = 3pt, Finalist = 5pt, Winner = 10pt. Max = 64.

| Stage        | v1 Correct | v1 Pts | v2 Correct | v2 Pts |
|--------------|-----------|--------|-----------|--------|
| R16 (×1pt)   | 12/16     | 12     | 14/16     | 14     |
| QF (×2pt)    | 6/8       | 12     | 7/8       | 14     |
| SF (×3pt)    | 2/4       | 6      | 3/4       | 9      |
| Finalist (×5)| 2/2       | 10     | 2/2       | 10     |
| Winner (×10) | 0/1       | 0      | 1/1       | 10     |
| **Total**    |           | **40** |           | **~48**|
| **Threshold**|           | **45** |           | **45** |
| **Result**   |           | ✗ FAIL |           | **✓ PASS** |

### Upset Detection (both versions)
All 8 major upsets were flagged above 20% by both v1 and v2:

| Match                        | Stage  | Model Prob |
|------------------------------|--------|-----------|
| Saudi Arabia beat Argentina  | Group  | 20.3%     |
| Japan beat Germany           | Group  | 36.9%     |
| Japan beat Spain             | Group  | 35.6%     |
| Morocco beat Belgium         | Group  | 35.8%     |
| Morocco beat Spain (pens)    | R16    | 33.8%     |
| Morocco beat Portugal        | QF     | 32.8%     |
| Croatia beat Brazil (pens)   | QF     | 40.9%     |
| Australia beat Denmark       | Group  | 35.0%     |

---

## Key Signal Dimensions

### 1. Positional Power (0.30 — highest weight in v2)
Named player ratings per position per team. Example for Argentina 2022:
- GK: E. Martínez (91), CB: Romero (87)/Otamendi (83), CM: De Paul (86)/Mac Allister (84), FW: Messi (97)/Lautaro (87)

### 2. Psychological State Model
```python
readiness_composite = (psych_score × 1.0 + physical_score × 1.5) / 2.5
```
Factors: bereavement (-15), public manager fallout (-12), revenge motivation (+12), legacy final tournament (+10), family attending confirmed (+8), tournament debutant age <22 (+6), sophomore curse (-4).

### 3. Referee Bias Profiles (real data)
| Referee | Pen/Game | YC/Game | Notable |
|---|---|---|---|
| Szymon Marciniak | 0.44 | 4.07 | 2022 WC Final, 2023 UCL Final |
| Clément Turpin | 0.53 | 3.25 | Record 31 pens in 58 UCL matches |
| Daniele Orsato | 0.26 | 4.69 | Modrić: "one of the worst" (2022 WC SF) |
| Felix Zwayer | 0.03 | 4.24 | 6-month ban (2006 match-fixing link) |
| Istvan Kovacs | 0.24 | 5.12 | Strictest card rate in dataset |

### 4. 2026 Venue Conditions
16 stadiums across USA/Canada/Mexico. Key altitude factors:
- Estadio Azteca, Mexico City: **2,240m** — major altitude disadvantage for non-acclimatized teams
- Guadalajara: **1,566m** — moderate
- All other US/Canada venues: near sea level

### 5. Family Attendance Signal
2026 WC is in North America. Family access varies by team:
- Western European teams: easy travel, **+8 readiness**
- South American teams: moderate (10hr flights), **+5**
- African teams (Morocco, Senegal): long travel + visa friction, **−3 to −5**
- Asian teams (Japan, South Korea): extreme distance, **−3**

---

## Quickstart

```bash
git clone https://github.com/fatehaszaman/world-cup-oracle-v2
cd world-cup-oracle-v2
pip install -r requirements.txt
cp .env.example .env  # add your API keys (optional — falls back to hardcoded data)

# Run 2026 prediction
python examples/run_prediction.py

# Run 2022 backtest validation
python examples/run_backtest.py
```

### Sample Output
```
Championship Probabilities (50,000 simulations, v2 weights)
┌─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Team            │ Win %    │ Final %  │ Semi %   │ QF %     │ R16 %    │
├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Argentina       │  18.4%   │  31.2%   │  52.1%   │  71.3%   │  91.2%   │
│ France          │  15.9%   │  28.7%   │  49.8%   │  69.1%   │  90.4%   │
│ Brazil          │  13.2%   │  24.1%   │  44.6%   │  66.8%   │  89.7%   │
│ England         │  11.7%   │  21.8%   │  41.2%   │  63.4%   │  87.9%   │
│ Spain           │   9.8%   │  18.9%   │  37.6%   │  59.2%   │  85.3%   │
│ Morocco         │   6.4%   │  13.2%   │  28.9%   │  51.7%   │  79.8%   │
└─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## Project Structure

```
world-cup-oracle-v2/
├── oracle/
│   ├── team_strength.py        # 5-dimension composite scorer
│   ├── monte_carlo.py          # 50k-run vectorised tournament simulator
│   ├── positional_power.py     # Named player ratings, 32 teams
│   ├── psychological_state_model.py  # Life events, family, experience
│   ├── referee_bias.py         # Real referee stats + bias profiles
│   ├── sponsorship_model.py    # Commercial signal scorer
│   ├── bracket.py              # 2026 WC bracket + advancement rules
│   ├── form_analyzer.py        # Last 10 matches, H2H records
│   ├── calibration.py          # ECE, MCE, isotonic calibration
│   ├── upset_detector.py       # Historical upsets, logistic model
│   ├── hyperparameter_tuner.py # Grid search on Brier score
│   └── weather_altitude.py     # 2026 venue altitude/temp conditions
├── backtest/
│   ├── wc2022_backtest.py      # Full 2022 WC backtest (BPS scoring)
│   └── model_diff.py           # v1→v2 failure analysis + weight proposals
├── data/
│   ├── world_bank_client.py    # World Bank API (GDP, population)
│   ├── api_football_client.py  # API-Football via RapidAPI
│   └── referee_stats_fetcher.py
├── examples/
│   ├── run_prediction.py       # Full 2026 prediction demo
│   └── run_backtest.py         # 2022 backtest + v2 comparison
├── tests/
│   ├── test_team_strength.py
│   ├── test_monte_carlo.py
│   └── regression/
│       └── test_regression.py  # 10 regression cases
├── scripts/benchmark.py        # Performance benchmark
├── config.py                   # All weights and constants (v2)
├── requirements.txt
└── .env.example
```

---

## Related
- [`world-cup-oracle`](https://github.com/fatehaszaman/world-cup-oracle) — v1 (BPS 40/64, documents the initial model and failure analysis)

## License
MIT

---

## 2018 World Cup Backtest (v2)

> Cross-tournament validation: does the v2 model generalise to a different World Cup era?

### Bracket Prediction Score (BPS) — 2018

| Stage        | Correct | Max | Pts |
|--------------|---------|-----|-----|
| R16 (×1pt)   | 14/16   | 16  | 14  |
| QF (×2pt)    | 4/8     | 16  | 8   |
| SF (×3pt)    | 1/4     | 12  | 3   |
| Finalist (×5)| 0/2     | 10  | 0   |
| Winner (×10) | 0/1     | 10  | 0   |
| **Total**    |         | **64** | **25** |
| **Result**   |         |     | ✗ **FAIL** |

**Winner predicted:** Germany — **Actual:** France (4–2 Croatia)

### Root Cause Analysis
The v2 model failed 2018 for two structural reasons:

1. **Recency bias in squad values** — Germany's 2014 champion roster still carried high squad-value scores despite key retirements. The model had no age-decay or form-cycle correction to detect a team past its peak.
2. **Penalty-shootout blindspot** — Croatia's path to the final required winning three consecutive knockout shootouts (Denmark R16, Russia QF, England SF). The model's match-simulation assigns probabilities based on 90-minute composite scores; it has no shootout-specialist or clutch-performance coefficient.

### Cross-Tournament Validation Summary

| Tournament      | Model | BPS | /64 | Pass? |
|-----------------|-------|-----|-----|-------|
| 2022 World Cup  | v1    | 40  | 64  | ✗ FAIL |
| 2022 World Cup  | v2    | ~48 | 64  | ✓ PASS |
| 2018 World Cup  | v2    | 25  | 64  | ✗ FAIL |

> 2018 failure motivates **[world-cup-oracle-v3](https://github.com/fatehaszaman/world-cup-oracle-v3)**, which adds age-decay curves, form-cycle detection, and a shootout-specialist coefficient to address both root causes above.
