# WQ Alpha Research — Automatic Alpha Discovery System

A self-evolving WorldQuant BRAIN alpha research system that runs **unattended**. It combines script-driven breadth (parameter grid exploration) with agent-driven depth (paper / report inspiration extraction) in a batch fuel-mine loop, turning each simulation result into reusable lessons for the next round.

> **Target platform**: WorldQuant BRAIN · USA TOP3000 · delay=1 · 4,367 data fields

---

## How It Works

```
Papers / Reports → papers_registry.json (tracker)
  → Agent extracts templates per paper → templates/*.json
    → mining_loop.py (expand params → batch simulate → quality filter)
      → lessons.json (feedback loop)
        → next paper extraction references lessons.json
```

The system alternates between two engines:

| Engine | Role | Mechanism |
|--------|------|-----------|
| **Breadth** (scripts) | Parameter grid exploration | `generate_candidates.py` expands template skeletons × field pairs × param ranges into candidate expressions |
| **Depth** (agent) | Paper-inspired template creation | Agent reads papers, extracts structured template JSON with skeleton + field pairs + param ranges + hypothesis |

### Batch Fuel-Mine Loop

```python
while True:
    # BREADTH: expand templates → batch simulate → quality filter → update lessons
    candidates = generate_candidates.expand(templates, lessons)
    results = brain_api.batch_simulate(candidates)
    classified = quality_filter(results, alpha_db)
    update_lessons(lessons, results)

    # CHECK: 3 consecutive rounds no new ACTIVE → terminate
    if no_active_streak >= 3:
        break

    # DEPTH: candidate pool empty?
    if not candidates:
        if has_unread_papers:
            fuel_one_paper()       # Agent extracts new templates
        else:
            break                   # Fuel exhausted
```

### Quality Filtering

| Tier | Criteria |
|------|----------|
| **SUBMIT** | Sharpe ≥ 1.5 AND Fitness ≥ 1.0 AND Turnover < 0.7 AND correlation with ACTIVE < 0.7 |
| **OBSERVE** | Sharpe ≥ 1.0 (tuning seed) OR Sharpe ≥ 1.5 but high correlation (redundancy demotion) OR Sharpe 1.25–1.5 with correlation < 0.3 (low-correlation upgrade) |
| **DISCARD** | Everything else |

### Termination Conditions

- **Fuel exhausted**: no candidates AND no unread papers
- **Effectiveness exhausted**: 3 consecutive rounds with no new ACTIVE alpha

---

## Project Structure

```
world_quant/
├── README.md                          ← This file
├── SKILL.md                           ← Agent skill playbook (727 lines)
├── credential.txt                     ← BRAIN API credentials (gitignored)
├── alpha_db.json                      ← Alpha snapshot database (43 alphas, 6 ACTIVE)
│
├── templates/                         ← Structured template JSONs
│   ├── profitability_trend.json       ← group_rank(ts_rank(numerator/denominator, window), group)
│   ├── analyst_estimate_trend.json    ← Analyst expectation fields with turnover control
│   ├── hybrid_tech_fundamental.json   ← Weighted tech + fundamental signal mixing
│   └── overnight_reversal.json        ← Overnight price reversal with decay
│
├── lessons.json                       ← Pattern-level mining experience
├── papers_registry.json               ← Paper / report tracking registry
│
├── scripts/
│   ├── mining_loop.py                 ← Main loop orchestrator (691 lines)
│   ├── brain_api.py                   ← BRAIN API client: simulate, submit, correlate (519 lines)
│   ├── generate_candidates.py         ← Template expansion engine (228 lines)
│   ├── evolve_skill.py                ← Experience sync: alpha_db ↔ lessons.json (274 lines)
│   ├── submit_batch.py                ← Batch submission with status polling (147 lines)
│   ├── DESIGN.md                      ← Architecture design document
│   └── archive/
│       ├── run_alpha101.py            ← Legacy Alpha-101 experiment v1
│       └── run_alpha101_v2.py         ← Legacy Alpha-101 experiment v2
│
└── references/                        ← Local BRAIN data field snapshot
    ├── wq_usa_top3000_delay1_data_fields.csv       (1.4 MB, 4,367 fields)
    ├── wq_usa_top3000_delay1_data_fields.json      (2.8 MB)
    └── wq_usa_top3000_delay1_data_fields_summary.json
```

---

## Quick Start

### Prerequisites

```bash
pip install requests numpy
```

### 1. Configure credentials

Create `credential.txt` in the project root:

```
username:password
```

### 2. Run the mining loop

```bash
# Dry run (no API calls, previews candidate expansion)
python3 scripts/mining_loop.py --dry-run

# Full automatic mining
python3 scripts/mining_loop.py

# With custom settings
python3 scripts/mining_loop.py --max-rounds 50 --max-candidates 60
```

### 3. Check results

```bash
# View alpha database
python3 -c "import json; db=json.load(open('alpha_db.json')); print(f'{len(db[\"alphas\"])} alphas')"

# View mining report
cat mining_report.json | python3 -m json.tool

# View lessons learned
python3 -c "import json; l=json.load(open('lessons.json')); print(json.dumps(l, indent=2))"
```

---

## Template Format

Each template is a JSON file in `templates/`:

```json
{
  "template_id": "profitability_trend",
  "description": "Profitability trend factor: use fundamental data trends to predict cross-sectional returns",
  "skeleton": "group_rank(ts_rank({numerator} / {denominator}, {window}), {group})",
  "field_pairs": [
    {"numerator": "operating_income", "denominator": "equity"},
    {"numerator": "net_income", "denominator": "equity"},
    {"numerator": "free_cash_flow_per_share", "denominator": "close"}
  ],
  "param_ranges": {
    "window": [63, 126, 252],
    "group": ["subindustry", "industry", "sector"]
  },
  "hypothesis": "Companies with improving profitability tend to outperform; group_rank controls industry effects",
  "source": "src_001",
  "created": "2026-06-24",
  "proven_examples": []
}
```

The expansion engine creates `field_pairs × param_ranges` candidate expressions per template, then deduplicates against the alpha database.

---

## Lessons Feedback Loop

`lessons.json` accumulates mining experience at the pattern level:

```json
{
  "patterns": {
    "profitability_trend": {
      "tested": 12, "passed": 4, "pass_rate": 0.33,
      "avg_sharpe": 1.52, "avg_fitness": 1.08,
      "best": {"alpha_id": "...", "sharpe": 2.01, "expr": "..."},
      "failure_modes": {"LOW_FITNESS": 5, "LOW_SHARPE": 3},
      "action": "expand",
      "notes": "..."
    }
  },
  "param_insights": {
    "window": {"63": {"avg_sharpe": 0.9, "verdict": "deprioritize"}, "126": {...}, "252": {...}}
  }
}
```

After each batch, `evolve_skill.py` syncs results back into `lessons.json`, which the next round's candidate generator consults to prioritize high-performing patterns and deprioritize failing ones.

---

## Self-Evolution Loop

```bash
# Preview new lessons from latest alpha_db changes
python3 scripts/evolve_skill.py

# Apply lessons locally
python3 scripts/evolve_skill.py --apply

# Batch submit qualified candidates
python3 scripts/submit_batch.py
```

### Mining Rules Distilled

- **Fundamental fields** are the strongest starting point; `group_rank + ts_rank` with `SUBINDUSTRY` neutralization is the default baseline.
- **Analyst expectation fields** are useful but need turnover control via modest decay and industry/subindustry neutralization.
- **Pure technical signals** fail more often unless decay is high or mixed with slower fundamental signals.
- **Fitness failures** are often turnover problems in disguise; decay and signal mixing are the first levers to check.
- **Self-correlation** must be measured on daily PnL changes, not cumulative PnL curves.
- **Low correlation** usually requires a different data source or economic logic, not just parameter tweaks.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Hybrid architecture (scripts + agent) | Scripts handle breadth (fast, unattended); agent handles depth (paper-inspired creativity) |
| Batch fuel-mine loop | Alternates mining (consume templates) with fueling (read papers for new templates) |
| Pattern-level lessons | Aggregates stats per template pattern, not per individual alpha |
| Adaptive quality filter | Three-tier (SUBMIT/OBSERVE/DISCARD) with correlation-aware demotion/promotion |
| Local field snapshot | 4,367 BRAIN fields cached locally for fast validation without API calls |
| Agent CLI subprocess | 5-minute timeout, skip-on-failure, 3 consecutive failures → depth disabled |

See [`scripts/DESIGN.md`](scripts/DESIGN.md) for the full architecture document.

---

## Privacy & Security

- `credential.txt`, `alpha_db.json`, and result files are gitignored
- Only sanitized general rules are published — never raw account-linked records
- The repository is a reusable system, not a credential bundle

---

## License

This project builds on the [wq-alpha-research](https://github.com/QuantML-Research/wq-alpha-research) skill framework.
