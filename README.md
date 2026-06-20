# WQ Alpha Research Skill

WorldQuant BRAIN alpha research skill for designing WQ Alpha expressions, searching fields, diagnosing simulation failures, checking IS metrics, managing self-correlation, and organizing low-correlation alpha portfolios.

This repository is a reusable agent skill, not a credential bundle. Keep all account credentials and personal alpha records local.

## Contents

```text
wq-alpha-research/
├── SKILL.md
├── scripts/
│   ├── evolve_skill.py
│   └── submit_batch.py
└── references/
    ├── wq_usa_top3000_delay1_data_fields.csv
    ├── wq_usa_top3000_delay1_data_fields.json
    └── wq_usa_top3000_delay1_data_fields_summary.json
```

## What It Helps With

- Search USA TOP3000 delay=1 BRAIN fields locally.
- Build Alpha expressions from common WQ operator patterns.
- Diagnose low Sharpe, low Fitness, high Turnover, concentrated weights, and sub-universe failures.
- Compare candidate alphas against existing ACTIVE alphas using daily-return correlation.
- Automate simulation/submission workflows with explicit post-submit status checks.
- Keep local alpha research notes evolving without committing private records.

## Quick Start

Read `SKILL.md` first. It contains the actual playbook and trigger instructions.

For local field lookup:

```python
import json
from pathlib import Path

fields = json.loads(
    Path("references/wq_usa_top3000_delay1_data_fields.json").read_text(encoding="utf-8")
)

keyword = "operating_income"
matches = [
    f for f in fields
    if keyword in f["id"].lower()
    or keyword in (f.get("description") or "").lower()
]
print(matches[:5])
```

## BRAIN Credentials

Use environment variables:

```bash
export WQ_BRAIN_USERNAME="your_username"
export WQ_BRAIN_PASSWORD="your_password"
```

Or create a local `credential.txt` file:

```json
["your_username", "your_password"]
```

`credential.txt` is ignored by git. Do not commit real credentials, cookies, sessions, tokens, alpha databases, or submission results.

## Scripts

Preview skill evolution output without modifying files:

```bash
python scripts/evolve_skill.py
```

Apply updates to local `SKILL.md` and `alpha_db.json`:

```bash
python scripts/evolve_skill.py --apply
```

Run the batch submission example:

```bash
python scripts/submit_batch.py
```

The scripts require `requests` and `numpy`.

## Safety Notes

The following files are intentionally ignored:

- `credential.txt`
- `alpha_db.json`
- `batch_submit_results.json`
- `.env`
- Python caches and virtual environments

If you want to publish research lessons from local runs, summarize them into general rules before adding them to `SKILL.md`. Do not publish raw alpha IDs, PnL series, account-linked submission statuses, or private candidate expressions.
