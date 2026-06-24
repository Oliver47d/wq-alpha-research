#!/usr/bin/env python3
"""mining_loop.py — Automatic alpha discovery loop.

Implements the batch fuel-mine pattern:
  while True:
    [BREADTH] expand templates → batch simulate → quality filter → update lessons
    [CHECK]   3 consecutive rounds no ACTIVE → terminate
    [DEPTH]   candidate pool empty? → read next paper → extract templates → back to BREADTH
              no unread papers → terminate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))

from brain_api import (  # noqa: E402
    BrainClient,
    DEFAULT_SETTINGS,
    classify_alpha,
    compute_correlation,
    load_alpha_db,
    load_lessons,
    quality_filter,
    save_alpha_db,
    save_lessons,
    update_lessons_from_result,
)
from generate_candidates import (  # noqa: E402
    deduplicate,
    expand_template,
    load_templates,
)

LESSONS_PATH = SKILL_DIR / "lessons.json"
PAPERS_REGISTRY_PATH = SKILL_DIR / "papers_registry.json"
ALPHA_DB_PATH = SKILL_DIR / "alpha_db.json"
TEMPLATES_DIR = SKILL_DIR / "templates"
REPORT_PATH = SKILL_DIR / "mining_report.json"
STATE_PATH = SKILL_DIR / "mining_state.json"

# Agent CLI for depth extraction (5-minute timeout)
AGENT_TIMEOUT = 300  # seconds
MAX_AGENT_FAILURES = 3

# Round limits
MAX_ROUNDS = 50  # safety cap
MAX_CANDIDATES_PER_ROUND = 60


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text("utf-8"))
    return {
        "round": 0,
        "consecutive_no_active": 0,
        "total_submitted": 0,
        "total_observe": 0,
        "total_discard": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "rounds": [],
    }


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), "utf-8")


def load_papers_registry() -> dict[str, Any]:
    if PAPERS_REGISTRY_PATH.exists():
        return json.loads(PAPERS_REGISTRY_PATH.read_text("utf-8"))
    return {"sources": {}, "stats": {"total": 0, "consumed": 0, "remaining": 0}}


def save_papers_registry(reg: dict) -> None:
    PAPERS_REGISTRY_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False), "utf-8")


# ---------------------------------------------------------------------------
# Breadth phase
# ---------------------------------------------------------------------------

def build_candidates(lessons: dict, max_per_template: int = 15) -> list[dict]:
    """Expand all templates into candidates, filtered by lessons actions."""
    templates = load_templates()
    if not templates:
        print("[breadth] No templates found.")
        return []

    patterns = lessons.get("patterns", {})
    all_candidates: list[dict] = []

    for tmpl in templates:
        tid = tmpl.get("template_id", tmpl.get("_filename", "unknown"))
        pat = patterns.get(tid, {})
        action = pat.get("action", "expand")

        if action == "skip":
            print(f"  [skip] {tid} (pass_rate={pat.get('pass_rate', 0):.0%}, tested={pat.get('tested', 0)})")
            continue

        # Deprioritize: reduce candidate count
        effective_max = max_per_template // 2 if action == "deprioritize" else max_per_template
        cands = expand_template(tmpl, max_candidates=effective_max)
        print(f"  [expand] {tid}: {len(cands)} candidates (action={action}, max={effective_max})")
        all_candidates.extend(cands)

    # Deduplicate
    all_candidates = deduplicate(all_candidates)

    # Cap total
    if len(all_candidates) > MAX_CANDIDATES_PER_ROUND:
        all_candidates = all_candidates[:MAX_CANDIDATES_PER_ROUND]
        print(f"  [cap] Truncated to {MAX_CANDIDATES_PER_ROUND} candidates")

    return all_candidates


def run_breadth_round(
    client: BrainClient,
    candidates: list[dict],
    lessons: dict,
    db: dict,
) -> dict[str, Any]:
    """Run one breadth round: simulate → filter → update lessons → submit good ones."""
    round_result: dict[str, Any] = {
        "candidate_count": len(candidates),
        "submitted": [],
        "observed": [],
        "discarded": 0,
        "errors": 0,
        "new_active": 0,
    }

    if not candidates:
        print("[breadth] No candidates to simulate.")
        return round_result

    # Batch simulate
    print(f"[breadth] Simulating {len(candidates)} candidates...")
    results = client.batch_simulate(candidates)

    # Fetch existing ACTIVE alphas' PnL for correlation
    active_alphas = {
        aid: a for aid, a in db.get("alphas", {}).items() if a.get("status") == "ACTIVE"
    }
    active_pnls: dict[str, list[float]] = {}
    for aid in active_alphas:
        pnl = client.fetch_pnl(aid)
        if len(pnl) >= 50:
            active_pnls[aid] = pnl

    # Process each result
    for r in results:
        sim = r.get("sim_result", {})
        status = sim.get("status", "ERROR")

        if status != "COMPLETE":
            round_result["errors"] += 1
            update_lessons_from_result(lessons, r, sim)
            continue

        sim_data = sim.get("sim_data", {})
        is_data = sim_data.get("is", {}) if isinstance(sim_data, dict) else {}
        sharpe = is_data.get("sharpe")
        fitness = is_data.get("fitness")
        turnover = is_data.get("turnover")
        alpha_id = sim.get("alpha_id")

        # Compute correlation against existing ACTIVE alphas
        max_corr = None
        if alpha_id and active_pnls:
            new_pnl = client.fetch_pnl(alpha_id)
            if len(new_pnl) >= 50:
                corr_list = compute_correlation(new_pnl, {"alphas": {aid: {"pnl": p} for aid, p in active_pnls.items()}})
                if corr_list:
                    max_corr = max(abs(c.get("correlation", 0)) for c in corr_list)

        # Quality filter
        action = quality_filter(sharpe, fitness, turnover, max_corr)

        # Update lessons
        update_lessons_from_result(lessons, r, sim, max_corr)

        if action == "SUBMIT":
            round_result["submitted"].append({
                "alpha_id": alpha_id,
                "expression": r.get("expression", ""),
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover": turnover,
                "max_corr": max_corr,
                "template_id": r.get("template_id", "unknown"),
            })

            # Attempt submission
            if alpha_id:
                print(f"  [SUBMIT] Attempting submission for {alpha_id} (Sharpe={sharpe})")
                submit_result = client.submit_alpha(alpha_id)
                if submit_result.get("status") == "ACTIVE":
                    round_result["new_active"] += 1
                    # Update DB
                    db["alphas"][alpha_id] = {
                        "expression": r.get("expression", ""),
                        "status": "ACTIVE",
                        "sharpe": sharpe,
                        "fitness": fitness,
                        "turnover": turnover,
                        "template_id": r.get("template_id", "unknown"),
                        "submitted_at": datetime.now(timezone.utc).isoformat(),
                    }
                    # Add to active_pnls for subsequent correlation checks
                    new_pnl = client.fetch_pnl(alpha_id)
                    if len(new_pnl) >= 50:
                        active_pnls[alpha_id] = new_pnl
                    print(f"  [ACTIVE] {alpha_id} activated!")
                else:
                    print(f"  [SUBMIT-FAIL] {alpha_id}: {submit_result.get('status', 'unknown')}")

        elif action == "OBSERVE":
            round_result["observed"].append({
                "alpha_id": alpha_id,
                "expression": r.get("expression", ""),
                "sharpe": sharpe,
                "fitness": fitness,
            })
        else:
            round_result["discarded"] += 1

    # Save lessons after each round
    save_lessons(lessons)
    save_alpha_db(db)

    return round_result


# ---------------------------------------------------------------------------
# Depth phase — fuel_one_paper via Agent CLI
# ---------------------------------------------------------------------------

def get_next_paper(reg: dict) -> str | None:
    """Find the next unread paper source ID."""
    for src_id, src in reg.get("sources", {}).items():
        if src.get("status") == "unread":
            return src_id
    return None


def fuel_one_paper(src_id: str, reg: dict, lessons: dict) -> bool:
    """Extract templates from a paper using the Agent CLI.

    Returns True if new templates were extracted, False otherwise.
    """
    src = reg["sources"][src_id]
    src_type = src.get("type", "unknown")
    locator = src.get("locator", "")
    title = src.get("title", locator)

    print(f"\n[depth] Fueling from paper: {title} ({src_type})")

    # Build the prompt for the Agent
    # Summarize lessons for the Agent to use as context
    patterns_summary = []
    for tid, pat in lessons.get("patterns", {}).items():
        patterns_summary.append(
            f"  - {tid}: tested={pat.get('tested',0)}, pass_rate={pat.get('pass_rate',0):.0%}, "
            f"action={pat.get('action') or 'expand'}, best_sharpe={(pat.get('best') or {}).get('sharpe', 'N/A')}"
        )
    lessons_context = "\n".join(patterns_summary) if patterns_summary else "  (no prior patterns)"

    param_insights = []
    for param, insights in lessons.get("param_insights", {}).items():
        param_insights.append(f"  - {param}: {json.dumps(insights, ensure_ascii=False)}")
    param_context = "\n".join(param_insights) if param_insights else "  (no param insights)"

    prompt = f"""You are an alpha research assistant. Read the following research source and extract NEW alpha factor templates.

SOURCE TYPE: {src_type}
SOURCE LOCATOR: {locator}
SOURCE TITLE: {title}

PRIOR MINING LESSONS (use these to guide what templates to extract):
{lessons_context}

PARAMETER INSIGHTS:
{param_context}

TASK:
1. Read the source material thoroughly.
2. Identify 1-3 alpha factor ideas that could be expressed as WorldQuant BRAIN FASTEXPR formulas.
3. For each idea, create a template JSON file in {TEMPLATES_DIR}/ with this structure:
{{
  "template_id": "descriptive_name",
  "description": "What this template captures",
  "skeleton": "group_rank(ts_rank({{numerator}} / {{denominator}}, {{window}}), {{group}})",
  "field_pairs": [
    {{"numerator": "fnd0_mol_12m_oper_inc", "denominator": "mkt_cap", "label": "operating profitability"}}
  ],
  "param_ranges": {{
    "window": [63, 126, 252],
    "group": ["subindustry", "industry", "sector"]
  }},
  "default_settings": {{
    "decay": [0, 2],
    "neutralization": ["SUBINDUSTRY", "INDUSTRY"],
    "truncation": 0.08
  }},
  "examples": [
    {{"expression": "actual_example_expr", "alpha_id": "XXXX", "sharpe": 2.0}}
  ]
}}

RULES:
- Use ONLY fields that exist in references/wq_usa_top3000_delay1_data_fields.json
- Prefer templates that are DIFFERENT from existing patterns in lessons
- Window 126 and 252 tend to work better (from param insights)
- SUBINDUSTRY neutralization works for fundamental signals
- decay 2 is generally safe; avoid decay 20
- Each template should have 3-8 field_pairs
- Write the JSON file(s) directly to {TEMPLATES_DIR}/

Output the filenames you created."""

    # Write prompt to a temp file
    prompt_file = SKILL_DIR / "._fuel_prompt.txt"
    prompt_file.write_text(prompt, "utf-8")

    # Try calling the agent CLI
    # We try multiple approaches since the exact CLI may vary
    agent_commands = [
        ["claude", "--print", "-p", prompt],
    ]

    for cmd_template in agent_commands:
        try:
            # Use subprocess with timeout
            result = subprocess.run(
                cmd_template[0:1] + cmd_template[1:],
                capture_output=True,
                text=True,
                timeout=AGENT_TIMEOUT,
                cwd=str(SKILL_DIR),
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"  [depth] Agent output: {output[:200]}...")

                # Check if new template files were created
                templates_before = set(p.name for p in TEMPLATES_DIR.glob("*.json"))
                # The agent should have written files; let's re-scan
                templates_after = set(p.name for p in TEMPLATES_DIR.glob("*.json"))
                new_templates = templates_after - templates_before

                if new_templates:
                    print(f"  [depth] New templates created: {new_templates}")
                    # Mark paper as consumed
                    reg["sources"][src_id]["status"] = "consumed"
                    reg["sources"][src_id]["read_date"] = datetime.now(timezone.utc).isoformat()
                    reg["sources"][src_id]["extracted_templates"] = list(new_templates)
                    reg["sources"][src_id]["extraction_round"] = reg.get("stats", {}).get("consumed", 0) + 1
                    reg["stats"]["consumed"] = reg["stats"].get("consumed", 0) + 1
                    reg["stats"]["remaining"] = max(0, reg["stats"].get("total", 0) - reg["stats"]["consumed"])
                    save_papers_registry(reg)
                    return True
                else:
                    print(f"  [depth] Agent ran but no new template files detected")
                    # Still mark as consumed to avoid re-reading
                    reg["sources"][src_id]["status"] = "consumed"
                    reg["sources"][src_id]["read_date"] = datetime.now(timezone.utc).isoformat()
                    reg["stats"]["consumed"] = reg["stats"].get("consumed", 0) + 1
                    reg["stats"]["remaining"] = max(0, reg["stats"].get("total", 0) - reg["stats"]["consumed"])
                    save_papers_registry(reg)
                    return False
            else:
                print(f"  [depth] Agent exited with code {result.returncode}: {result.stderr[:200]}")
                continue
        except subprocess.TimeoutExpired:
            print(f"  [depth] Agent timed out after {AGENT_TIMEOUT}s")
            continue
        except FileNotFoundError:
            print(f"  [depth] Agent CLI not found: {cmd_template[0]}")
            continue
        except Exception as e:
            print(f"  [depth] Agent error: {e}")
            continue

    # If we get here, all agent attempts failed
    print(f"  [depth] All agent attempts failed for {src_id}")
    return False


# ---------------------------------------------------------------------------
# Depth phase — manual fuel (no external agent dependency)
# ---------------------------------------------------------------------------

def fuel_one_paper_manual(src_id: str, reg: dict, lessons: dict) -> bool:
    """Fallback: manually extract templates from a paper without Agent CLI.

    This reads the paper if it's a local file and tries to generate templates
    using simple heuristics. Used when Agent CLI is unavailable.
    """
    src = reg["sources"][src_id]
    src_type = src.get("type", "unknown")
    locator = src.get("locator", "")

    print(f"\n[depth-manual] Attempting manual extraction from: {locator}")

    if src_type == "pdf" or src_type == "markdown":
        # Try to read the file
        path = Path(locator)
        if not path.is_absolute():
            path = SKILL_DIR / locator
        if not path.exists():
            print(f"  [manual] File not found: {path}")
            return False

        try:
            text = path.read_text("utf-8", errors="ignore")[:10000]
        except Exception as e:
            print(f"  [manual] Failed to read: {e}")
            return False

        # Simple heuristic: look for formula-like patterns
        # This is a very basic fallback
        print(f"  [manual] Read {len(text)} chars. Heuristic extraction not implemented.")
        print(f"  [manual] Marking as consumed to avoid re-processing.")

        reg["sources"][src_id]["status"] = "consumed"
        reg["sources"][src_id]["read_date"] = datetime.now(timezone.utc).isoformat()
        reg["stats"]["consumed"] = reg["stats"].get("consumed", 0) + 1
        reg["stats"]["remaining"] = max(0, reg["stats"].get("total", 0) - reg["stats"]["consumed"])
        save_papers_registry(reg)
        return False

    # For web/feishu sources, we can't easily fetch without tools
    print(f"  [manual] Cannot extract from {src_type} source without Agent CLI")
    return False


# ---------------------------------------------------------------------------
# Termination logic
# ---------------------------------------------------------------------------

def should_terminate(state: dict, reg: dict, has_candidates: bool) -> tuple[bool, str]:
    """Check termination conditions."""
    # Check round cap
    if state["round"] >= MAX_ROUNDS:
        return True, f"Reached max rounds ({MAX_ROUNDS})"

    # Check consecutive no-active
    if state["consecutive_no_active"] >= 3:
        return True, "3 consecutive rounds with no new ACTIVE alphas"

    # Check candidate pool + papers
    remaining = reg.get("stats", {}).get("remaining", 0)
    if not has_candidates and remaining == 0:
        return True, "Candidate pool empty and no unread papers remaining"

    return False, ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_mining_loop(max_rounds: int | None = None, dry_run: bool = False) -> None:
    """Main entry point for the automatic alpha mining loop."""
    global MAX_ROUNDS
    if max_rounds:
        MAX_ROUNDS = max_rounds

    print("=" * 70)
    print("  WorldQuant BRAIN — Automatic Alpha Discovery System")
    print("=" * 70)
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Max rounds: {MAX_ROUNDS}")
    print(f"  Max candidates per round: {MAX_CANDIDATES_PER_ROUND}")
    print(f"  Skill dir: {SKILL_DIR}")
    print("=" * 70)

    # Load state
    state = load_state()
    lessons = load_lessons()
    db = load_alpha_db()
    reg = load_papers_registry()

    agent_failures = 0

    # Connect to BRAIN API
    if not dry_run:
        print("\n[init] Connecting to BRAIN API...")
        client = BrainClient()
        try:
            client.connect()
            print("[init] Connected successfully.")
        except Exception as e:
            print(f"[init] FATAL: Failed to connect to BRAIN API: {e}")
            sys.exit(1)
    else:
        client = None  # type: ignore
        print("[init] Dry run mode — no API calls will be made.")

    # Main loop
    while True:
        state["round"] += 1
        round_num = state["round"]
        print(f"\n{'─' * 70}")
        print(f"  ROUND {round_num}")
        print(f"{'─' * 70}")

        # ── BREADTH PHASE ──
        print("\n[breadth] Building candidates from templates...")
        candidates = build_candidates(lessons)
        has_candidates = len(candidates) > 0

        round_data: dict[str, Any] = {
            "round": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(candidates),
        }

        if has_candidates and not dry_run:
            # Run breadth round
            round_result = run_breadth_round(client, candidates, lessons, db)
            round_data.update(round_result)

            new_active = round_result["new_active"]
            if new_active > 0:
                state["consecutive_no_active"] = 0
            else:
                state["consecutive_no_active"] += 1

            state["total_submitted"] += len(round_result["submitted"])
            state["total_observe"] += len(round_result["observed"])
            state["total_discard"] += round_result["discarded"]

            print(f"\n[breadth] Round {round_num} summary:")
            print(f"  Candidates: {round_result['candidate_count']}")
            print(f"  SUBMIT: {len(round_result['submitted'])} (new ACTIVE: {new_active})")
            print(f"  OBSERVE: {len(round_result['observed'])}")
            print(f"  DISCARD: {round_result['discarded']}")
            print(f"  ERRORS:  {round_result['errors']}")

        elif dry_run and has_candidates:
            print(f"\n[dry-run] Would simulate {len(candidates)} candidates")
            for i, c in enumerate(candidates[:5]):
                print(f"  [{i+1}] {c.get('expression', '?')[:80]}")
            if len(candidates) > 5:
                print(f"  ... and {len(candidates) - 5} more")
            round_data["dry_run"] = True

        else:
            print("\n[breadth] No candidates generated.")
            round_data["candidate_count"] = 0

        # ── CHECK TERMINATION ──
        should_stop, reason = should_terminate(state, reg, has_candidates)
        if should_stop:
            print(f"\n[terminate] {reason}")
            round_data["termination_reason"] = reason
            state["rounds"].append(round_data)
            break

        # ── DEPTH PHASE ──
        # Only trigger depth if candidate pool is empty (templates exhausted)
        if not has_candidates:
            next_paper = get_next_paper(reg)
            if next_paper:
                print(f"\n[depth] Candidate pool empty. Reading next paper: {next_paper}")
                fueled = False

                # Try Agent CLI first
                if agent_failures < MAX_AGENT_FAILURES:
                    try:
                        fueled = fuel_one_paper(next_paper, reg, lessons)
                    except Exception as e:
                        print(f"  [depth] Agent exception: {e}")
                        agent_failures += 1

                # Fallback to manual
                if not fueled and agent_failures >= MAX_AGENT_FAILURES:
                    print(f"\n[depth] Agent failed {agent_failures} times. Falling back to manual extraction.")
                    fuel_one_paper_manual(next_paper, reg, lessons)

                round_data["depth_triggered"] = True
                round_data["paper_read"] = next_paper

                # After reading a paper, continue to next breadth round
                # (templates may have been added)
            else:
                print(f"\n[terminate] No unread papers remaining and candidate pool empty.")
                round_data["termination_reason"] = "No unread papers and empty candidate pool"
                state["rounds"].append(round_data)
                break

        # Save state after each round
        state["rounds"].append(round_data)
        save_state(state)

        # Brief pause between rounds
        if not dry_run:
            print("\n[loop] Pausing 5s before next round...")
            time.sleep(5)

    # ── FINAL REPORT ──
    state["ended_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    # Generate mining report
    report = {
        "started_at": state["started_at"],
        "ended_at": state["ended_at"],
        "total_rounds": state["round"],
        "total_submitted": state["total_submitted"],
        "total_observe": state["total_observe"],
        "total_discard": state["total_discard"],
        "consecutive_no_active": state["consecutive_no_active"],
        "rounds": state["rounds"],
        "lessons_snapshot": lessons,
        "papers_registry_snapshot": load_papers_registry(),
        "active_alphas": {
            aid: {"sharpe": a.get("sharpe"), "expression": a.get("expression", "")[:100]}
            for aid, a in db.get("alphas", {}).items()
            if a.get("status") == "ACTIVE"
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")

    print(f"\n{'=' * 70}")
    print("  MINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total rounds:       {state['round']}")
    print(f"  Total SUBMIT:       {state['total_submitted']}")
    print(f"  Total OBSERVE:      {state['total_observe']}")
    print(f"  Total DISCARD:      {state['total_discard']}")
    print(f"  Active alphas in DB: {sum(1 for a in db.get('alphas', {}).values() if a.get('status') == 'ACTIVE')}")
    print(f"  Report saved:       {REPORT_PATH}")
    print(f"  State saved:        {STATE_PATH}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automatic alpha discovery mining loop."
    )
    parser.add_argument(
        "--max-rounds", type=int, default=None,
        help=f"Maximum number of rounds (default: {MAX_ROUNDS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't actually call the API; just show what would happen.",
    )
    parser.add_argument(
        "--reset-state", action="store_true",
        help="Reset mining state before starting.",
    )
    args = parser.parse_args()

    if args.reset_state and STATE_PATH.exists():
        STATE_PATH.unlink()
        print("[init] Mining state reset.")

    run_mining_loop(max_rounds=args.max_rounds, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
