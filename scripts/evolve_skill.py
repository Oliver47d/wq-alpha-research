#!/usr/bin/env python3
"""evolve_skill.py — Experience synchronizer.

Syncs alpha_db.json results into lessons.json and prints summary reports.
This is the bridge between raw simulation results and the lessons feedback loop.

Usage:
    python3 evolve_skill.py --sync          # Sync alpha_db into lessons
    python3 evolve_skill.py --report        # Print lessons report
    python3 evolve_skill.py --sync --apply  # Sync and save
    python3 evolve_skill.py --sync --connect  # Sync with live API connection
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
ALPHA_DB_PATH = SKILL_DIR / "alpha_db.json"
LESSONS_PATH = SKILL_DIR / "lessons.json"


def load_alpha_db() -> dict[str, Any]:
    if ALPHA_DB_PATH.exists():
        return json.loads(ALPHA_DB_PATH.read_text(encoding="utf-8"))
    return {"alphas": {}, "last_update": None, "version": 1}


def load_lessons() -> dict[str, Any]:
    if LESSONS_PATH.exists():
        return json.loads(LESSONS_PATH.read_text(encoding="utf-8"))
    return {"patterns": {}, "param_insights": {}, "version": 1}


def save_lessons(lessons: dict[str, Any]) -> None:
    lessons["last_updated"] = datetime.now(timezone.utc).isoformat()
    LESSONS_PATH.write_text(json.dumps(lessons, indent=2, ensure_ascii=False), encoding="utf-8")


def sync_alpha_db_to_lessons(
    lessons: dict[str, Any],
    db: dict[str, Any],
    client=None,
) -> int:
    """Sync alpha_db entries into lessons.json.

    Only processes entries not yet synced (tracked via _synced_alpha_ids).
    Returns number of newly synced entries.
    """
    synced_ids: list[str] = lessons.setdefault("_synced_alpha_ids", [])
    synced_set = set(synced_ids)
    count = 0

    for alpha_id, alpha in db.get("alphas", {}).items():
        if alpha_id in synced_set:
            continue

        expr = alpha.get("expression", alpha.get("regular", ""))
        if not expr:
            continue

        # Classify alpha
        template_id = alpha.get("template_id", "imported")
        sharpe = alpha.get("sharpe")
        fitness = alpha.get("fitness")
        turnover = alpha.get("turnover")
        status = alpha.get("status", "UNKNOWN")

        # If client provided and we need PnL for correlation
        pnl = alpha.get("pnl")
        if client and not pnl and status == "ACTIVE":
            try:
                pnl = client.fetch_pnl(alpha_id)
                alpha["pnl"] = pnl
            except Exception:
                pass

        # Ensure pattern exists
        patterns = lessons.setdefault("patterns", {})
        if template_id not in patterns:
            patterns[template_id] = {
                "description": f"Template: {template_id}",
                "tested": 0, "passed": 0, "pass_rate": 0.0,
                "avg_sharpe": 0.0, "avg_fitness": 0.0,
                "best": None, "failure_modes": {}, "action": "expand",
                "notes": "",
            }

        p = patterns[template_id]
        p["tested"] += 1

        # Determine pass/fail (ACTIVE = passed)
        passed = status == "ACTIVE"
        if passed:
            p["passed"] += 1
        else:
            if sharpe is None:
                mode = "SIM_ERROR"
            elif sharpe < 1.0:
                mode = "LOW_SHARPE"
            elif fitness is not None and fitness < 1.0:
                mode = "LOW_FITNESS"
            elif turnover is not None and turnover > 0.7:
                mode = "HIGH_TURNOVER"
            else:
                mode = "OTHER"
            fm = p.setdefault("failure_modes", {})
            fm[mode] = fm.get(mode, 0) + 1

        # Update averages
        if sharpe is not None:
            old_count = p["tested"] - 1
            if old_count > 0:
                p["avg_sharpe"] = (p["avg_sharpe"] * old_count + sharpe) / p["tested"]
            else:
                p["avg_sharpe"] = sharpe

        if fitness is not None:
            old_count = p["tested"] - 1
            if old_count > 0:
                p["avg_fitness"] = (p["avg_fitness"] * old_count + fitness) / p["tested"]
            else:
                p["avg_fitness"] = fitness

        p["pass_rate"] = p["passed"] / p["tested"] if p["tested"] > 0 else 0.0

        # Update best
        if sharpe is not None:
            best = p.get("best")
            if best is None or sharpe > best.get("sharpe", 0):
                p["best"] = {
                    "alpha_id": alpha_id,
                    "sharpe": sharpe,
                    "expr": expr,
                }

        # Auto-update action
        if p["tested"] >= 5:
            if p["pass_rate"] == 0.0:
                p["action"] = "skip"
            elif p["pass_rate"] < 0.2:
                p["action"] = "deprioritize"
            else:
                p["action"] = "expand"

        # Update param insights
        settings = alpha.get("settings", {})
        params = {
            "decay": str(settings.get("decay", 0)),
            "neutralization": str(settings.get("neutralization", "INDUSTRY")),
        }
        param_insights = lessons.setdefault("param_insights", {})
        for param_name, param_val in params.items():
            pi = param_insights.setdefault(param_name, {})
            entry = pi.setdefault(param_val, {
                "avg_sharpe": 0.0, "verdict": "neutral", "notes": "", "count": 0
            })
            entry["count"] += 1
            if sharpe is not None:
                old_count = entry["count"] - 1
                if old_count > 0:
                    entry["avg_sharpe"] = (entry["avg_sharpe"] * old_count + sharpe) / entry["count"]
                else:
                    entry["avg_sharpe"] = sharpe
                if entry["count"] >= 3:
                    if entry["avg_sharpe"] >= 1.5:
                        entry["verdict"] = "prefer"
                    elif entry["avg_sharpe"] < 0.8:
                        entry["verdict"] = "deprioritize"

        synced_ids.append(alpha_id)
        count += 1

    lessons["last_updated"] = datetime.now(timezone.utc).isoformat()
    return count


def build_report(lessons: dict[str, Any], db: dict[str, Any]) -> str:
    """Build a human-readable report from lessons and alpha_db."""
    lines = []
    lines.append("=" * 70)
    lines.append("Alpha Mining Lessons Report")
    lines.append("=" * 70)

    patterns = lessons.get("patterns", {})
    if not patterns:
        lines.append("\nNo patterns recorded yet.")
    else:
        lines.append(f"\nPatterns ({len(patterns)}):\n")
        lines.append(f"{'Template':<35} {'Tested':>7} {'Passed':>7} {'Rate':>6} {'Avg Sharpe':>11} {'Action':>12}")
        lines.append("-" * 90)
        for tid, p in sorted(patterns.items()):
            tested = p.get("tested", 0)
            passed = p.get("passed", 0)
            rate = p.get("pass_rate", 0.0)
            avg_sharpe = p.get("avg_sharpe", 0.0)
            action = p.get("action", "expand")
            best_sharpe = (p.get("best") or {}).get("sharpe", "N/A")
            lines.append(f"{tid:<35} {tested:>7} {passed:>7} {rate:>5.0%} {avg_sharpe:>11.2f} {action:>12}")

            # Failure modes
            fm = p.get("failure_modes", {})
            if fm:
                fm_str = ", ".join(f"{k}:{v}" for k, v in sorted(fm.items(), key=lambda x: -x[1]))
                lines.append(f"  └─ Failures: {fm_str}")
            if best_sharpe != "N/A":
                lines.append(f"  └─ Best: Sharpe={best_sharpe}")

    # Param insights
    param_insights = lessons.get("param_insights", {})
    if param_insights:
        lines.append(f"\n\nParam Insights ({len(param_insights)} params):\n")
        for pname, vals in sorted(param_insights.items()):
            lines.append(f"  {pname}:")
            for val, info in sorted(vals.items(), key=lambda x: -x[1].get("count", 0)):
                count = info.get("count", 0)
                avg = info.get("avg_sharpe", 0.0)
                verdict = info.get("verdict", "neutral")
                lines.append(f"    {val:<20} count={count:>3}  avg_sharpe={avg:.2f}  → {verdict}")

    # DB stats
    alphas = db.get("alphas", {})
    active = sum(1 for a in alphas.values() if a.get("status") == "ACTIVE")
    total = len(alphas)
    lines.append(f"\n\nAlpha DB: {total} total, {active} ACTIVE")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Sync alpha_db into lessons and print reports")
    parser.add_argument("--sync", action="store_true", help="Sync alpha_db entries into lessons")
    parser.add_argument("--report", action="store_true", help="Print lessons report")
    parser.add_argument("--apply", action="store_true", help="Save lessons to disk (use with --sync)")
    parser.add_argument("--connect", action="store_true", help="Connect to BRAIN API for live data")
    args = parser.parse_args()

    db = load_alpha_db()
    lessons = load_lessons()

    client = None
    if args.connect:
        try:
            from brain_api import BrainClient
            client = BrainClient()
            client.connect()
        except Exception as e:
            print(f"Warning: Could not connect to BRAIN API: {e}", flush=True)

    if args.sync:
        count = sync_alpha_db_to_lessons(lessons, db, client)
        print(f"Synced {count} new entries into lessons", flush=True)
        if args.apply:
            save_lessons(lessons)
            print(f"Lessons saved to {LESSONS_PATH}", flush=True)
        else:
            print("(dry run — use --apply to save)", flush=True)

    if args.report or not args.sync:
        report = build_report(lessons, db)
        print(report)

    if client:
        if client.session:
            client.session.close()


if __name__ == "__main__":
    main()
