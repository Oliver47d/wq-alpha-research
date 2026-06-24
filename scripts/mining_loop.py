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
    FIELDS_PATH,
    FieldValidator,
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
DEPTH_REQUEST_PATH = SKILL_DIR / "depth_request.json"
DEPTH_RESPONSE_PATH = SKILL_DIR / "depth_response.json"

# Agent CLI for depth extraction (5-minute timeout)
AGENT_TIMEOUT = 300  # seconds
MAX_AGENT_FAILURES = 3

# Round limits
MAX_ROUNDS = 50  # safety cap
MAX_CANDIDATES_PER_ROUND = 60
DEPTH_BACKENDS = {"handoff", "claude", "manual", "none"}


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


def _scan_papers_folder() -> dict[str, dict]:
    """Scan papers/ directory and return {filename: entry} for unregistered PDFs."""
    papers_dir = SKILL_DIR / "papers"
    if not papers_dir.is_dir():
        return {}
    entries = {}
    for pdf_file in sorted(papers_dir.glob("*.pdf")):
        locator = f"papers/{pdf_file.name}"
        # Extract title: strip leading number prefix like "1. " or "10. "
        name = pdf_file.stem
        name = name.split(". ", 1)[-1] if ". " in name else name
        entries[locator] = {
            "locator": locator,
            "title": name,
            "type": "research_report",
            "status": "unread",
        }
    return entries


def _generate_src_id(existing_keys: list[str]) -> str:
    """Generate next src_XXX ID."""
    nums = []
    for k in existing_keys:
        if k.startswith("src_"):
            try:
                nums.append(int(k.split("_")[1]))
            except (IndexError, ValueError):
                pass
    nxt = max(nums) + 1 if nums else 1
    return f"src_{nxt:03d}"


def load_papers_registry() -> dict[str, Any]:
    reg: dict[str, Any] = {
        "sources": {},
        "stats": {"total": 0, "consumed": 0, "remaining": 0},
    }
    if PAPERS_REGISTRY_PATH.exists():
        reg = json.loads(PAPERS_REGISTRY_PATH.read_text("utf-8"))

    # Auto-scan papers/ and register any new PDFs not yet in registry
    scanned = _scan_papers_folder()
    existing_locators = {v["locator"] for v in reg.get("sources", {}).values()}
    added = 0
    for locator, entry in scanned.items():
        if locator not in existing_locators:
            new_key = _generate_src_id(list(reg["sources"].keys()))
            reg["sources"][new_key] = entry
            added += 1

    if added > 0:
        sources = reg["sources"]
        reg["stats"] = {
            "total": len(sources),
            "consumed": sum(1 for s in sources.values() if s.get("status") == "consumed"),
            "remaining": sum(1 for s in sources.values() if s.get("status") != "consumed"),
        }
        save_papers_registry(reg)
        print(f"[papers] Auto-registered {added} new PDF(s) from papers/")

    return reg


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

    # Always validate fields against BRAIN reference to avoid simulation errors
    validator = FieldValidator(FIELDS_PATH)
    print(f"  [field-validator] Loaded {len(validator.field_list)} fields for validation")

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
        cands = expand_template(tmpl, max_candidates=effective_max, validator=validator)
        print(f"  [expand] {tid}: {len(cands)} candidates (action={action}, max={effective_max})")
        all_candidates.extend(cands)

    # Deduplicate
    all_candidates = deduplicate(all_candidates)

    # Cap total
    if len(all_candidates) > MAX_CANDIDATES_PER_ROUND:
        all_candidates = all_candidates[:MAX_CANDIDATES_PER_ROUND]
        print(f"  [cap] Truncated to {MAX_CANDIDATES_PER_ROUND} candidates")

    return all_candidates


def _compact_json(value: Any, max_chars: int = 2000) -> Any:
    """Keep error payloads readable in mining reports."""
    if value is None:
        return None
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return json.loads(text)
    return text[:max_chars] + "...[truncated]"


def _error_detail(result: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_idx": result.get("batch_idx"),
        "expression": result.get("expression", ""),
        "template_id": result.get("template_id", "unknown"),
        "settings": result.get("settings", {}),
        "status": sim.get("status"),
        "status_code": sim.get("status_code"),
        "error": sim.get("error"),
        "attempts": sim.get("attempts"),
        "simulation_id": sim.get("simulation_id"),
        "alpha_id": sim.get("alpha_id"),
        "sim_data": _compact_json(sim.get("sim_data")),
    }


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
        "error_details": [],
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
            round_result["error_details"].append(_error_detail(r, sim))
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
                submit_status = submit_result.get("status", "unknown")
                if submit_status == "ACTIVE":
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
                elif submit_status == "PENDING":
                    # Submission accepted but still under review — not a failure
                    db["alphas"][alpha_id] = {
                        "expression": r.get("expression", ""),
                        "status": "PENDING",
                        "sharpe": sharpe,
                        "fitness": fitness,
                        "turnover": turnover,
                        "template_id": r.get("template_id", "unknown"),
                        "submitted_at": datetime.now(timezone.utc).isoformat(),
                    }
                    print(f"  [SUBMIT-PENDING] {alpha_id} submitted, awaiting review")
                else:
                    print(f"  [SUBMIT-FAIL] {alpha_id}: {submit_status}")

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

def load_skill_knowledge() -> str:
    """Extract key knowledge from SKILL.md to inject into DEPTH prompts.

    This ensures the Agent has WorldQuant domain expertise even when
    SKILL.md is not auto-loaded by the Claude Code skill system.
    """
    skill_path = SKILL_DIR / "SKILL.md"
    if not skill_path.exists():
        return "(SKILL.md not found)"

    text = skill_path.read_text("utf-8", errors="ignore")

    sections: list[str] = []

    # Section 4: High-win-rate templates + recommended settings
    sec4 = _extract_section(text, "## 4. \u56e0\u5b50\u6a21\u677f\u5e93", "## 5.")
    if sec4:
        sections.append("### HIGH-WIN-RATE TEMPLATES & DEFAULT SETTINGS\n" + sec4)

    # Section 6: Problem diagnosis & fixes
    sec6 = _extract_section(text, "## 6. \u95ee\u9898\u8bca\u65ad\u4e0e\u4fee\u590d", "## 7.")
    if sec6:
        sections.append("### PROBLEM DIAGNOSIS & FIXES\n" + sec6)

    # Section 10: Core experience (one-liners)
    sec10 = _extract_section(text, "## 10. \u6838\u5fc3\u7ecf\u9a8c", "## 11.")
    if sec10:
        sections.append("### CORE EXPERIENCE (ONE-LINERS)\n" + sec10)

    if not sections:
        return "(No relevant sections found in SKILL.md)"

    return "\n\n".join(sections)


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """Extract a section from markdown text between two markers."""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    end_idx = text.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        end_idx = len(text)
    return text[start_idx:end_idx].strip()


def get_next_paper(reg: dict) -> str | None:
    """Find the next unread paper source ID."""
    for src_id, src in reg.get("sources", {}).items():
        if src.get("status") == "unread":
            return src_id
    return None


def _refresh_registry_stats(reg: dict) -> None:
    sources = reg.get("sources", {})
    reg["stats"] = {
        "total": len(sources),
        "consumed": sum(1 for s in sources.values() if s.get("status") == "consumed"),
        "remaining": sum(1 for s in sources.values() if s.get("status") != "consumed"),
    }


def create_depth_request(src_id: str, reg: dict, lessons: dict, reason: str) -> dict[str, Any]:
    """Create a handoff task for the outer Trae Agent/subagent depth phase."""
    src = reg["sources"][src_id]
    existing_templates = sorted(p.name for p in TEMPLATES_DIR.glob("*.json"))
    request = {
        "status": "NEED_AGENT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "source_id": src_id,
        "paper": {
            "title": src.get("title", src.get("locator", "")),
            "locator": src.get("locator", ""),
            "type": src.get("type", "unknown"),
        },
        "paths": {
            "skill_dir": str(SKILL_DIR),
            "skill_path": str(SKILL_DIR / "SKILL.md"),
            "templates_dir": str(TEMPLATES_DIR),
            "lessons_path": str(LESSONS_PATH),
            "fields_path": str(FIELDS_PATH),
            "papers_registry_path": str(PAPERS_REGISTRY_PATH),
            "depth_response_path": str(DEPTH_RESPONSE_PATH),
        },
        "existing_templates": existing_templates,
        "lessons_summary": {
            "patterns": lessons.get("patterns", {}),
            "param_insights": lessons.get("param_insights", {}),
        },
        "agent_task": {
            "instructions": [
                "Read paths.skill_path first and follow its WorldQuant BRAIN alpha design rules.",
                "Read paths.lessons_path and use prior mining lessons to avoid repeated failures.",
                "Read the paper at paper.locator from this workspace.",
                "Extract 1-3 WorldQuant BRAIN FASTEXPR template ideas.",
                "Use only fields present in paths.fields_path.",
                "Write valid template JSON files directly into paths.templates_dir.",
                "Write depth_response.json with status=DONE, source_id, created_templates, and notes.",
            ],
            "template_contract": {
                "required_keys": [
                    "template_id",
                    "description",
                    "skeleton",
                    "field_pairs",
                    "param_ranges",
                    "default_settings",
                    "hypothesis",
                    "source",
                ],
                "max_templates": 3,
            },
        },
    }
    DEPTH_REQUEST_PATH.write_text(json.dumps(request, indent=2, ensure_ascii=False), "utf-8")
    reg["sources"][src_id]["status"] = "agent_requested"
    reg["sources"][src_id]["request_date"] = request["created_at"]
    _refresh_registry_stats(reg)
    save_papers_registry(reg)
    return request


def consume_depth_response(reg: dict) -> str:
    """Consume depth_response.json from the outer Agent and update paper registry.

    Returns: absent | consumed | blocked.
    """
    if not DEPTH_RESPONSE_PATH.exists():
        return "absent"

    try:
        response = json.loads(DEPTH_RESPONSE_PATH.read_text("utf-8"))
    except json.JSONDecodeError as e:
        print(f"[depth] Invalid depth response JSON: {e}")
        return "blocked"

    pending_request = None
    if DEPTH_REQUEST_PATH.exists():
        try:
            pending_request = json.loads(DEPTH_REQUEST_PATH.read_text("utf-8"))
        except json.JSONDecodeError as e:
            print(f"[depth] Invalid pending depth request JSON: {e}")
            return "blocked"
    else:
        print("[depth] depth_response.json exists without a matching depth_request.json.")
        return "blocked"

    if response.get("status") != "DONE":
        print(f"[depth] Found depth response with status={response.get('status')}; leaving it untouched.")
        return "blocked"

    src_id = response.get("source_id")
    if not src_id or src_id not in reg.get("sources", {}):
        print(f"[depth] Invalid depth response source_id: {src_id}")
        return "blocked"

    if pending_request.get("source_id") != src_id:
        print(
            "[depth] Depth response source_id does not match pending request: "
            f"response={src_id}, request={pending_request.get('source_id')}"
        )
        return "blocked"

    created_templates = response.get("created_templates", [])
    if not isinstance(created_templates, list):
        print("[depth] Invalid depth response: created_templates must be a list")
        return "blocked"

    normalized_templates = []
    missing = []
    existing_templates = set(pending_request.get("existing_templates", []))
    for name in created_templates:
        if not isinstance(name, str):
            print(f"[depth] Invalid template name in response: {name!r}")
            return "blocked"
        filename = name if str(name).endswith(".json") else f"{name}.json"
        if filename in existing_templates:
            print(f"[depth] Depth response references pre-existing template: {filename}")
            return "blocked"
        if Path(filename).name != filename:
            print(f"[depth] Invalid template path in response: {filename}")
            return "blocked"
        template_path = (TEMPLATES_DIR / filename).resolve()
        templates_root = TEMPLATES_DIR.resolve()
        if template_path.parent != templates_root:
            print(f"[depth] Template path escapes templates directory: {filename}")
            return "blocked"
        if template_path.exists():
            normalized_templates.append(filename)
        else:
            missing.append(filename)
    if missing:
        print(f"[depth] Depth response references missing template files: {missing}")
        return "blocked"

    src = reg["sources"][src_id]
    src["status"] = "consumed"
    src["read_date"] = response.get("completed_at", datetime.now(timezone.utc).isoformat())
    src["extracted_templates"] = normalized_templates
    src["extraction_round"] = reg.get("stats", {}).get("consumed", 0) + 1
    if response.get("notes"):
        src["notes"] = response["notes"]
    _refresh_registry_stats(reg)
    save_papers_registry(reg)

    DEPTH_RESPONSE_PATH.unlink()
    if DEPTH_REQUEST_PATH.exists():
        DEPTH_REQUEST_PATH.unlink()

    print(f"[depth] Consumed depth response for {src_id}: {normalized_templates}")
    return "consumed"


def fuel_one_paper(src_id: str, reg: dict, lessons: dict) -> bool:
    """Extract templates from a paper using the Agent CLI.

    Returns True if new templates were extracted, False otherwise.
    """
    src = reg["sources"][src_id]
    src_type = src.get("type", "unknown")
    locator = src.get("locator", "")
    title = src.get("title", locator)

    print(f"\n[depth] Fueling from paper: {title} ({src_type})")

    # Load SKILL.md knowledge for the prompt
    skill_knowledge = load_skill_knowledge()
    print(f"  [depth] Loaded SKILL.md knowledge ({len(skill_knowledge)} chars)")

    # List existing templates so Agent avoids duplicates
    existing_templates = [p.stem for p in TEMPLATES_DIR.glob("*.json")]
    existing_list = ", ".join(existing_templates) if existing_templates else "(none)"

    # Snapshot templates BEFORE agent runs (fix: was computed after agent ran)
    templates_before = set(p.name for p in TEMPLATES_DIR.glob("*.json"))

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

EXISTING TEMPLATES (do NOT duplicate these): {existing_list}

PRIOR MINING LESSONS (use these to guide what templates to extract):
{lessons_context}

PARAMETER INSIGHTS:
{param_context}

DOMAIN KNOWLEDGE FROM SKILL.md (use these rules and patterns):
{skill_knowledge}

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

RULES (from SKILL.md domain knowledge):
- Use ONLY fields that exist in references/wq_usa_top3000_delay1_data_fields.json
- Prefer templates that are DIFFERENT from existing patterns in lessons and existing templates listed above
- group_rank + ts_rank is the golden combination
- SUBINDUSTRY neutralization has highest pass rate for fundamental signals
- Window 126 and 252 tend to work better (from param insights)
- Decay: 0 for fundamentals, 0-4 for analyst, 10-30 for technical reversal
- Fundamental > hybrid > technical in terms of pass rate
- Low correlation requires different DATA SOURCES, not just parameter tweaks
- Each template should have 3-8 field_pairs
- Include a clear "hypothesis" field explaining the economic logic
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

                # Check if new template files were created (templates_before was snapshotted before agent ran)
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
    print(f"  [depth] Claude CLI unavailable. Prompt saved to {prompt_file}")
    print(f"  [depth] To fuel manually: copy the prompt to Mira Agent or another LLM")
    print(f"  [depth] The prompt includes SKILL.md knowledge ({len(skill_knowledge)} chars) + lessons context")
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
        return True, (
            "3 consecutive rounds with no new ACTIVE alphas. "
            "Tip: run with --reset-state to start fresh."
        )

    # Check candidate pool + papers
    remaining = reg.get("stats", {}).get("remaining", 0)
    if not has_candidates and remaining == 0:
        return True, "Candidate pool empty and no unread papers remaining"

    return False, ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_mining_loop(
    max_rounds: int | None = None,
    dry_run: bool = False,
    depth_backend: str = "handoff",
) -> None:
    """Main entry point for the automatic alpha mining loop."""
    global MAX_ROUNDS
    if max_rounds:
        MAX_ROUNDS = max_rounds
    if depth_backend not in DEPTH_BACKENDS:
        raise ValueError(f"Invalid depth backend: {depth_backend}")

    print("=" * 70)
    print("  WorldQuant BRAIN — Automatic Alpha Discovery System")
    print("=" * 70)
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Max rounds: {MAX_ROUNDS}")
    print(f"  Max candidates per round: {MAX_CANDIDATES_PER_ROUND}")
    print(f"  Depth backend: {depth_backend}")
    print(f"  Skill dir: {SKILL_DIR}")
    print("=" * 70)

    # Load state
    state = load_state()
    lessons = load_lessons()
    db = load_alpha_db()
    reg = load_papers_registry()
    depth_response_status = consume_depth_response(reg)

    if depth_backend == "handoff" and depth_response_status == "blocked":
        print("\n[depth] A depth_response.json file exists but could not be safely consumed.")
        print(f"  Response: {DEPTH_RESPONSE_PATH}")
        print("  Fix or remove the response file before rerunning mining_loop.py.")
        return

    if depth_backend == "handoff" and DEPTH_REQUEST_PATH.exists() and not DEPTH_RESPONSE_PATH.exists():
        print("\n[depth] Existing handoff request is pending.")
        print(f"  Request:  {DEPTH_REQUEST_PATH}")
        print(f"  Response: {DEPTH_RESPONSE_PATH}")
        print("  Ask the outer Agent/subagent to process the request, then rerun mining_loop.py.")
        return

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

                if dry_run:
                    if depth_backend == "none":
                        print("[dry-run] Depth backend disabled; would not read paper.")
                        round_data["depth_triggered"] = False
                    elif depth_backend == "handoff":
                        print(f"[dry-run] Would create depth handoff request for {next_paper}")
                        round_data["depth_triggered"] = True
                    else:
                        print(f"[dry-run] Would extract depth source {next_paper} using backend={depth_backend}")
                        round_data["depth_triggered"] = True
                    round_data["depth_backend"] = depth_backend
                    round_data["paper_read"] = next_paper
                    state["rounds"].append(round_data)
                    save_state(state)
                    return

                if depth_backend == "handoff":
                    request = create_depth_request(
                        next_paper,
                        reg,
                        lessons,
                        reason="candidate_pool_empty",
                    )
                    round_data["depth_triggered"] = True
                    round_data["depth_backend"] = "handoff"
                    round_data["paper_read"] = next_paper
                    round_data["depth_request"] = str(DEPTH_REQUEST_PATH)
                    state["rounds"].append(round_data)
                    save_state(state)
                    print("\n[depth] Handoff request created.")
                    print(f"  Source:   {request['source_id']} — {request['paper']['title']}")
                    print(f"  Request:  {DEPTH_REQUEST_PATH}")
                    print(f"  Response: {DEPTH_RESPONSE_PATH}")
                    print("  Process this request with the outer Agent/subagent, then rerun mining_loop.py.")
                    return

                if depth_backend == "none":
                    print("[depth] Depth backend disabled; skipping paper extraction.")
                    round_data["depth_triggered"] = False
                    round_data["depth_backend"] = "none"
                    state["rounds"].append(round_data)
                    save_state(state)
                    return

                # Try Agent CLI first
                if depth_backend == "claude" and agent_failures < MAX_AGENT_FAILURES:
                    try:
                        fueled = fuel_one_paper(next_paper, reg, lessons)
                    except Exception as e:
                        print(f"  [depth] Agent exception: {e}")
                        agent_failures += 1

                # Fallback to manual
                if depth_backend == "manual" or (not fueled and agent_failures >= MAX_AGENT_FAILURES):
                    print(f"\n[depth] Agent failed {agent_failures} times. Falling back to manual extraction.")
                    fuel_one_paper_manual(next_paper, reg, lessons)

                round_data["depth_triggered"] = True
                round_data["depth_backend"] = depth_backend
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
    parser.add_argument(
        "--depth-backend",
        choices=sorted(DEPTH_BACKENDS),
        default="handoff",
        help="Depth extraction backend. handoff creates depth_request.json for the outer Agent/subagent.",
    )
    args = parser.parse_args()

    if args.reset_state and STATE_PATH.exists():
        STATE_PATH.unlink()
        print("[init] Mining state reset.")

    run_mining_loop(
        max_rounds=args.max_rounds,
        dry_run=args.dry_run,
        depth_backend=args.depth_backend,
    )


if __name__ == "__main__":
    main()
