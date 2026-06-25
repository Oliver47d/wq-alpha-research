#!/usr/bin/env python3
"""llm_producer.py — Minimal LLM-driven candidate producer (prototype).

This is an alternative *producer* for the existing **candidate seam**:

    {"expression": <FASTEXPR str>, "settings": {...}, ...}

The rest of the pipeline (brain_api.simulate / batch_simulate_stream /
compute_correlation / quality_filter / submit_alpha / update_lessons_from_result)
consumes candidates and does NOT care whether the expression came from a
template grid (generate_candidates.expand_template) or from an LLM. So this file
can drop in next to `expand_template` without touching the downstream.

Design (mirrors the project's existing depth handoff, see mining_loop.py):
  1. build_generation_request(...)  -> write llm_request.json describing the
     task + field menu + prior lessons. An outer agent/LLM fills llm_response.json.
     NOTE: we deliberately do NOT call any model endpoint from here. The project
     already uses this file-handoff pattern for its depth phase, and direct model
     calls from mining code are out of scope for a producer prototype.
  2. ExpressionValidator                -> FASTEXPR sanity: balanced parens,
     known operators, and every field token exists in the BRAIN reference.
  3. concept_signature(expr)            -> a stable "concept" key (sorted
     operators + fields) to replace template_id in the lessons feedback loop
     and to dedup near-identical LLM output.
  4. to_candidates(items)               -> validate each item and emit drop-in
     candidate dicts.

CLI:
  # 1) emit a request for the LLM to fill
  python3 scripts/llm_producer.py request --n 8 --out llm_request.json
  # 2) after the LLM writes llm_response.json, turn it into candidates
  python3 scripts/llm_producer.py build --response llm_response.json --out candidates.json
  # quick self-test with a built-in sample (no LLM needed)
  python3 scripts/llm_producer.py selftest
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse existing infrastructure rather than reinventing it.
from generate_candidates import (  # noqa: E402
    FieldValidator,
    deduplicate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
FIELDS_PATH = SKILL_DIR / "references" / "wq_usa_top3000_delay1_data_fields.json"
LESSONS_PATH = SKILL_DIR / "lessons.json"

DEFAULT_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 4,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "nanHandling": "ON",
    "unitHandling": "VERIFY",
}

# --------------------------------------------------------------------------- #
# FASTEXPR knowledge: operators and price/volume builtins.
# Kept intentionally small + explicit; extend as needed.
# --------------------------------------------------------------------------- #
KNOWN_OPERATORS = {
    # cross-sectional
    "rank", "group_rank", "group_mean", "group_neutralize", "group_zscore",
    "zscore", "scale", "winsorize", "normalize", "quantile",
    # time-series
    "ts_rank", "ts_mean", "ts_std_dev", "ts_delta", "ts_delay", "ts_sum",
    "ts_max", "ts_min", "ts_corr", "ts_covariance", "ts_regression",
    "ts_decay_linear", "ts_count", "ts_zscore", "ts_product", "ts_argmax",
    "ts_argmin", "ts_scale",
    # arithmetic / logic
    "abs", "log", "sign", "power", "sqrt", "max", "min", "if_else",
    "add", "subtract", "multiply", "divide",
}

# Price/volume + grouping builtins that are valid tokens but are NOT in the
# data-field reference (which only lists fundamental/analyst/etc. fields).
PRICE_VOLUME_BUILTINS = {
    "close", "open", "high", "low", "vwap", "volume", "returns", "cap",
    "sharesout", "adv20", "adv60", "adv120",
    "industry", "subindustry", "sector", "market",
}

# Numeric-literal / noise tokens to skip during field validation.
SKIP_TOKENS = KNOWN_OPERATORS | PRICE_VOLUME_BUILTINS | {
    "true", "false", "na", "inf", "nan",
}


# --------------------------------------------------------------------------- #
# Expression validation
# --------------------------------------------------------------------------- #
class ExpressionValidator:
    """Layered validation for a raw LLM-produced FASTEXPR expression."""

    def __init__(self, field_validator: FieldValidator | None = None):
        self.fv = field_validator or FieldValidator(FIELDS_PATH)

    def _balanced(self, expr: str) -> bool:
        depth = 0
        for ch in expr:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    def _unknown_function_calls(self, expr: str) -> list[str]:
        """Any `name(` where name is not a known operator is suspicious."""
        called = re.findall(r"([a-z_][a-z0-9_]*)\s*\(", expr.lower())
        return sorted({c for c in called if c not in KNOWN_OPERATORS})

    def _identifiers(self, expr: str) -> list[str]:
        """Identifier tokens that are NOT immediately followed by '(' (i.e. not
        function calls) — candidate field names."""
        # Strip numeric literals first (incl. scientific notation like 1e-6),
        # otherwise the 'e' in '1e-6' is mis-tokenized as a field name.
        cleaned = re.sub(r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b", " ", expr)
        ids = []
        for m in re.finditer(r"[a-z_][a-z0-9_]*", cleaned.lower()):
            tok = m.group(0)
            after = cleaned[m.end():m.end() + 1]
            if after.lstrip().startswith("("):
                continue  # it's a function call, handled separately
            ids.append(tok)
        return ids

    def validate(self, expr: str) -> tuple[bool, list[str]]:
        """Return (ok, errors)."""
        errors: list[str] = []
        if not expr or not expr.strip():
            return False, ["empty expression"]
        if not self._balanced(expr):
            errors.append("unbalanced parentheses")
        unknown_fns = self._unknown_function_calls(expr)
        if unknown_fns:
            errors.append(f"unknown operator(s): {unknown_fns}")
        bad_fields = []
        for tok in self._identifiers(expr):
            if tok in SKIP_TOKENS:
                continue
            if tok.isdigit():
                continue
            if not self.fv.is_valid(tok):
                bad_fields.append(tok)
        if bad_fields:
            errors.append(f"unknown field(s): {sorted(set(bad_fields))}")
        return (len(errors) == 0), errors


# --------------------------------------------------------------------------- #
# Concept signature (replaces template_id in the lessons feedback loop)
# --------------------------------------------------------------------------- #
def concept_signature(expr: str) -> str:
    """Stable signature = sorted operators '+' sorted non-builtin fields.

    Lets the lessons loop aggregate LLM factors by *idea* even though each one
    is a unique string, and dedup near-identical variants.
    """
    low = expr.lower()
    ops = sorted({op for op in KNOWN_OPERATORS if re.search(rf"\b{op}\s*\(", low)})
    cleaned = re.sub(r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b", " ", low)
    ids = set(re.findall(r"[a-z_][a-z0-9_]*", cleaned)) - KNOWN_OPERATORS
    fields = sorted(i for i in ids if not i.isdigit())
    return "ops:" + ",".join(ops) + "|f:" + ",".join(fields)


# --------------------------------------------------------------------------- #
# Request handoff (LLM fills this)
# --------------------------------------------------------------------------- #
def _field_menu(fv: FieldValidator, limit: int = 60) -> list[str]:
    """A small, high-signal slice of the field universe for the LLM prompt."""
    return fv.field_list[:limit]


def build_generation_request(
    lessons: dict[str, Any],
    n: int = 8,
    fields_path: Path | None = None,
) -> dict[str, Any]:
    fv = FieldValidator(fields_path or FIELDS_PATH)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "Generate WorldQuant BRAIN FASTEXPR alpha expressions directly (no templates).",
        "n_requested": n,
        "rules": [
            "Each expression must be valid FASTEXPR for USA TOP3000, delay 1.",
            f"Use ONLY these operators: {sorted(KNOWN_OPERATORS)}.",
            "Fields must come from price/volume builtins or fields_menu (full list in references).",
            f"Price/volume builtins: {sorted(PRICE_VOLUME_BUILTINS)}.",
            "Prefer cross-sectional neutralization (rank/group_rank) for stationarity.",
            "Avoid reusing concepts marked 'deprioritize' in lessons_summary.",
        ],
        "fields_menu_sample": _field_menu(fv),
        "fields_reference_path": str(fields_path or FIELDS_PATH),
        "lessons_summary": {
            "concepts": lessons.get("concepts", lessons.get("patterns", {})),
            "param_insights": lessons.get("param_insights", {}),
        },
        "response_contract": {
            "write_to": "llm_response.json",
            "schema": {
                "status": "DONE",
                "items": [
                    {
                        "expression": "<FASTEXPR string>",
                        "hypothesis": "<why this should have edge>",
                        "settings": "<optional overrides, e.g. {'decay': 8}>",
                    }
                ],
            },
        },
    }


# --------------------------------------------------------------------------- #
# Build candidates from an LLM response
# --------------------------------------------------------------------------- #
def to_candidates(
    items: list[dict[str, Any]],
    validator: ExpressionValidator | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate LLM items and emit drop-in candidates.

    Returns (candidates, rejected) where each rejected entry carries its errors.
    """
    validator = validator or ExpressionValidator()
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in items:
        expr = str(item.get("expression", "")).strip()
        ok, errors = validator.validate(expr)
        if not ok:
            rejected.append({"expression": expr, "errors": errors})
            continue
        settings = {**DEFAULT_SETTINGS, **(item.get("settings") or {})}
        sig = concept_signature(expr)
        candidates.append({
            "expression": expr,
            "settings": settings,
            # `template_id` kept for downstream compatibility (lessons key);
            # here it holds the concept signature instead of a template name.
            "template_id": sig,
            "concept_id": sig,
            "source": "llm",
            "hypothesis": item.get("hypothesis", ""),
            "params": {
                "decay": settings.get("decay"),
                "neutralization": settings.get("neutralization"),
            },
        })
    return deduplicate(candidates), rejected


# --------------------------------------------------------------------------- #
# Built-in sample (for selftest — no LLM round trip needed)
# --------------------------------------------------------------------------- #
SAMPLE_LLM_ITEMS = [
    {
        "expression": "group_rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 1e-6), subindustry)",
        "hypothesis": "Risk-adjusted momentum, industry-neutral.",
        "settings": {"decay": 8},
    },
    {
        "expression": "rank(-ts_delta(close, 5) / close)",
        "hypothesis": "Short-term price reversal.",
    },
    {
        "expression": "group_rank(operating_income / equity, industry)",
        "hypothesis": "Profitability cross-section.",
    },
    {
        # invalid: unknown operator 'magic_smooth'
        "expression": "magic_smooth(returns, 10)",
        "hypothesis": "should be rejected",
    },
    {
        # invalid: unknown field 'totally_made_up_field'
        "expression": "rank(totally_made_up_field / close)",
        "hypothesis": "should be rejected",
    },
    {
        # invalid: unbalanced parens
        "expression": "rank(ts_mean(returns, 10)",
        "hypothesis": "should be rejected",
    },
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_lessons() -> dict[str, Any]:
    if LESSONS_PATH.exists():
        try:
            return json.loads(LESSONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal LLM candidate producer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_req = sub.add_parser("request", help="Write llm_request.json for the LLM to fill")
    p_req.add_argument("--n", type=int, default=8)
    p_req.add_argument("--out", type=str, default=str(SKILL_DIR / "llm_request.json"))

    p_build = sub.add_parser("build", help="Turn llm_response.json into candidates")
    p_build.add_argument("--response", type=str, default=str(SKILL_DIR / "llm_response.json"))
    p_build.add_argument("--out", type=str, default=str(SKILL_DIR / "candidates.json"))

    sub.add_parser("selftest", help="Run built-in sample through the producer")

    args = parser.parse_args()

    if args.cmd == "request":
        req = build_generation_request(_load_lessons(), n=args.n)
        Path(args.out).write_text(json.dumps(req, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote generation request to {args.out} (n={args.n})")
        return

    if args.cmd == "build":
        resp = json.loads(Path(args.response).read_text(encoding="utf-8"))
        items = resp.get("items", [])
        cands, rejected = to_candidates(items)
        Path(args.out).write_text(json.dumps(cands, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"Accepted {len(cands)} candidate(s), rejected {len(rejected)}.")
        for r in rejected:
            print(f"  [reject] {r['expression'][:60]!r}: {r['errors']}")
        print(f"Written to {args.out}")
        return

    if args.cmd == "selftest":
        print("Loading field reference for validation...")
        cands, rejected = to_candidates(SAMPLE_LLM_ITEMS)
        print(f"\nAccepted {len(cands)} / rejected {len(rejected)} (expect 3 / 3)\n")
        for c in cands:
            print(f"  [ok] {c['expression']}")
            print(f"        concept_id = {c['concept_id']}")
            print(f"        decay={c['settings']['decay']} neut={c['settings']['neutralization']}")
        print()
        for r in rejected:
            print(f"  [reject] {r['expression'][:55]!r}\n           -> {r['errors']}")
        return


if __name__ == "__main__":
    main()
