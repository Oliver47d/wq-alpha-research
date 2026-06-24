#!/usr/bin/env python3
"""generate_candidates.py — Breadth engine: expand templates into candidate alphas.

Loads template JSONs from templates/, expands field_pairs × param_ranges,
and produces a deduplicated list of simulation-ready candidates.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
TEMPLATES_DIR = SKILL_DIR / "templates"
FIELDS_PATH = SKILL_DIR / "references" / "wq_usa_top3000_delay1_data_fields.json"


# --------------------------------------------------------------------------- #
# Field Validator (two-layer)
# --------------------------------------------------------------------------- #
class FieldValidator:
    """Validate field names against the BRAIN data field reference.

    Layer 1: exact match.
    Layer 2: fuzzy match (substring) to suggest alternatives.
    """

    def __init__(self, fields_path: Path | None = None):
        self.valid_fields: set[str] = set()
        self.field_list: list[str] = []
        if fields_path and fields_path.exists():
            data = json.loads(fields_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Each item is a dict with an 'id' key, e.g. {"id": "operating_income", "alphaCount": 200, ...}
                self.field_list = [f["id"] for f in data if isinstance(f, dict) and "id" in f]
            elif isinstance(data, dict):
                # Could be {"fields": [...]} or {"id": [...]} or flat {"field_name": {...}}
                for v in data.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        self.field_list = [item["id"] for item in v if isinstance(item, dict) and "id" in item]
                        break
                if not self.field_list:
                    # Flat dict: keys are field names
                    self.field_list = [k for k, v in data.items() if not isinstance(v, dict)]
                    if not self.field_list:
                        self.field_list = list(data.keys())
            self.valid_fields = set(self.field_list)
        self._checked: dict[str, bool] = {}

    def is_valid(self, field: str) -> bool:
        if field in self._checked:
            return self._checked[field]
        ok = field in self.valid_fields
        self._checked[field] = ok
        if not ok:
            suggestions = self.suggest(field)
            if suggestions:
                print(f"  [field] '{field}' not found. Suggestions: {suggestions[:5]}", flush=True)
            else:
                print(f"  [field] '{field}' not found in field reference.", flush=True)
        return ok

    def suggest(self, field: str, limit: int = 5) -> list[str]:
        field_lower = field.lower()
        matches = [f for f in self.field_list if field_lower in f.lower()]
        return matches[:limit]

    def validate_expression(self, expr: str) -> bool:
        """Extract field-like tokens from expression and validate them."""
        # Skip operators and functions, extract potential field names
        # Field names are typically lowercase with underscores and digits
        tokens = re.findall(r'[a-z][a-z0-9_]+', expr.lower())
        skip = {
            "group_rank", "ts_rank", "rank", "ts_mean", "ts_std_dev",
            "ts_delta", "ts_delay", "ts_sum", "ts_max", "ts_min",
            "ts_regression", "ts_corr", "ts_covariance",
            "subindustry", "industry", "sector", "market",
            "close", "open", "high", "low", "vwap", "volume", "returns",
            "cap", "sharesout", "adv20", "adv60", "adv120",
            "trade", "halt", "na", "inf", "true", "false",
        }
        all_valid = True
        for token in tokens:
            if token in skip:
                continue
            if not self.is_valid(token):
                all_valid = False
        return all_valid


# --------------------------------------------------------------------------- #
# Template expansion
# --------------------------------------------------------------------------- #
def load_templates(templates_dir: Path | None = None) -> list[dict[str, Any]]:
    tdir = templates_dir or TEMPLATES_DIR
    if not tdir.exists():
        return []
    templates = []
    for fp in sorted(tdir.glob("*.json")):
        try:
            tpl = json.loads(fp.read_text(encoding="utf-8"))
            tpl["_filename"] = fp.name
            templates.append(tpl)
        except Exception as e:
            print(f"  [warn] Failed to load {fp.name}: {e}", flush=True)
    return templates


def _fill_skeleton(skeleton: str, replacements: dict[str, str]) -> str:
    expr = skeleton
    for key, val in replacements.items():
        expr = expr.replace(f"{{{key}}}", str(val))
    return expr


def expand_template(
    template: dict[str, Any],
    max_candidates: int = 20,
    validator: FieldValidator | None = None,
) -> list[dict[str, Any]]:
    """Expand a template into a list of simulation candidates.

    Each candidate: {expression, settings, template_id, field_pair, params}
    """
    skeleton = template.get("skeleton", "")
    template_id = template.get("template_id", template.get("_filename", "unknown"))
    field_pairs = template.get("field_pairs", [])
    param_ranges = template.get("param_ranges", {})
    default_settings = template.get("default_settings", {})

    if not skeleton or not field_pairs:
        return []

    # Build param grid
    param_keys = list(param_ranges.keys())
    param_combos = [{}]
    for key in param_keys:
        values = param_ranges[key]
        new_combos = []
        for combo in param_combos:
            for val in values:
                new_combos.append({**combo, key: val})
        param_combos = new_combos

    candidates: list[dict[str, Any]] = []
    for fp in field_pairs:
        for pc in param_combos:
            if len(candidates) >= max_candidates:
                break

            replacements = {**fp, **pc}
            expr = _fill_skeleton(skeleton, replacements)

            # Build settings
            settings = dict(default_settings)
            # Map common param names to settings
            if "decay" in pc:
                settings["decay"] = int(pc["decay"]) if str(pc["decay"]).isdigit() else pc["decay"]
            if "neutralization" in pc:
                settings["neutralization"] = pc["neutralization"]
            if "delay" in pc:
                settings["delay"] = int(pc["delay"])

            # Validate fields if validator provided
            if validator and not validator.validate_expression(expr):
                continue

            candidate = {
                "expression": expr,
                "settings": settings,
                "template_id": template_id,
                "field_pair": fp,
                "params": pc,
            }
            candidates.append(candidate)

        if len(candidates) >= max_candidates:
            break

    return candidates


def deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate expression+settings combos."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for cand in candidates:
        key = cand["expression"] + json.dumps(cand.get("settings", {}), sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate alpha candidates from templates")
    parser.add_argument("--max-per-template", type=int, default=20, help="Max candidates per template")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--validate-fields", action="store_true", help="Validate field names against reference")
    args = parser.parse_args()

    templates = load_templates()
    print(f"Loaded {len(templates)} templates", flush=True)

    validator = FieldValidator(FIELDS_PATH) if args.validate_fields else None
    if validator and validator.field_list:
        print(f"Loaded {len(validator.field_list)} fields for validation", flush=True)

    all_candidates: list[dict[str, Any]] = []
    for tpl in templates:
        tpl_id = tpl.get("template_id", tpl.get("_filename", "?"))
        cands = expand_template(tpl, max_candidates=args.max_per_template, validator=validator)
        print(f"  {tpl_id}: {len(cands)} candidates", flush=True)
        all_candidates.extend(cands)

    before = len(all_candidates)
    all_candidates = deduplicate(all_candidates)
    print(f"Total: {before} candidates ({len(all_candidates)} after dedup)", flush=True)

    output = args.output or str(SKILL_DIR / "candidates.json")
    Path(output).write_text(json.dumps(all_candidates, indent=2, default=str), encoding="utf-8")
    print(f"Written to {output}", flush=True)


if __name__ == "__main__":
    main()
