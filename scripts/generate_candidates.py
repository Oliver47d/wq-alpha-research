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
        # field id -> category id (e.g. "operating_income" -> "fundamental").
        # Used by structure_fingerprint() to classify fields by data category.
        self.field_categories: dict[str, str] = {}
        if fields_path and fields_path.exists():
            data = json.loads(fields_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Each item is a dict with an 'id' key, e.g. {"id": "operating_income", "alphaCount": 200, ...}
                self.field_list = [f["id"] for f in data if isinstance(f, dict) and "id" in f]
                self.field_categories = {
                    f["id"]: (f.get("category") or {}).get("id", "unknown")
                    for f in data
                    if isinstance(f, dict) and "id" in f and isinstance(f.get("category"), dict)
                }
            elif isinstance(data, dict):
                # Could be {"fields": [...]} or {"id": [...]} or flat {"field_name": {...}}
                for v in data.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        self.field_list = [item["id"] for item in v if isinstance(item, dict) and "id" in item]
                        self.field_categories = {
                            item["id"]: (item.get("category") or {}).get("id", "unknown")
                            for item in v
                            if isinstance(item, dict) and "id" in item and isinstance(item.get("category"), dict)
                        }
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
        # Remove template placeholders (e.g. {numerator}, {vwap_dev}) before validation
        # so we only validate actual field names used in the final FASTEXPR.
        expr = re.sub(r'\{[^}]+\}', '', expr)
        # Skip operators and functions, extract potential field names
        # Field names are typically lowercase with underscores and digits
        tokens = re.findall(r'[a-z][a-z0-9_]+', expr.lower())
        # Derive the skip set from the single sources of truth so the operator
        # list can never drift out of sync (a hand-maintained copy here used to
        # miss ts_count / if_else / group_mean etc., silently killing every
        # candidate of the templates that used them).
        skip = (
            KNOWN_OPERATORS
            | GROUP_BUILTINS
            | PRICE_VOLUME_BUILTINS
            | {"trade", "halt", "na", "inf", "true", "false"}
        )
        all_valid = True
        for token in tokens:
            if token in skip:
                continue
            if not self.is_valid(token):
                all_valid = False
        return all_valid


# --------------------------------------------------------------------------- #
# Shared FASTEXPR knowledge (single source of truth for both producers).
# llm_producer imports these so the two paths can never drift apart.
# --------------------------------------------------------------------------- #
KNOWN_OPERATORS = {
    # cross-sectional
    "rank", "group_rank", "group_mean", "group_neutralize", "group_zscore",
    "zscore", "scale", "winsorize", "normalize", "quantile",
    # time-series
    "ts_rank", "ts_mean", "ts_std_dev", "ts_delta", "ts_delay", "ts_sum",
    "ts_corr", "ts_covariance", "ts_regression",
    "ts_decay_linear", "ts_zscore", "ts_product", "ts_arg_max",
    "ts_arg_min", "ts_scale",
    "ts_backfill", "ts_av_diff", "ts_count_nans", "ts_quantile",
    "last_diff_value", "days_from_last_change",
    # arithmetic / logic
    "abs", "log", "sign", "power", "sqrt", "max", "min", "if_else",
    "add", "subtract", "multiply", "divide",
    "signed_power", "inverse", "reverse", "hump",
    "group_scale", "trade_when",
}

# Price/volume + grouping builtins: valid tokens that are NOT in the data-field
# reference (which only lists fundamental/analyst/etc. fields). For field-class
# classification these all map to the "pv" category.
PRICE_VOLUME_BUILTINS = {
    "close", "open", "high", "low", "vwap", "volume", "returns", "cap",
    "sharesout", "adv20", "adv60", "adv120",
}
GROUP_BUILTINS = {"industry", "subindustry", "sector", "market"}


# --------------------------------------------------------------------------- #
# Structure fingerprint — the v2 lessons aggregation key.
#
# A factor's *idea* is its operator/field structure, not its exact string. Two
# expressions that differ only in window sizes or which concrete pv field they
# use share a structure and should aggregate together in the experience log.
# Shared by both producers (template grid + LLM) so lessons accumulate across
# both paths under one key space.
# --------------------------------------------------------------------------- #
def _max_paren_depth(expr: str) -> int:
    depth = 0
    best = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            best = max(best, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return best


def _expr_operators(expr: str) -> list[str]:
    """Sorted list of known operators actually called in the expression."""
    low = expr.lower()
    return sorted({op for op in KNOWN_OPERATORS if re.search(rf"\b{op}\s*\(", low)})


def _expr_fields(expr: str) -> list[str]:
    """Identifier tokens that are NOT function calls and NOT group builtins —
    i.e. the data fields (incl. pv builtins) the expression reads."""
    # Strip numeric literals (incl. scientific notation) so the 'e' in 1e-6
    # is not mistaken for a field token.
    cleaned = re.sub(r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b", " ", expr.lower())
    fields: list[str] = []
    for m in re.finditer(r"[a-z_][a-z0-9_]*", cleaned):
        tok = m.group(0)
        after = cleaned[m.end():m.end() + 1]
        if after.lstrip().startswith("("):
            continue  # function call
        if tok in KNOWN_OPERATORS or tok in GROUP_BUILTINS:
            continue
        fields.append(tok)
    return sorted(set(fields))


def structure_fingerprint(
    expr: str,
    field_categories: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a structural fingerprint of a FASTEXPR expression.

    Keys:
      * ast_hash      — short hash of the normalized skeleton (every field token
                        replaced by 'F', every numeric literal by 'N'). Stable
                        across window/field swaps; the primary rollup key.
      * ops           — sorted operators used.
      * fields        — sorted concrete fields read (incl. pv builtins).
      * field_classes — sorted data categories of those fields (pv builtins ->
                        "pv"; reference fields -> their category id; unknown ->
                        "unknown"). The cross-idea generalization key.
      * depth         — max parenthesis nesting (rough structural complexity).
    """
    field_categories = field_categories or {}
    low = expr.lower()

    # Normalized skeleton: numbers -> N, then field identifiers -> F. Operators
    # and group builtins are preserved so the skeleton still encodes structure.
    skel = re.sub(r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b", "N", low)
    fields = _expr_fields(expr)

    def _classify(f: str) -> str:
        if f in PRICE_VOLUME_BUILTINS:
            return "pv"
        return field_categories.get(f, "unknown")

    # Replace whole-word field tokens with F (longest first to avoid partials).
    for f in sorted(fields, key=len, reverse=True):
        skel = re.sub(rf"\b{re.escape(f)}\b", "F", skel)
    # Collapse whitespace so trivial spacing differences don't change the hash.
    skel_norm = re.sub(r"\s+", "", skel)

    import hashlib
    ast_hash = hashlib.sha1(skel_norm.encode("utf-8")).hexdigest()[:12]

    field_classes = sorted({_classify(f) for f in fields}) if fields else []

    return {
        "ast_hash": ast_hash,
        "ops": _expr_operators(expr),
        "fields": fields,
        "field_classes": field_classes,
        "depth": _max_paren_depth(expr),
    }


# --------------------------------------------------------------------------- #
# Template expansion
# --------------------------------------------------------------------------- #
def load_templates(templates_dir: Path | None = None, strict: bool = True) -> list[dict[str, Any]]:
    tdir = templates_dir or TEMPLATES_DIR
    if not tdir.exists():
        return []
    templates = []
    errors: list[str] = []
    for fp in sorted(tdir.glob("*.json")):
        try:
            tpl = json.loads(fp.read_text(encoding="utf-8"))
            tpl["_filename"] = fp.name
            templates.append(tpl)
        except json.JSONDecodeError as e:
            # A corrupt template means a paper's extracted factor never gets
            # mined. Do NOT swallow this silently — surface it loudly.
            msg = f"{fp.name}: {e}"
            errors.append(msg)
            print(f"  [ERROR] Invalid template JSON {msg}", flush=True)
        except Exception as e:
            msg = f"{fp.name}: {e}"
            errors.append(msg)
            print(f"  [ERROR] Failed to load template {msg}", flush=True)
    if errors and strict:
        raise ValueError(
            f"Failed to load {len(errors)} template(s):\n  - "
            + "\n  - ".join(errors)
        )
    return templates


def _fill_skeleton(skeleton: str, replacements: dict[str, str]) -> str:
    expr = skeleton
    for key, val in replacements.items():
        expr = expr.replace(f"{{{key}}}", str(val))
    return expr


def _placeholders(text: str) -> set[str]:
    """Return the set of {placeholder} names in a string."""
    return set(re.findall(r"\{([^}]+)\}", text))


def _signal_expr(fp: dict[str, Any]) -> str | None:
    """Build a signal expression from a field_pair's numerator/denominator.

    Returns None for legacy direct-key field_pairs that have no numerator
    (those are filled by direct key substitution instead).
    """
    num = str(fp.get("numerator", "")).strip()
    if not num:
        return None
    den = str(fp.get("denominator", "1")).strip()
    if den in ("", "1"):
        return num
    return f"({num}) / ({den})"


def _fp_for_slot(slot: str, field_pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Map a skeleton signal slot to the field_pair whose `name` matches.

    Accepts both exact name match ({gap} -> name 'gap') and the common
    `<name>_signal` convention ({ir_signal} -> name 'ir').
    """
    base = slot[:-len("_signal")] if slot.endswith("_signal") else slot
    for fp in field_pairs:
        name = fp.get("name")
        if name == slot or name == base:
            return fp
    return None


def _param_combo_priority(
    combo: dict[str, Any],
    param_insights: dict[str, Any] | None,
) -> tuple[int, float]:
    """Rank a param combo using lessons.param_insights.

    Returns (tier, -avg_sharpe_sum) where lower sorts first:
      tier 0 = at least one `prefer` value and no `deprioritize`,
      tier 1 = neutral / unknown,
      tier 2 = contains a `deprioritize` value.
    Within a tier, combos whose values have higher historical avg_sharpe sort
    first. This lets a later `[:max_candidates]` truncation keep the combos the
    feedback loop likes and drop the ones it dislikes, instead of cutting blind.
    """
    if not param_insights:
        return (1, 0.0)
    has_prefer = False
    has_depri = False
    sharpe_sum = 0.0
    for name, val in combo.items():
        pi = param_insights.get(name)
        if not isinstance(pi, dict):
            continue
        entry = pi.get(str(val))
        if not isinstance(entry, dict):
            continue
        verdict = entry.get("verdict")
        if verdict == "prefer":
            has_prefer = True
        elif verdict == "deprioritize":
            has_depri = True
        sharpe_sum += float(entry.get("avg_sharpe", 0.0) or 0.0)
    if has_depri:
        tier = 2
    elif has_prefer:
        tier = 0
    else:
        tier = 1
    return (tier, -sharpe_sum)


def expand_template(
    template: dict[str, Any],
    max_candidates: int = 20,
    validator: FieldValidator | None = None,
    param_insights: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand a template into a list of simulation candidates.

    Supports two template styles:

    * Legacy direct-key: skeleton placeholders ({numerator}, {denominator},
      {estimate_field}, ...) match field_pair keys directly.
    * Named-signal: skeleton references signal slots ({signal}, {rvs_signal},
      {gap}, ...). Each signal is assembled from a field_pair's
      numerator/denominator. A skeleton with one slot + multiple field_pairs is
      treated as *alternatives* (one candidate per field_pair); a skeleton with
      >=2 distinct slots is *combined* (all field_pairs merged into one expr).

    Each candidate: {expression, settings, template_id, field_pair, params}
    """
    skeleton = template.get("skeleton", "")
    template_id = template.get("template_id", template.get("_filename", "unknown"))
    field_pairs = template.get("field_pairs", [])
    param_ranges = template.get("param_ranges", {})
    default_settings = template.get("default_settings", {})

    if not skeleton or not field_pairs:
        return []

    # Split param_ranges into a grid (list values) and derived params (string
    # templates that reference other params, e.g. est_up = "ts_delta({est_field}, {window})").
    grid_params = {k: v for k, v in param_ranges.items() if isinstance(v, list)}
    derived_params = {k: v for k, v in param_ranges.items() if isinstance(v, str)}

    # Build the cartesian product of grid params.
    param_combos: list[dict[str, Any]] = [{}]
    for key, values in grid_params.items():
        param_combos = [{**combo, key: val} for combo in param_combos for val in values]

    # #5: order combos by lessons.param_insights so a later truncation keeps
    # the preferred params and drops the deprioritized ones (was blind cut).
    if param_insights:
        param_combos.sort(key=lambda c: _param_combo_priority(c, param_insights))

    # Identify signal slots: skeleton placeholders that are neither grid/derived
    # params nor direct field_pair keys.
    fp_direct_keys: set[str] = set()
    for fp in field_pairs:
        fp_direct_keys |= set(fp.keys())
    sk_ph = _placeholders(skeleton)
    signal_slots = [p for p in sk_ph if p not in param_ranges and p not in fp_direct_keys]

    def _finalize(expr: str, pc: dict[str, Any]) -> str:
        # Resolve derived params using the current param combo first, then fill.
        rep = dict(pc)
        for dk, dtmpl in derived_params.items():
            rep[dk] = _fill_skeleton(str(dtmpl), pc)
        expr = _fill_skeleton(expr, rep)
        # Second pass: derived params may introduce further param references.
        expr = _fill_skeleton(expr, rep)
        return expr

    def _settings_for(pc: dict[str, Any]) -> dict[str, Any]:
        settings = dict(default_settings)
        if "decay" in pc:
            settings["decay"] = int(pc["decay"]) if str(pc["decay"]).isdigit() else pc["decay"]
        if "neutralization" in pc:
            settings["neutralization"] = pc["neutralization"]
        if "delay" in pc:
            settings["delay"] = int(pc["delay"])
        if "truncation" in pc:
            settings["truncation"] = pc["truncation"]
        return settings

    candidates: list[dict[str, Any]] = []

    # Decide combined vs alternatives mode.
    combined = False
    slot_map: dict[str, dict[str, Any]] = {}
    if len(signal_slots) >= 2:
        slot_map = {s: _fp_for_slot(s, field_pairs) for s in signal_slots}
        combined = all(v is not None for v in slot_map.values())

    if combined:
        # Fill every slot with its mapped field_pair's signal expression.
        for pc in param_combos:
            if len(candidates) >= max_candidates:
                break
            expr = skeleton
            for slot, fp in slot_map.items():
                expr = expr.replace(f"{{{slot}}}", _signal_expr(fp) or "")
            expr = _finalize(expr, pc)
            if validator and not validator.validate_expression(expr):
                continue
            candidates.append({
                "expression": expr,
                "settings": _settings_for(pc),
                "template_id": template_id,
                "field_pair": [fp.get("name") for fp in field_pairs],
                "params": pc,
            })
        return candidates

    # Alternatives / legacy direct-key mode: each field_pair yields candidates.
    for fp in field_pairs:
        sig = _signal_expr(fp)
        for pc in param_combos:
            if len(candidates) >= max_candidates:
                break
            expr = skeleton
            if signal_slots and sig is not None:
                for slot in signal_slots:
                    expr = expr.replace(f"{{{slot}}}", sig)
            # Legacy direct-key fill (numerator/denominator/estimate_field/...).
            expr = _fill_skeleton(
                expr,
                {k: v for k, v in fp.items() if k not in ("name", "description")},
            )
            expr = _finalize(expr, pc)
            if validator and not validator.validate_expression(expr):
                continue
            candidates.append({
                "expression": expr,
                "settings": _settings_for(pc),
                "template_id": template_id,
                "field_pair": fp,
                "params": pc,
            })
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
