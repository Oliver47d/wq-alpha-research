#!/usr/bin/env python3
"""factor_seeds.py — build the GP initial population from real factors (#21 B).

The GP engine (factor_gp.py) mutates/crosses typed AST Nodes. It needs a
starting population. The richest, most legal, most *diverse* seed source is not
the template grid — it's the factors we have already simulated: alpha_db.json
holds 166 real BRAIN factors, each carrying sharpe/fitness/turnover. Those metrics
are exactly what a later multi-objective fitness wants, so seeding from them also
warm-starts the fitness landscape.

Escaping template homogeneity is GP's job (breeding + low-corr fitness), NOT the
seed's — so a template-shaped seed is fine. Seeds only need: (1) high legality,
(2) diverse parts. This module supplies both.

Public API:
    load_seeds(...)  -> list[Seed]   parse real factors into typed-AST seeds
    Seed             dataclass: tree (Node) + provenance + metrics + ast_hash

Every seed is parsed through factor_ast.from_fastexpr, so it is legal BY the same
type check the GP operators preserve. Unparseable factors (e.g. ones with a
leading /* HYPOTHESIS */ comment block) are stripped first; if still unparseable
they are skipped, never guessed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any, Optional

from factor_ast import Node, from_fastexpr, to_fastexpr, ParseError
from generate_candidates import FieldValidator, structure_fingerprint

SKILL_DIR = Path(__file__).resolve().parent.parent
ALPHA_DB_PATH = SKILL_DIR / "alpha_db.json"
FIELDS_PATH = SKILL_DIR / "references" / "wq_usa_top3000_delay1_data_fields.json"


@dataclass
class Seed:
    """One initial-population member: a legal AST plus where it came from."""
    tree: Node
    expr: str                       # canonical FASTEXPR (from to_fastexpr(tree))
    ast_hash: str                   # structure fingerprint (dedup / lessons key)
    source: str                     # 'alpha_db' | 'template' | ...
    origin_id: str = ""             # alpha id / template id
    metrics: dict[str, Any] = _dc_field(default_factory=dict)  # sharpe/fitness/turnover/status


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #
def _expr_text(expression: Any) -> str:
    """Remote BRAIN records store the formula as {'code': ...}; local ones as a
    bare string. Mirror mining_loop._expr_text so both shapes work."""
    if isinstance(expression, dict):
        return str(expression.get("code") or "")
    if expression is None:
        return ""
    return str(expression)


def _strip_comments(expr: str) -> str:
    """Drop C-style /* ... */ and #-line comments BRAIN allows but our AST's
    Python-based lexer rejects. Purely a de-noise step — no semantics touched."""
    expr = re.sub(r"/\*.*?\*/", " ", expr, flags=re.DOTALL)
    expr = re.sub(r"#.*$", " ", expr, flags=re.MULTILINE)
    return expr.strip()


# --------------------------------------------------------------------------- #
# Seed construction
# --------------------------------------------------------------------------- #
def _make_seed(
    raw_expr: str, source: str, origin_id: str, metrics: dict,
    fv: Optional[FieldValidator], fcats: dict[str, str],
) -> Optional[Seed]:
    """Parse one expression into a Seed, or None if it can't be made legal."""
    expr = _strip_comments(_expr_text(raw_expr))
    if not expr:
        return None
    try:
        tree = from_fastexpr(expr, fv=fv)
    except ParseError:
        return None
    canonical = to_fastexpr(tree)              # normalize (parens/decimals)
    ast_hash = structure_fingerprint(canonical, fcats)["ast_hash"]
    return Seed(tree=tree, expr=canonical, ast_hash=ast_hash,
                source=source, origin_id=origin_id, metrics=metrics)


def _seeds_from_alpha_db(
    path: Path, fv: Optional[FieldValidator], fcats: dict[str, str],
) -> tuple[list[Seed], int]:
    """Real simulated factors. Returns (seeds, skipped_count)."""
    if not path.exists():
        return [], 0
    db = json.loads(path.read_text(encoding="utf-8")).get("alphas", {})
    seeds, skipped = [], 0
    for aid, rec in db.items():
        metrics = {k: rec.get(k) for k in ("status", "sharpe", "fitness", "turnover")}
        s = _make_seed(rec.get("expression"), "alpha_db", aid, metrics, fv, fcats)
        if s:
            seeds.append(s)
        else:
            skipped += 1
    return seeds, skipped


def load_seeds(
    alpha_db_path: Path = ALPHA_DB_PATH,
    fields_path: Path = FIELDS_PATH,
    dedup: bool = True,
) -> list[Seed]:
    """Build the GP initial population from real factors.

    Currently sources alpha_db.json (166 simulated factors). Deduplicates by the
    canonical expression string (keeps structural diversity while collapsing
    exact repeats). Order is preserved so callers can slice a subset.
    """
    fv = FieldValidator(fields_path) if fields_path.exists() else None
    fcats = fv.field_categories if fv else {}
    seeds, _skipped = _seeds_from_alpha_db(alpha_db_path, fv, fcats)
    if dedup:
        seen: set[str] = set()
        unique: list[Seed] = []
        for s in seeds:
            if s.expr in seen:
                continue
            seen.add(s.expr)
            unique.append(s)
        seeds = unique
    return seeds


if __name__ == "__main__":
    pop = load_seeds()
    hashes = {s.ast_hash for s in pop}
    print(f"seeds loaded: {len(pop)}   distinct ast_hash: {len(hashes)}")
    for s in sorted(pop, key=lambda x: -(x.metrics.get("sharpe") or 0))[:5]:
        print(f"  sharpe={s.metrics.get('sharpe')} [{s.metrics.get('status')}] {s.expr[:70]}")
