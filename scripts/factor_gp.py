#!/usr/bin/env python3
"""factor_gp.py — genetic operators over the typed factor AST (#21 engine).

The template/LLM producers cannot escape structural homogeneity because the
factor's root shell is a frozen constant (see BUGS_AND_IMPROVEMENTS.md path A).
Genetic programming mutates the WHOLE tree — including the root — so the shell
itself evolves. This module provides the four genetic operators, built so that
every produced tree is legal BY CONSTRUCTION (no generate-then-reject retry),
which is only possible because factor_ast gives us a type system:

  crossover(a, b)   swap V-typed subtrees   (V never lands in an N/G slot -> safe)
  mutate(tree)      one random legal edit:
      * op      : swap operator for another in the SAME signature class
      * field   : FIELD -> another field of the SAME data category
      * window  : window CONST -> another value from WINDOW_CHOICES
      * const   : plain CONST -> +/-50% jitter (epsilon guards are skipped)
      * subtree : a V node -> a fresh random legal V subtree
  random_v_subtree(...)  grow a type-correct subtree returning V

All operators return NEW trees (inputs are copied, never mutated in place).
`validate(tree, fv)` re-runs the AST type check as a belt-and-suspenders test.
"""
from __future__ import annotations

import random
from typing import Optional

from factor_ast import (
    Node, OP, FIELD, CONST, GROUP,
    OP_SIGNATURES, VIRTUAL_OPS, WINDOW_CHOICES, CONST_JITTER,
    is_epsilon_const, to_fastexpr, from_fastexpr, ParseError,
)
from generate_candidates import (
    KNOWN_OPERATORS, GROUP_BUILTINS, PRICE_VOLUME_BUILTINS, FieldValidator,
)

# --------------------------------------------------------------------------- #
# Precomputed pools
# --------------------------------------------------------------------------- #
# Operators grouped by exact signature (argtypes tuple, rtype). Ops in the same
# class are freely interchangeable — same arity, same slot types, same return.
def _signature_classes() -> dict[tuple, list[str]]:
    classes: dict[tuple, list[str]] = {}
    for op, sig in OP_SIGNATURES.items():
        if op == "cmp":            # cmp returns B, only lives inside if_else
            continue
        classes.setdefault(sig, []).append(op)
    return classes


_SIG_CLASSES = _signature_classes()

# Ops that return V (candidates for random subtree roots). Exclude cmp (B).
_V_OPS = [op for op, (args, rt) in OP_SIGNATURES.items() if rt == "V" and op != "cmp"]

# Comparators for building B nodes (if_else conditions).
_CMP_SYMBOLS = ["<", ">", "<=", ">="]


def _field_pool_by_category(fv: Optional[FieldValidator]) -> dict[str, list[str]]:
    """category id -> list of field ids. pv builtins live under 'pv'.

    GROUP_BUILTINS (sector/industry/market/subindustry) also appear in the field
    reference, but they parse as GROUP (type G), not V. A FIELD node must never
    mutate into one, so they are excluded from every field pool.
    """
    pool: dict[str, list[str]] = {"pv": sorted(PRICE_VOLUME_BUILTINS)}
    if fv is not None:
        for fid, cat in fv.field_categories.items():
            if fid in GROUP_BUILTINS:
                continue
            pool.setdefault(cat or "unknown", []).append(fid)
    return pool


# --------------------------------------------------------------------------- #
# Tree position utilities
# --------------------------------------------------------------------------- #
def _all_positions(root: Node) -> list[tuple[Optional[Node], int, Node]]:
    """List of (parent, index_in_parent, node); root is (None, -1, root)."""
    out: list[tuple[Optional[Node], int, Node]] = [(None, -1, root)]
    stack = [root]
    while stack:
        n = stack.pop()
        for i, c in enumerate(n.children):
            out.append((n, i, c))
            stack.append(c)
    return out


def _slot_type(parent: Optional[Node], idx: int) -> Optional[str]:
    """Expected type of the idx-th child of parent: V/VN/N:win/N:const/G/B."""
    if parent is None or parent.kind != OP:
        return None
    sig = OP_SIGNATURES.get(parent.value)
    if not sig or idx >= len(sig[0]):
        return None
    return sig[0][idx]


# --------------------------------------------------------------------------- #
# Random subtree generation (type-correct by construction)
# --------------------------------------------------------------------------- #
def _rand_field(rng: random.Random, pool: dict[str, list[str]]) -> Node:
    cat = rng.choice([c for c, fs in pool.items() if fs])
    return Node(FIELD, rng.choice(pool[cat]))


def _rand_window(rng: random.Random) -> Node:
    return Node(CONST, rng.choice(WINDOW_CHOICES))


def _rand_const(rng: random.Random) -> Node:
    return Node(CONST, round(rng.uniform(0.5, 3.0), 3))


def _rand_group(rng: random.Random) -> Node:
    return Node(GROUP, rng.choice(sorted(GROUP_BUILTINS)))


def random_v_subtree(
    rng: random.Random,
    field_pool: Optional[dict[str, list[str]]] = None,
    max_depth: int = 3,
    _depth: int = 0,
) -> Node:
    """Grow a legal subtree whose return type is V."""
    pool = field_pool or {"pv": sorted(PRICE_VOLUME_BUILTINS)}
    # Terminate: at max depth, or probabilistically once past the root.
    if _depth >= max_depth or (_depth > 0 and rng.random() < 0.35):
        return _rand_field(rng, pool)
    op = rng.choice(_V_OPS)
    argtypes = OP_SIGNATURES[op][0]
    children: list[Node] = []
    for slot in argtypes:
        if slot == "N:win":
            children.append(_rand_window(rng))
        elif slot == "N:const":
            children.append(_rand_const(rng))
        elif slot == "G":
            children.append(_rand_group(rng))
        elif slot == "B":
            children.append(_rand_bool(rng, pool, max_depth, _depth + 1))
        else:  # V or VN -> a vector subtree
            children.append(random_v_subtree(rng, pool, max_depth, _depth + 1))
    return Node(OP, op, children)


def _rand_bool(rng: random.Random, pool, max_depth: int, depth: int) -> Node:
    left = random_v_subtree(rng, pool, max_depth, depth)
    right = Node(CONST, 0)  # compare against 0 — the common indicator idiom
    n = Node(OP, "cmp", [left, right])
    n.symbol = rng.choice(_CMP_SYMBOLS)  # type: ignore[attr-defined]
    return n


# --------------------------------------------------------------------------- #
# Crossover — swap V-typed subtrees (legal by construction)
# --------------------------------------------------------------------------- #
def crossover(a: Node, b: Node, rng: random.Random) -> Node:
    """Return a copy of `a` with one of its V-subtrees replaced by a V-subtree
    copied from `b`. V nodes never occupy N/G/B slots, so the result is always
    type-legal without any post-hoc rejection.
    """
    ca = a.copy()
    cb = b.copy()
    recipients = [(p, i, n) for (p, i, n) in _all_positions(ca) if n.rtype() == "V"]
    donors = [n for (_, _, n) in _all_positions(cb) if n.rtype() == "V"]
    if not recipients or not donors:
        return ca
    parent, idx, _ = rng.choice(recipients)
    donor = rng.choice(donors).copy()
    if parent is None:
        return donor  # swapped at the root
    parent.children[idx] = donor
    return ca


# --------------------------------------------------------------------------- #
# Mutation — one random legal edit
# --------------------------------------------------------------------------- #
def mutate(
    tree: Node,
    rng: random.Random,
    field_pool: Optional[dict[str, list[str]]] = None,
    max_depth: int = 4,
) -> Node:
    """Apply one random legal mutation and return the new tree."""
    pool = field_pool or {"pv": sorted(PRICE_VOLUME_BUILTINS)}
    t = tree.copy()
    positions = _all_positions(t)

    # Build candidate edit buckets (each guaranteed legal).
    op_nodes = [
        n for (_, _, n) in positions
        if n.kind == OP and n.value in OP_SIGNATURES
        and len(_SIG_CLASSES.get(OP_SIGNATURES[n.value], [])) > 1
    ]
    field_nodes = [n for (_, _, n) in positions if n.kind == FIELD]
    window_consts = [
        n for (p, i, n) in positions
        if n.kind == CONST and _slot_type(p, i) == "N:win"
    ]
    free_consts = [
        n for (p, i, n) in positions
        if n.kind == CONST and _slot_type(p, i) in ("N:const", "VN")
        and not is_epsilon_const(n)
    ]
    v_positions = [(p, i, n) for (p, i, n) in positions if n.rtype() == "V"]

    strategies = []
    if op_nodes:      strategies.append("op")
    if field_nodes:   strategies.append("field")
    if window_consts: strategies.append("window")
    if free_consts:   strategies.append("const")
    if v_positions:   strategies.append("subtree")
    if not strategies:
        return t

    choice = rng.choice(strategies)
    if choice == "op":
        n = rng.choice(op_nodes)
        cls = _SIG_CLASSES[OP_SIGNATURES[n.value]]
        n.value = rng.choice([o for o in cls if o != n.value])
    elif choice == "field":
        n = rng.choice(field_nodes)
        cat = _field_category(n.value, pool)
        alts = [f for f in pool.get(cat, []) if f != n.value]
        if alts:
            n.value = rng.choice(alts)
    elif choice == "window":
        n = rng.choice(window_consts)
        n.value = rng.choice([w for w in WINDOW_CHOICES if w != n.value])
    elif choice == "const":
        n = rng.choice(free_consts)
        factor = 1.0 + rng.uniform(-CONST_JITTER, CONST_JITTER)
        n.value = round(float(n.value) * factor, 6) or 0.001
    else:  # subtree
        parent, idx, _ = rng.choice(v_positions)
        new = random_v_subtree(rng, pool, max_depth=max_depth)
        if parent is None:
            return new
        parent.children[idx] = new
    return t


def _field_category(field_name: str, pool: dict[str, list[str]]) -> str:
    for cat, fs in pool.items():
        if field_name in fs:
            return cat
    return "pv" if field_name in PRICE_VOLUME_BUILTINS else "unknown"


# --------------------------------------------------------------------------- #
# Validation (belt & suspenders: re-parse the emitted string)
# --------------------------------------------------------------------------- #
def validate(tree: Node, fv: Optional[FieldValidator] = None) -> tuple[bool, str]:
    """Emit -> re-parse with full type check. Returns (ok, expr_or_error)."""
    try:
        expr = to_fastexpr(tree)
    except Exception as e:  # noqa: BLE001
        return False, f"emit error: {e}"
    try:
        from_fastexpr(expr, fv=fv)
    except ParseError as e:
        return False, f"{expr}  ->  {e}"
    return True, expr
