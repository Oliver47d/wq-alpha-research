#!/usr/bin/env python3
"""factor_ast.py — FASTEXPR <-> typed AST (the #20 foundation for GP #21).

A factor is currently a flat string. Template expansion can only fill leaf holes
(field / window / group); the *root* operator and skeleton are frozen constants,
so every produced factor is the same tree — this is the structural homogeneity
the template/LLM producers cannot escape (see BUGS_AND_IMPROVEMENTS.md, path A).

To let genetic programming mutate the WHOLE tree (including the root shell), a
factor must first be a tree, not a string. This module provides:

  * Node                — one dataclass, kind in {OP, FIELD, CONST, GROUP}.
  * OP_SIGNATURES       — operator arg-types + return-type table, DERIVED from
                          generate_candidates.KNOWN_OPERATORS so the two can
                          never drift (a hand-kept copy once drifted and silently
                          killed 5 templates; see the 2026-07-03 fix-log entry).
  * to_fastexpr(node)   — tree -> string (plain decimals, NO scientific notation
                          which BRAIN's parser rejects; minimal parenthesisation).
  * from_fastexpr(s)    — string -> tree, reusing Python's `ast` module for
                          lexing/parsing (we only write a ~1-node translation
                          layer), then a recursive TYPE CHECK against the
                          signatures + field/group validity. Illegal -> ParseError.

Type system (4 types):
  V  vector  — a per-stock column / any operator's return value (a factor is V)
  N  number  — numeric literal / time-series window (positive int)
  G  group   — a grouping builtin (industry/subindustry/sector/market)
  B  bool    — a condition (only used by if_else's first arg / comparisons)
"""
from __future__ import annotations

import ast as _pyast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Union

# Single source of truth: reuse the exact operator/field vocabulary the rest of
# the pipeline validates against. Never re-declare these here.
from generate_candidates import (  # noqa: E402
    KNOWN_OPERATORS,
    GROUP_BUILTINS,
    PRICE_VOLUME_BUILTINS,
    FieldValidator,
)

# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
OP, FIELD, CONST, GROUP = "OP", "FIELD", "CONST", "GROUP"


@dataclass
class Node:
    kind: str                      # OP | FIELD | CONST | GROUP
    value: Union[str, float]       # OP->op name; FIELD->field; CONST->number; GROUP->group
    children: list["Node"] = field(default_factory=list)

    # --- GP primitives ---
    def rtype(self) -> str:
        """Return type of this node: V / N / G / B."""
        if self.kind == FIELD:
            return "V"
        if self.kind == CONST:
            return "N"
        if self.kind == GROUP:
            return "G"
        # OP: look up the signature's declared return type.
        sig = OP_SIGNATURES.get(self.value)
        return sig[1] if sig else "V"

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def copy(self) -> "Node":
        return Node(self.kind, self.value, [c.copy() for c in self.children])


# --------------------------------------------------------------------------- #
# Operator signature table (arg-type tuple, return-type)
#   V=vector  N:win=time-series window  N:const=plain numeric  G=group  B=bool
# --------------------------------------------------------------------------- #
OP_SIGNATURES: dict[str, tuple[tuple[str, ...], str]] = {
    # ---- cross-sectional: V (+ optional G) -> V ----
    "rank":             (("V",),               "V"),
    "zscore":           (("V",),               "V"),
    "scale":            (("V",),               "V"),
    "winsorize":        (("V",),               "V"),
    "normalize":        (("V",),               "V"),
    "quantile":         (("V",),               "V"),
    "group_rank":       (("V", "G"),           "V"),
    "group_zscore":     (("V", "G"),           "V"),
    "group_mean":       (("V", "G"),           "V"),
    "group_neutralize": (("V", "G"),           "V"),
    # ---- time-series: V + window -> V ----
    "ts_rank":          (("V", "N:win"),       "V"),
    "ts_mean":          (("V", "N:win"),       "V"),
    "ts_std_dev":       (("V", "N:win"),       "V"),
    "ts_delta":         (("V", "N:win"),       "V"),
    "ts_delay":         (("V", "N:win"),       "V"),
    "ts_sum":           (("V", "N:win"),       "V"),
    "ts_max":           (("V", "N:win"),       "V"),
    "ts_min":           (("V", "N:win"),       "V"),
    "ts_decay_linear":  (("V", "N:win"),       "V"),
    "ts_count":         (("V", "N:win"),       "V"),
    "ts_zscore":        (("V", "N:win"),       "V"),
    "ts_product":       (("V", "N:win"),       "V"),
    "ts_argmax":        (("V", "N:win"),       "V"),
    "ts_argmin":        (("V", "N:win"),       "V"),
    "ts_scale":         (("V", "N:win"),       "V"),
    "ts_corr":          (("V", "V", "N:win"),  "V"),
    "ts_covariance":    (("V", "V", "N:win"),  "V"),
    "ts_regression":    (("V", "V", "N:win"),  "V"),
    # ---- arithmetic / logic ----
    "abs":              (("V",),               "V"),
    "log":              (("V",),               "V"),
    "sign":             (("V",),               "V"),
    "sqrt":             (("V",),               "V"),
    "power":            (("V", "N:const"),     "V"),
    "add":              (("VN", "VN"),         "V"),
    "subtract":         (("VN", "VN"),         "V"),
    "multiply":         (("VN", "VN"),         "V"),
    "divide":           (("VN", "VN"),         "V"),
    "max":              (("VN", "VN"),         "V"),
    "min":              (("VN", "VN"),         "V"),
    "if_else":          (("B", "VN", "VN"),    "V"),
}

# Virtual/internal operators — NOT part of KNOWN_OPERATORS (they come from infix
# sugar in the source string), so they are exempt from the drift assertion.
VIRTUAL_OPS: dict[str, tuple[tuple[str, ...], str]] = {
    "neg":  (("V",),        "V"),   # unary minus  -x
    "cmp":  (("VN", "VN"),  "B"),   # a <,>,<=,>=,==,!= b  -> bool (for if_else)
}
OP_SIGNATURES.update(VIRTUAL_OPS)

# Infix / comparison sugar.
INFIX_TO_OP = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
UNARY_NEG = "neg"

# Mutation domains (used by GP #21; declared here next to the signatures).
WINDOW_CHOICES = [3, 5, 10, 21, 42, 63, 126, 252]
CONST_JITTER = 0.5  # multiplicative +/-50% for CONST mutation

# Small denominator-guard epsilons must NOT be mutated (mutating 0.001 -> 0 would
# re-introduce the divide-by-zero the epsilon exists to prevent). A CONST is a
# protected epsilon if it is small and sits directly inside an add whose sibling
# is a vector-bearing subtree. We flag them by magnitude here; GP checks this.
EPSILON_MAX = 0.01


def is_epsilon_const(node: "Node") -> bool:
    """True if this CONST looks like a divide-by-zero guard (small positive)."""
    return node.kind == CONST and isinstance(node.value, (int, float)) and 0 < float(node.value) <= EPSILON_MAX


# --------------------------------------------------------------------------- #
# Drift guard: the real-operator half of OP_SIGNATURES MUST equal KNOWN_OPERATORS
# --------------------------------------------------------------------------- #
def _assert_no_drift() -> None:
    real_ops = set(OP_SIGNATURES) - set(VIRTUAL_OPS)
    missing = KNOWN_OPERATORS - real_ops
    extra = real_ops - KNOWN_OPERATORS
    if missing or extra:
        raise AssertionError(
            "OP_SIGNATURES drifted from KNOWN_OPERATORS "
            f"(missing signatures for: {sorted(missing)}; "
            f"unknown ops in table: {sorted(extra)}). "
            "Keep the operator vocabulary in a single source of truth."
        )


_assert_no_drift()


class ParseError(ValueError):
    """Raised when a FASTEXPR string cannot be parsed into a valid typed AST."""


# --------------------------------------------------------------------------- #
# to_fastexpr : tree -> string
# --------------------------------------------------------------------------- #
_INFIX_PREC = {"add": 1, "subtract": 1, "multiply": 2, "divide": 2}


def _fmt_const(v: Union[int, float]) -> str:
    """Plain decimal, NEVER scientific notation (BRAIN's parser rejects '1e-6')."""
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
        return str(int(v))
    # Use repr-ish but force plain decimal; strip any exponent form.
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def to_fastexpr(node: "Node") -> str:
    if node.kind in (FIELD, GROUP):
        return str(node.value)
    if node.kind == CONST:
        return _fmt_const(node.value)  # type: ignore[arg-type]
    # OP
    op = node.value
    if op == UNARY_NEG:
        inner = _emit(node.children[0], parent_prec=3)
        return f"-{inner}"
    if op == "cmp":
        sym = getattr(node, "symbol", "<")
        left = _emit(node.children[0], parent_prec=0)
        right = _emit(node.children[1], parent_prec=0)
        return f"{left} {sym} {right}"
    if op in _INFIX_PREC:
        prec = _INFIX_PREC[op]
        sym = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[op]
        left = _emit(node.children[0], parent_prec=prec)
        right = _emit(node.children[1], parent_prec=prec + 1)  # right-assoc guard
        return f"{left} {sym} {right}"
    # regular function call
    args = ", ".join(_emit(c, parent_prec=0) for c in node.children)
    return f"{op}({args})"


def _emit(node: "Node", parent_prec: int) -> str:
    """Emit a child, adding parens only when operator precedence requires it."""
    s = to_fastexpr(node)
    if node.kind == OP and node.value == "cmp" and parent_prec > 0:
        # comparisons bind looser than any arithmetic; parenthesise inside ops.
        return f"({s})"
    if node.kind == OP and node.value in _INFIX_PREC:
        if _INFIX_PREC[node.value] < parent_prec:
            return f"({s})"
    if node.kind == OP and node.value == UNARY_NEG and parent_prec >= 2:
        return f"({s})"
    return s


# --------------------------------------------------------------------------- #
# from_fastexpr : string -> tree  (reuse Python `ast` for lexing/parsing)
# --------------------------------------------------------------------------- #
_CMP_SYMBOLS = {
    _pyast.Lt: "<", _pyast.Gt: ">", _pyast.LtE: "<=",
    _pyast.GtE: ">=", _pyast.Eq: "==", _pyast.NotEq: "!=",
}
_BINOP_MAP = {
    _pyast.Add: "add", _pyast.Sub: "subtract",
    _pyast.Mult: "multiply", _pyast.Div: "divide",
}


def _make_cmp(left: "Node", right: "Node", symbol: str) -> "Node":
    n = Node(OP, "cmp", [left, right])
    n.symbol = symbol  # type: ignore[attr-defined]
    return n


def _convert(py: _pyast.AST) -> "Node":
    """Translate a Python AST node into our factor Node (structure only)."""
    if isinstance(py, _pyast.Expression):
        return _convert(py.body)
    if isinstance(py, _pyast.Call):
        name = getattr(py.func, "id", None)
        if name is None:
            raise ParseError("unsupported call target")
        return Node(OP, name, [_convert(a) for a in py.args])
    if isinstance(py, _pyast.BinOp):
        op = _BINOP_MAP.get(type(py.op))
        if op is None:
            raise ParseError(f"unsupported binary operator: {type(py.op).__name__}")
        return Node(OP, op, [_convert(py.left), _convert(py.right)])
    if isinstance(py, _pyast.UnaryOp) and isinstance(py.op, _pyast.USub):
        return Node(OP, UNARY_NEG, [_convert(py.operand)])
    if isinstance(py, _pyast.UnaryOp) and isinstance(py.op, _pyast.UAdd):
        return _convert(py.operand)  # unary plus is a no-op
    if isinstance(py, _pyast.Compare):
        if len(py.ops) != 1 or len(py.comparators) != 1:
            raise ParseError("chained comparisons not supported")
        sym = _CMP_SYMBOLS.get(type(py.ops[0]))
        if sym is None:
            raise ParseError(f"unsupported comparator: {type(py.ops[0]).__name__}")
        return _make_cmp(_convert(py.left), _convert(py.comparators[0]), sym)
    if isinstance(py, _pyast.Name):
        tok = py.id
        if tok in GROUP_BUILTINS:
            return Node(GROUP, tok)
        return Node(FIELD, tok)  # field validity checked in the type pass
    if isinstance(py, _pyast.Constant):
        if isinstance(py.value, (int, float)) and not isinstance(py.value, bool):
            return Node(CONST, py.value)
        raise ParseError(f"unsupported constant: {py.value!r}")
    raise ParseError(f"unsupported syntax node: {type(py).__name__}")


def _type_check(node: "Node", fv: FieldValidator | None) -> str:
    """Recursively verify types against OP_SIGNATURES + field/group validity.

    Returns the node's return type, or raises ParseError.
    """
    if node.kind == FIELD:
        tok = str(node.value)
        if tok in PRICE_VOLUME_BUILTINS:
            return "V"
        if fv is not None and not fv.is_valid(tok):
            raise ParseError(f"unknown field: {tok!r}")
        return "V"
    if node.kind == CONST:
        return "N"
    if node.kind == GROUP:
        if str(node.value) not in GROUP_BUILTINS:
            raise ParseError(f"unknown group: {node.value!r}")
        return "G"
    # OP
    op = node.value
    sig = OP_SIGNATURES.get(op)
    if sig is None:
        raise ParseError(f"unknown operator: {op!r}")
    argtypes, rtype = sig
    if len(node.children) != len(argtypes):
        raise ParseError(
            f"{op}: expected {len(argtypes)} args, got {len(node.children)}"
        )
    for child, want in zip(node.children, argtypes):
        got = _type_check(child, fv)
        if want == "VN":
            # scalar-broadcast position: vector, numeric literal, OR a boolean
            # indicator (a comparison used as a 0/1 mask, a BRAIN idiom).
            if got not in ("V", "N", "B"):
                raise ParseError(f"{op}: arg type {got} not V/N/B")
            continue
        if want.startswith("N"):
            # window/const positions must be a numeric literal
            if child.kind != CONST:
                raise ParseError(f"{op}: arg must be a number, got {child.kind}")
            if want == "N:win" and (not float(child.value).is_integer() or float(child.value) <= 0):
                raise ParseError(f"{op}: window must be a positive integer, got {child.value}")
            continue
        if got != want:
            raise ParseError(f"{op}: arg type {got} != expected {want}")
    return rtype


def from_fastexpr(s: str, fv: FieldValidator | None = None, type_check: bool = True) -> "Node":
    """Parse a FASTEXPR string into a validated typed AST.

    Raises ParseError on any lexical, structural, or type error.
    """
    if not s or not s.strip():
        raise ParseError("empty expression")
    try:
        py = _pyast.parse(s.strip(), mode="eval")
    except SyntaxError as e:
        raise ParseError(f"syntax error: {e}") from e
    node = _convert(py)
    if type_check:
        rt = _type_check(node, fv)
        if rt != "V":
            raise ParseError(f"top-level expression must be a vector (V), got {rt}")
    return node


# --------------------------------------------------------------------------- #
# Convenience: does a round-trip preserve the structure fingerprint?
# --------------------------------------------------------------------------- #
def roundtrip_equiv(expr: str, fv: FieldValidator | None = None) -> tuple[bool, str, str]:
    """Return (equivalent, reparsed_string, reason).

    Operational definition of 'semantically the same factor' = IDEMPOTENCE:
    parse -> emit -> parse -> emit reaches a fixed point, i.e. the second emit
    equals the first. (We do NOT compare against the raw source's fingerprint,
    because the source may carry redundant parens like '/(close)' that emit
    canonically strips — a normalization, not a semantic change.)
    """
    node1 = from_fastexpr(expr, fv=fv)
    out1 = to_fastexpr(node1)
    node2 = from_fastexpr(out1, fv=fv)
    out2 = to_fastexpr(node2)
    return (out1 == out2), out1, ("fixed point" if out1 == out2 else f"{out1!r} != {out2!r}")
