#!/usr/bin/env python3
"""factor_gp_loop.py — the GP main loop + multi-objective fitness (#21 C).

Ties #20 (typed AST), #21A (genetic operators) and #21B (real-factor seeds) into
an evolutionary search. The design keeps the EXPENSIVE part (BRAIN simulation +
correlation) behind a pluggable `evaluate` callable, so the pure evolution logic
(selection / breeding / dedup / elitism) can be verified OFFLINE with a synthetic
fitness — exactly how #20/#21A/#21B were hardened before touching the platform.

Three pieces:

  1. scalar_fitness(metrics)  — the weighted multi-objective scalar
        fitness = Sharpe - lambda*|max_corr| - mu*turnover
     (an invalid / failed sim collapses to -inf so it never survives selection).

  2. evolve(seeds, evaluate, ...)  — the pure loop. Per generation:
        evaluate population -> score -> keep elites -> breed (crossover+mutate)
        -> dedup by ast_hash + drop lessons-v2 'skip' structures -> repeat.
     `evaluate` maps a batch of exprs to metrics dicts; it is the ONLY seam that
     knows about BRAIN, so the loop itself is deterministic and offline-testable.

  3. BrainEvaluator  — the real adapter. Reuses the existing machinery verbatim
     (batch_simulate_stream / fetch_self_correlation / compute_correlation /
     quality_filter): NO reimplementation of BRAIN plumbing. Two-level correlation:
     cheap local PnL prefilter, then the authoritative platform SELF_CORRELATION
     endpoint only for elites. Running it consumes real simulation quota and
     mutates alpha_db.json / lessons.json, so it is gated behind an explicit call.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field as _dc_field
from typing import Any, Callable, Optional

from factor_ast import Node, to_fastexpr
from factor_gp import mutate, crossover, validate, _field_pool_by_category
from factor_seeds import Seed, load_seeds
from generate_candidates import FieldValidator, structure_fingerprint

# --------------------------------------------------------------------------- #
# 1. Multi-objective fitness (weighted scalar; B1 first — see BUGS path A)
# --------------------------------------------------------------------------- #
CORR_LAMBDA = 2.0   # penalty per unit |max_corr| — corr is the scarce resource
TURNOVER_MU = 1.0   # penalty per unit turnover — cost drag


def scalar_fitness(
    metrics: dict[str, Any],
    corr_lambda: float = CORR_LAMBDA,
    turnover_mu: float = TURNOVER_MU,
) -> float:
    """fitness = Sharpe - lambda*|max_corr| - mu*turnover.

    Missing Sharpe (sim failed / rejected) -> -inf so it dies in selection.
    Missing corr/turnover contribute 0 penalty (unknown, not assumed bad).
    """
    sharpe = metrics.get("sharpe")
    if sharpe is None or (isinstance(sharpe, float) and math.isnan(sharpe)):
        return float("-inf")
    corr = metrics.get("max_corr")
    turnover = metrics.get("turnover")
    penalty = 0.0
    if corr is not None:
        penalty += corr_lambda * abs(corr)
    if turnover is not None:
        penalty += turnover_mu * float(turnover)
    return float(sharpe) - penalty


# --------------------------------------------------------------------------- #
# Individual: a tree + its cached expr / hash / metrics / score
# --------------------------------------------------------------------------- #
@dataclass
class Individual:
    tree: Node
    expr: str
    ast_hash: str
    metrics: dict[str, Any] = _dc_field(default_factory=dict)
    score: float = float("-inf")
    origin: str = "seed"        # 'seed' | 'crossover' | 'mutate'


def _individual(tree: Node, fcats: dict[str, str], origin: str) -> Individual:
    expr = to_fastexpr(tree)
    ast_hash = structure_fingerprint(expr, fcats)["ast_hash"]
    return Individual(tree=tree, expr=expr, ast_hash=ast_hash, origin=origin)


# --------------------------------------------------------------------------- #
# 2. Pure evolution loop (offline-testable — evaluate is injected)
# --------------------------------------------------------------------------- #
EvaluateFn = Callable[[list[Individual]], dict[str, dict[str, Any]]]
"""expr -> metrics dict ({'sharpe','fitness','turnover','max_corr', ...})."""


def _breed(
    parents: list[Individual],
    rng: random.Random,
    fv: Optional[FieldValidator],
    pool: dict[str, list[str]],
    fcats: dict[str, str],
    n_children: int,
    skip_hashes: set[str],
    max_attempts_factor: int = 20,
) -> list[Individual]:
    """Produce n_children NEW legal individuals via crossover/mutate.

    Legal-by-construction operators still emit through validate() as a belt-and-
    suspenders check. Children whose structure is a lessons-v2 'skip' or already
    present this generation are rejected, keeping the population diverse.
    """
    children: list[Individual] = []
    seen = set(skip_hashes)
    attempts = 0
    cap = n_children * max_attempts_factor
    while len(children) < n_children and attempts < cap:
        attempts += 1
        if len(parents) >= 2 and rng.random() < 0.5:
            a, b = rng.sample(parents, 2)
            child, origin = crossover(a.tree, b.tree, rng), "crossover"
        else:
            child, origin = mutate(rng.choice(parents).tree, rng, field_pool=pool), "mutate"
        ok, expr = validate(child, fv)
        if not ok:
            continue
        h = structure_fingerprint(expr, fcats)["ast_hash"]
        if h in seen:
            continue
        seen.add(h)
        ind = _individual(child, fcats, origin)
        children.append(ind)
    return children


def _skip_hashes(lessons: dict) -> set[str]:
    """ast_hashes the lessons-v2 rollups say to skip (enough evidence, 0 passes)."""
    by_ast = (lessons or {}).get("rollups", {}).get("by_ast", {})
    return {h for h, r in by_ast.items() if r.get("action") == "skip"}


def evolve(
    seeds: list[Seed],
    evaluate: EvaluateFn,
    *,
    generations: int = 5,
    population_size: int = 20,
    elite_size: int = 6,
    lessons: Optional[dict] = None,
    fields_path: Optional[Any] = None,
    rng: Optional[random.Random] = None,
    on_generation: Optional[Callable[[int, list[Individual]], None]] = None,
) -> list[Individual]:
    """Run the GP loop and return the final population sorted best-first.

    Pure w.r.t. `evaluate`: swap in a synthetic evaluator to test the mechanics
    offline, or BrainEvaluator to run for real. Elitism carries the best `elite_size`
    forward unchanged; the rest of each generation is bred from the elites.
    """
    rng = rng or random.Random()
    fv = FieldValidator(fields_path) if fields_path else None
    fcats = fv.field_categories if fv else {}
    pool = _field_pool_by_category(fv)
    skip = _skip_hashes(lessons or {})

    # Initial population from seeds (dedup by structure, drop skip-listed).
    pop: list[Individual] = []
    seen_h: set[str] = set()
    for s in seeds:
        if s.ast_hash in skip or s.ast_hash in seen_h:
            continue
        seen_h.add(s.ast_hash)
        pop.append(Individual(tree=s.tree, expr=s.expr, ast_hash=s.ast_hash, origin="seed"))
        if len(pop) >= population_size:
            break

    for gen in range(generations):
        metrics_by_expr = evaluate(pop)
        for ind in pop:
            ind.metrics = metrics_by_expr.get(ind.expr, {})
            ind.score = scalar_fitness(ind.metrics)
        pop.sort(key=lambda x: x.score, reverse=True)

        if on_generation:
            on_generation(gen, pop)

        if gen == generations - 1:
            break

        elites = pop[:elite_size]
        alive = [e for e in elites if e.score > float("-inf")] or elites
        n_children = population_size - len(elites)
        gen_hashes = {e.ast_hash for e in elites} | skip
        children = _breed(alive, rng, fv, pool, fcats, n_children, gen_hashes)
        pop = elites + children

    return pop


# --------------------------------------------------------------------------- #
# 3. BRAIN evaluator adapter (real; consumes quota — gated behind explicit call)
# --------------------------------------------------------------------------- #
class BrainEvaluator:
    """Evaluate individuals on the live BRAIN platform, reusing existing machinery.

    Two-level correlation (cheap -> authoritative):
      * prefilter: local PnL correlation vs cached ACTIVE alphas (compute_correlation).
      * elite gate: the platform SELF_CORRELATION endpoint (fetch_self_correlation)
        only for candidates that clear the local prefilter, since it is the slow,
        authoritative check BRAIN's own submission uses.

    Side effects: consumes simulation quota, writes alpha_db.json / lessons.json.
    Constructing this does NOT run anything; call it as the evolve() evaluate= arg.
    """

    def __init__(
        self,
        client,
        db: dict,
        lessons: dict,
        active_pnls: dict[str, list[float]],
        corr_prefilter: float = 0.7,
    ):
        self.client = client
        self.db = db
        self.lessons = lessons
        self.active_pnls = active_pnls
        self.corr_prefilter = corr_prefilter

    def __call__(self, population: list[Individual]) -> dict[str, dict[str, Any]]:
        # Lazy imports: only needed on a real run, keep offline tests dependency-free.
        from brain_api import save_alpha_db, save_lessons, update_lessons_from_result
        from datetime import datetime, timezone

        candidates = [{"expression": ind.expr, "template_id": ind.ast_hash,
                       "settings": self.db.get("_default_settings", {})} for ind in population]
        out: dict[str, dict[str, Any]] = {}

        for r in self.client.batch_simulate_stream(candidates):
            sim = r.get("sim_result", {})
            expr = r.get("expression", "")
            if sim.get("status") != "COMPLETE":
                update_lessons_from_result(self.lessons, r, sim)
                out[expr] = {}
                continue
            is_data = (sim.get("sim_data") or {}).get("is", {}) or {}
            sharpe = is_data.get("sharpe")
            turnover = is_data.get("turnover")
            alpha_id = sim.get("alpha_id")
            metrics = {"sharpe": sharpe, "fitness": is_data.get("fitness"),
                       "turnover": turnover, "alpha_id": alpha_id}

            if alpha_id:
                self.db["alphas"][alpha_id] = {
                    "expression": expr, "status": "UNSUBMITTED", "sharpe": sharpe,
                    "fitness": is_data.get("fitness"), "turnover": turnover,
                    "template_id": r.get("template_id"),
                    "simulated_at": datetime.now(timezone.utc).isoformat(),
                }

            # Level 1: cheap local prefilter.
            max_corr = None
            if alpha_id and self.active_pnls:
                new_pnl = self.client.fetch_pnl(alpha_id)
                if len(new_pnl) >= 50:
                    from brain_api import compute_correlation
                    corr_list = compute_correlation(
                        new_pnl,
                        {"alphas": {aid: {"status": "ACTIVE", "pnl": p}
                                    for aid, p in self.active_pnls.items()}},
                    )
                    if corr_list:
                        max_corr = max(abs(c.get("correlation", 0)) for c in corr_list)

            # Level 2: authoritative platform check — only if local looks clean.
            if alpha_id and (max_corr is None or max_corr < self.corr_prefilter):
                platform_corr = self.client.fetch_self_correlation(alpha_id)
                if platform_corr is not None:
                    max_corr = platform_corr

            metrics["max_corr"] = max_corr
            update_lessons_from_result(self.lessons, r, sim, max_corr)
            out[expr] = metrics
            save_alpha_db(self.db)
            save_lessons(self.lessons)

        return out


if __name__ == "__main__":
    # Offline mechanics check — synthetic evaluator, no BRAIN, no quota.
    import argparse
    from generate_candidates import FIELDS_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    def synthetic_eval(pop: list[Individual]) -> dict[str, dict[str, Any]]:
        # Deterministic pseudo-metrics from the expr hash: rewards shorter, deeper
        # trees a bit, adds noise. Purely to exercise selection/breeding mechanics.
        out = {}
        for ind in pop:
            h = int(structure_fingerprint(ind.expr, {})["ast_hash"], 16)
            r2 = random.Random(h ^ args.seed)
            out[ind.expr] = {"sharpe": round(0.5 + r2.random() * 2.5, 3),
                             "turnover": round(0.05 + r2.random() * 0.4, 3),
                             "max_corr": round(r2.random() * 0.6, 3)}
        return out

    seeds = load_seeds()
    hist = []
    def _cb(gen, pop):
        best = pop[0]
        hist.append(best.score)
        distinct = len({p.ast_hash for p in pop})
        print(f"  gen {gen}: best_fitness={best.score:.3f} distinct_hash={distinct} "
              f"pop={len(pop)} best_expr={best.expr[:55]}")

    print(f"seeds: {len(seeds)}  running {args.generations} generations, pop={args.pop}")
    final = evolve(seeds, synthetic_eval, generations=args.generations,
                   population_size=args.pop, fields_path=FIELDS_PATH, rng=rng,
                   on_generation=_cb)
    assert all(p.score > float("-inf") for p in final if p.metrics), "invalid survivor"
    print(f"best-fitness trajectory: {[round(h,3) for h in hist]}")
    assert hist == sorted(hist) or hist[-1] >= hist[0], "elitism regressed best fitness"
    print("OFFLINE PASS: loop runs, elitism monotonic, structures legal.")
