"""Shared BRAIN API utilities for the alpha mining system.

Provides:
  - BrainClient: session management, adaptive concurrency, batch simulation
  - DB I/O: load/save alpha_db.json, lessons.json
  - Quality classification and lessons update

Usage:
    from brain_api import BrainClient, load_lessons, save_lessons
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from requests.auth import HTTPBasicAuth

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
CREDENTIAL_PATH = SKILL_DIR / "credential.txt"
ALPHA_DB_PATH = SKILL_DIR / "alpha_db.json"
LESSONS_PATH = SKILL_DIR / "lessons.json"

API_BASE = "https://api.worldquantbrain.com"

HEADERS = {
    "Accept": "application/json;version=2.0",
    "Content-Type": "application/json",
}

DEFAULT_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "truncation": 0.08,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "maxTrade": "OFF",
    "maxPosition": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def load_credentials() -> tuple[str, str]:
    env_user = os.getenv("WQ_BRAIN_USERNAME")
    env_password = os.getenv("WQ_BRAIN_PASSWORD")
    if env_user and env_password:
        return env_user, env_password
    if CREDENTIAL_PATH.exists():
        username, password = json.loads(CREDENTIAL_PATH.read_text(encoding="utf-8"))
        return str(username), str(password)
    raise FileNotFoundError(
        "BRAIN credentials not found. Set WQ_BRAIN_USERNAME/WQ_BRAIN_PASSWORD "
        'or create credential.txt with ["username", "password"].'
    )


# --------------------------------------------------------------------------- #
# DB I/O
# --------------------------------------------------------------------------- #
def load_alpha_db() -> dict[str, Any]:
    if ALPHA_DB_PATH.exists():
        return json.loads(ALPHA_DB_PATH.read_text(encoding="utf-8"))
    return {"alphas": {}, "last_update": None, "version": 1}


def save_alpha_db(db: dict[str, Any]) -> None:
    db["last_update"] = datetime.now(timezone.utc).isoformat()
    ALPHA_DB_PATH.write_text(json.dumps(db, indent=2, default=str), encoding="utf-8")


def load_lessons() -> dict[str, Any]:
    if LESSONS_PATH.exists():
        return json.loads(LESSONS_PATH.read_text(encoding="utf-8"))
    return {"patterns": {}, "param_insights": {}, "version": 1}


def save_lessons(lessons: dict[str, Any]) -> None:
    lessons["last_updated"] = datetime.now(timezone.utc).isoformat()
    LESSONS_PATH.write_text(json.dumps(lessons, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# BrainClient
# --------------------------------------------------------------------------- #
class BrainClient:
    """BRAIN API client with adaptive concurrency control."""

    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.session: requests.Session | None = None
        self._429_count = 0
        self._success_streak = 0

    def connect(self) -> None:
        username, password = load_credentials()
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(username, password)
        self.session.headers.update(HEADERS)
        resp = self.session.post(f"{API_BASE}/authentication")
        if resp.status_code != 201:
            raise RuntimeError(f"BRAIN auth failed: {resp.status_code} {resp.text}")
        print(f"[brain_api] Authenticated", flush=True)

    def _ensure_session(self) -> requests.Session:
        if self.session is None:
            self.connect()
        return self.session

    def _adjust_concurrency(self, got_429: bool) -> None:
        if got_429:
            self._429_count += 1
            self._success_streak = 0
            if self.max_concurrent > 1:
                self.max_concurrent -= 1
                print(f"[brain_api] 429 received, concurrency -> {self.max_concurrent}", flush=True)
        else:
            self._success_streak += 1
            self._429_count = 0
            if self._success_streak >= 10 and self.max_concurrent < 8:
                self.max_concurrent += 1
                self._success_streak = 0
                print(f"[brain_api] Success streak, concurrency -> {self.max_concurrent}", flush=True)

    def get_with_retry(self, url: str, retries: int = 3, **kwargs) -> requests.Response:
        s = self._ensure_session()
        for attempt in range(retries):
            try:
                resp = s.get(url, timeout=(10, 60), **kwargs)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    time.sleep(retry_after)
                    self._adjust_concurrency(True)
                    continue
                return resp
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"GET {url} failed after {retries} retries")

    def post_with_retry(self, url: str, json_body: dict, retries: int = 3, **kwargs) -> requests.Response:
        s = self._ensure_session()
        for attempt in range(retries):
            try:
                resp = s.post(url, json=json_body, timeout=(10, 60), **kwargs)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    time.sleep(retry_after)
                    self._adjust_concurrency(True)
                    continue
                self._adjust_concurrency(False)
                return resp
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"POST {url} failed after {retries} retries")

    # ----------------------------------------------------------------- #
    # Simulation
    # ----------------------------------------------------------------- #
    def build_payload(self, expression: str, settings: dict) -> dict:
        merged = {**DEFAULT_SETTINGS, **settings}
        return {
            "type": "REGULAR",
            "settings": merged,
            "regular": expression,
        }

    def simulate(self, expression: str, settings: dict) -> dict[str, Any]:
        """Submit a single simulation and poll until complete."""
        payload = self.build_payload(expression, settings)
        resp = self.post_with_retry(f"{API_BASE}/simulations", payload)
        if resp.status_code != 201:
            return {"status": "ERROR", "error": resp.text[:500], "status_code": resp.status_code}

        location = resp.headers.get("Location", "")
        sim_id = location.rstrip("/").split("/")[-1]

        result = self._poll_simulation(sim_id)
        return result

    def _poll_simulation(self, sim_id: str, timeout: int = 600) -> dict[str, Any]:
        start = time.time()
        while time.time() - start < timeout:
            resp = self.get_with_retry(f"{API_BASE}/simulations/{sim_id}")
            if resp.status_code != 200:
                time.sleep(8)
                continue
            data = resp.json()
            status = data.get("status", "UNKNOWN")
            if status == "COMPLETE":
                alpha_id = data.get("alpha")
                return {"status": "COMPLETE", "alpha_id": alpha_id, "sim_data": data}
            if status in ("ERROR", "FAILED"):
                return {"status": "ERROR", "sim_data": data}
            time.sleep(5)
        return {"status": "TIMEOUT", "simulation_id": sim_id}

    def batch_simulate(
        self, candidates: list[dict], max_concurrent: int | None = None
    ) -> list[dict[str, Any]]:
        """Simulate a batch of candidates with limited concurrency.

        Each candidate: {expression, settings, ...}
        Returns list of results with candidate info merged.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        concurrency = max_concurrent or self.max_concurrent
        results: list[dict[str, Any]] = []

        def _run_one(idx: int, cand: dict) -> dict[str, Any]:
            expr = cand["expression"]
            settings = cand.get("settings", {})
            try:
                sim_result = self.simulate(expr, settings)
                return {**cand, "sim_result": sim_result, "batch_idx": idx}
            except Exception as e:
                return {**cand, "sim_result": {"status": "ERROR", "error": str(e)}, "batch_idx": idx}

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_run_one, i, cand): i
                for i, cand in enumerate(candidates)
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                # Log progress
                status = result.get("sim_result", {}).get("status", "?")
                expr_short = result["expression"][:50]
                print(f"  [{len(results)}/{len(candidates)}] {status} — {expr_short}", flush=True)

        # Sort by original order
        results.sort(key=lambda r: r.get("batch_idx", 0))
        return results

    # ----------------------------------------------------------------- #
    # Metrics & PnL
    # ----------------------------------------------------------------- #
    def get_alpha(self, alpha_id: str) -> dict:
        resp = self.get_with_retry(f"{API_BASE}/alphas/{alpha_id}")
        if resp.status_code == 200:
            return resp.json()
        return {}

    def fetch_pnl(self, alpha_id: str) -> list[float]:
        try:
            resp = self.get_with_retry(f"{API_BASE}/alphas/{alpha_id}/recordsets/pnl")
        except Exception:
            return []
        if resp.status_code != 200 or not resp.text.strip():
            return []
        try:
            data = resp.json()
        except Exception:
            return []

        schema = data.get("schema", {})
        props = schema.get("properties", [])
        if isinstance(props, list):
            date_idx = next((i for i, p in enumerate(props) if p.get("name", "").lower() == "date"), 0)
            pnl_idx = next(
                (i for i, p in enumerate(props) if p.get("name", "").lower() in ("pnl", "cum_pnl", "returns", "ret")),
                1,
            )
        else:
            date_idx = next((v["index"] for k, v in props.items() if k.lower() == "date"), 0)
            pnl_idx = next(
                (v["index"] for k, v in props.items() if k.lower() in ("pnl", "cum_pnl", "returns", "ret")),
                1,
            )

        records = sorted(data.get("records", []), key=lambda r: r[date_idx])
        out: list[float] = []
        for row in records:
            rec = row[0] if isinstance(row, list) and len(row) == 1 and isinstance(row[0], list) else row
            try:
                out.append(float(rec[pnl_idx]))
            except Exception:
                continue
        return out

    def submit_alpha(self, alpha_id: str) -> dict[str, Any]:
        """Submit alpha and poll for result."""
        resp = self.post_with_retry(f"{API_BASE}/alphas/{alpha_id}/submit")
        if resp.status_code not in (200, 201):
            return {"submitted": False, "status_code": resp.status_code, "text": resp.text[:300]}

        for _ in range(30):
            time.sleep(10)
            alpha = self.get_alpha(alpha_id)
            status = alpha.get("status")
            if status == "ACTIVE":
                return {"submitted": True, "status": "ACTIVE", "alpha": alpha}
            checks = alpha.get("is", {}).get("checks", [])
            sc = next((c for c in checks if c.get("name") == "SELF_CORRELATION"), {})
            if sc.get("result") == "FAIL":
                return {"submitted": True, "status": status, "self_correlation": "FAIL", "alpha": alpha}
        return {"submitted": True, "status": "PENDING"}


# --------------------------------------------------------------------------- #
# Quality classification & lessons update
# --------------------------------------------------------------------------- #
def classify_alpha(expr: str) -> str:
    expr_lower = expr.lower()
    tokens = []
    if any(f in expr_lower for f in ["operating_income/equity", "oi/equity", "operating_income/sales"]):
        tokens.append("profitability")
    if any(f in expr_lower for f in ["est_eps", "est_fcf", "est_revenue", "est_ebitda", "est_ptp"]):
        tokens.append("analyst")
    if any(f in expr_lower for f in ["free_cash_flow", "cashflow_op", "cash_flow"]):
        tokens.append("cashflow")
    if any(f in expr_lower for f in ["close/open", "open/close", "vwap", "returns", "volume", "high + low"]):
        tokens.append("technical")
    if any(f in expr_lower for f in ["scl12_buzz", "scl12_sentiment", "sentiment"]):
        tokens.append("sentiment")
    if any(f in expr_lower for f in ["equity/assets", "liabilities/assets", "sales/assets"]):
        tokens.append("quality/leverage")
    return "+".join(tokens) if tokens else "other"


def daily_returns(cum_pnl: list[float]) -> list[float]:
    return [cum_pnl[i + 1] - cum_pnl[i] for i in range(len(cum_pnl) - 1)]


def compute_correlation(
    new_pnl: list[float], db: dict[str, Any], min_records: int = 50
) -> list[dict[str, Any]]:
    if len(new_pnl) < min_records + 1:
        return []
    new_ret = np.array(daily_returns(new_pnl))
    results: list[dict[str, Any]] = []
    for old_id, old in db.get("alphas", {}).items():
        if old.get("status") != "ACTIVE" or not old.get("pnl"):
            continue
        old_ret = np.array(daily_returns(old["pnl"]))
        if len(new_ret) != len(old_ret):
            continue
        corr = float(np.corrcoef(new_ret, old_ret)[0, 1])
        results.append({"alpha_id": old_id, "corr": corr, "sharpe": old.get("sharpe"), "fitness": old.get("fitness")})
    results.sort(key=lambda x: abs(x["corr"]), reverse=True)
    return results


def quality_filter(
    sharpe: float | None,
    fitness: float | None,
    turnover: float | None,
    max_corr: float | None,
    *,
    sharpe_threshold: float = 1.5,
    fitness_threshold: float = 1.0,
    turnover_threshold: float = 0.7,
    corr_threshold: float = 0.7,
) -> str:
    """Classify a simulation result into SUBMIT / OBSERVE / DISCARD."""
    if sharpe is None or fitness is None:
        return "DISCARD"

    if turnover is not None and turnover > turnover_threshold:
        return "DISCARD"

    if max_corr is not None and abs(max_corr) >= corr_threshold:
        return "DISCARD"

    if sharpe >= sharpe_threshold and fitness >= fitness_threshold:
        return "SUBMIT"

    if sharpe >= 1.0 or fitness >= 0.8:
        return "OBSERVE"

    return "DISCARD"


def _extract_params(candidate: dict) -> dict[str, str]:
    """Extract key params from a candidate for lessons tracking."""
    params = {}
    settings = candidate.get("settings", {})
    params["decay"] = str(settings.get("decay", 0))
    params["neutralization"] = str(settings.get("neutralization", "INDUSTRY"))
    field_pair = candidate.get("field_pair", {})
    params.update({k: str(v) for k, v in field_pair.items()})
    params.update({k: str(v) for k, v in candidate.get("params", {}).items()})
    return params


def update_lessons_from_result(
    lessons: dict[str, Any],
    candidate: dict,
    sim_result: dict,
    max_corr: float | None = None,
) -> None:
    """Update lessons.json with results from a simulation."""
    template_id = candidate.get("template_id", "unknown")
    sim_data = sim_result.get("sim_data", {})
    is_data = sim_data.get("is", {}) if isinstance(sim_data, dict) else {}
    if not is_data:
        # Try alpha data directly
        is_data = sim_data.get("is", {}) if sim_data else {}

    sharpe = is_data.get("sharpe")
    fitness = is_data.get("fitness")
    turnover = is_data.get("turnover")

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

    # Determine pass/fail
    action = quality_filter(sharpe, fitness, turnover, max_corr)
    passed = action == "SUBMIT"

    if passed:
        p["passed"] += 1
    else:
        # Record failure mode
        if sharpe is None:
            mode = "SIM_ERROR"
        elif sharpe < 1.0:
            mode = "LOW_SHARPE"
        elif fitness is not None and fitness < 1.0:
            mode = "LOW_FITNESS"
        elif turnover is not None and turnover > 0.7:
            mode = "HIGH_TURNOVER"
        elif max_corr is not None and abs(max_corr) >= 0.7:
            mode = "HIGH_CORR"
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
                "alpha_id": sim_result.get("alpha_id", "?"),
                "sharpe": sharpe,
                "expr": candidate.get("expression", ""),
            }

    # Auto-update action based on pass rate
    if p["tested"] >= 5:
        if p["pass_rate"] == 0.0:
            p["action"] = "skip"
        elif p["pass_rate"] < 0.2:
            p["action"] = "deprioritize"
        else:
            p["action"] = "expand"

    # Update param insights
    params = _extract_params(candidate)
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

    lessons["last_updated"] = datetime.now(timezone.utc).isoformat()
