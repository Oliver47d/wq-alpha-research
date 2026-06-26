"""Shared BRAIN API utilities for the alpha mining system.

Provides:
  - BrainClient: session management, adaptive concurrency, batch simulation
  - DB I/O: load/save alpha_db.json, lessons.json
  - Quality classification and lessons update

Usage:
    from brain_api import BrainClient, load_lessons, save_lessons
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from requests.auth import HTTPBasicAuth

# Structure fingerprint is the v2 lessons aggregation key. Defined in
# generate_candidates (the lower-level producer module) so both the template
# grid and the LLM path share one key space. generate_candidates does NOT
# import brain_api, so this import is safe (no cycle).
try:
    from generate_candidates import structure_fingerprint, FieldValidator, FIELDS_PATH as _FIELDS_PATH
except Exception:  # pragma: no cover - keeps brain_api importable in isolation
    structure_fingerprint = None  # type: ignore
    FieldValidator = None  # type: ignore
    _FIELDS_PATH = None  # type: ignore

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
CREDENTIAL_PATH = SKILL_DIR / "credential.txt"
ALPHA_DB_PATH = SKILL_DIR / "alpha_db.json"
LESSONS_PATH = SKILL_DIR / "lessons.json"

API_BASE = "https://api.worldquantbrain.com"

LOG_LEVEL = os.getenv("WQ_LOG_LEVEL", "INFO").upper()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logger = logging.getLogger(__name__)

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


def _expr_fingerprint(expression: str) -> str:
    """Stable short identifier for an expression without logging the formula."""
    return hashlib.sha1(expression.encode("utf-8")).hexdigest()[:12]


def _text_fingerprint(text: str) -> str | None:
    if not text:
        return None
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def load_credentials() -> tuple[str, str]:
    env_user = os.getenv("WQ_BRAIN_USERNAME")
    env_password = os.getenv("WQ_BRAIN_PASSWORD")
    if env_user and env_password:
        logger.info("Loaded BRAIN credentials from environment")
        return env_user, env_password
    if CREDENTIAL_PATH.exists():
        username, password = json.loads(CREDENTIAL_PATH.read_text(encoding="utf-8"))
        logger.info("Loaded BRAIN credentials from credential file path=%s", CREDENTIAL_PATH)
        return str(username), str(password)
    logger.info("BRAIN credentials not found env_present=%s credential_path=%s", bool(env_user or env_password), CREDENTIAL_PATH)
    raise FileNotFoundError(
        "BRAIN credentials not found. Set WQ_BRAIN_USERNAME/WQ_BRAIN_PASSWORD "
        'or create credential.txt with ["username", "password"].'
    )


# --------------------------------------------------------------------------- #
# DB I/O
# --------------------------------------------------------------------------- #
def load_alpha_db() -> dict[str, Any]:
    if ALPHA_DB_PATH.exists():
        db = json.loads(ALPHA_DB_PATH.read_text(encoding="utf-8"))
        logger.info("Loaded alpha DB path=%s alpha_count=%s", ALPHA_DB_PATH, len(db.get("alphas", {})))
        return db
    logger.info("Alpha DB not found; initializing empty DB path=%s", ALPHA_DB_PATH)
    return {"alphas": {}, "last_update": None, "version": 1}


def save_alpha_db(db: dict[str, Any]) -> None:
    db["last_update"] = datetime.now(timezone.utc).isoformat()
    ALPHA_DB_PATH.write_text(json.dumps(db, indent=2, default=str), encoding="utf-8")
    logger.info("Saved alpha DB path=%s alpha_count=%s", ALPHA_DB_PATH, len(db.get("alphas", {})))


LESSONS_VERSION = 2


def _empty_lessons() -> dict[str, Any]:
    """A fresh, fully-formed v2 lessons document.

    v2 is a *superset* of v1: the v1 concept/template aggregates (`patterns`,
    `param_insights`) are retained (still consumed by the skip logic and the LLM
    prompt), and two new structures are added:

      * experiments — append-only list of raw simulation facts (one per result).
        This is the actual "past-experience log": immutable evidence, never
        rewritten, that any future analysis can re-aggregate from scratch.
      * rollups     — a *derived cache* keyed by structure (ast_hash), data
        category (field_class), and decay. Recomputable from `experiments` at
        any time; kept inline so consumers don't have to re-scan every round.
    """
    return {
        "patterns": {},
        "param_insights": {},
        "experiments": [],
        "rollups": {"by_ast": {}, "by_field_class": {}, "by_decay": {}},
        "version": LESSONS_VERSION,
    }


def _migrate_lessons(lessons: dict[str, Any]) -> dict[str, Any]:
    """Bring any older/partial lessons doc up to the v2 shape, in place.

    Backward compatible: a v1 file (or the canonical empty
    {"patterns": {}, "param_insights": {}, "version": 1}) simply gains the new
    `experiments`/`rollups` keys; nothing existing is dropped or rewritten.
    """
    lessons.setdefault("patterns", {})
    lessons.setdefault("param_insights", {})
    lessons.setdefault("experiments", [])
    rollups = lessons.setdefault("rollups", {})
    rollups.setdefault("by_ast", {})
    rollups.setdefault("by_field_class", {})
    rollups.setdefault("by_decay", {})
    lessons["version"] = LESSONS_VERSION
    return lessons


def load_lessons() -> dict[str, Any]:
    if LESSONS_PATH.exists():
        lessons = json.loads(LESSONS_PATH.read_text(encoding="utf-8"))
        prev_version = lessons.get("version")
        lessons = _migrate_lessons(lessons)
        logger.info(
            "Loaded lessons path=%s version=%s->%s pattern_count=%s experiment_count=%s",
            LESSONS_PATH, prev_version, lessons["version"],
            len(lessons.get("patterns", {})), len(lessons.get("experiments", [])),
        )
        return lessons
    logger.info("Lessons not found; initializing empty lessons path=%s", LESSONS_PATH)
    return _empty_lessons()


def save_lessons(lessons: dict[str, Any]) -> None:
    lessons["last_updated"] = datetime.now(timezone.utc).isoformat()
    LESSONS_PATH.write_text(json.dumps(lessons, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved lessons path=%s pattern_count=%s", LESSONS_PATH, len(lessons.get("patterns", {})))


# --------------------------------------------------------------------------- #
# BrainClient
# --------------------------------------------------------------------------- #
class BrainClient:
    """BRAIN API client with adaptive concurrency control."""

    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.session: requests.Session | None = None
        self._429_count = 0
        self._success_streak = 0

    def connect(self) -> None:
        username, password = load_credentials()
        logger.info("Connecting to BRAIN API authentication endpoint")
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(username, password)
        self.session.headers.update(HEADERS)
        resp = self.session.post(f"{API_BASE}/authentication")
        if resp.status_code != 201:
            logger.info(
                "BRAIN authentication failed status_code=%s body_len=%s body_hash=%s",
                resp.status_code,
                len(resp.text or ""),
                _text_fingerprint(resp.text or ""),
            )
            logger.debug("BRAIN authentication failure body=%s", resp.text[:1000])
            raise RuntimeError(f"BRAIN auth failed: {resp.status_code} {resp.text}")
        logger.info("BRAIN authentication succeeded status_code=%s", resp.status_code)
        print(f"[brain_api] Authenticated", flush=True)

    def _ensure_session(self) -> requests.Session:
        if self.session is None:
            self.connect()
        return self.session

    def _adjust_concurrency(self, got_429: bool) -> None:
        if got_429:
            self._429_count += 1
            self._success_streak = 0
            logger.info(
                "BRAIN rate limit observed count=%s current_concurrency=%s",
                self._429_count,
                self.max_concurrent,
            )
            if self.max_concurrent > 1:
                self.max_concurrent -= 1
                logger.info("Reducing BRAIN concurrency new_concurrency=%s", self.max_concurrent)
                print(f"[brain_api] 429 received, concurrency -> {self.max_concurrent}", flush=True)
        else:
            self._success_streak += 1
            self._429_count = 0
            if self._success_streak >= 10 and self.max_concurrent < 8:
                self.max_concurrent += 1
                self._success_streak = 0
                logger.info("Increasing BRAIN concurrency new_concurrency=%s", self.max_concurrent)
                print(f"[brain_api] Success streak, concurrency -> {self.max_concurrent}", flush=True)

    def get_with_retry(self, url: str, retries: int = 3, **kwargs) -> requests.Response:
        s = self._ensure_session()
        for attempt in range(retries):
            try:
                logger.debug("GET request attempt=%s/%s url=%s", attempt + 1, retries, url)
                resp = s.get(url, timeout=(10, 60), **kwargs)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.info("GET rate limited url=%s retry_after=%ss", url, retry_after)
                    time.sleep(retry_after)
                    self._adjust_concurrency(True)
                    continue
                if resp.status_code in (401, 403):
                    logger.info("GET auth expired status_code=%s url=%s; re-authenticating", resp.status_code, url)
                    print("[brain_api] Session expired, re-authenticating...", flush=True)
                    self.connect()
                    s = self.session  # type: ignore
                    continue
                if resp.status_code >= 400:
                    logger.info(
                        "GET returned error status_code=%s url=%s body_len=%s body_hash=%s",
                        resp.status_code,
                        url,
                        len(resp.text or ""),
                        _text_fingerprint(resp.text or ""),
                    )
                    logger.debug("GET error body url=%s body=%s", url, resp.text[:1000])
                return resp
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                logger.info("GET transient exception attempt=%s/%s url=%s error=%s", attempt + 1, retries, url, e)
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"GET {url} failed after {retries} retries")

    def post_with_retry(self, url: str, json_body: dict, retries: int = 3, **kwargs) -> requests.Response:
        s = self._ensure_session()
        for attempt in range(retries):
            try:
                logger.debug("POST request attempt=%s/%s url=%s", attempt + 1, retries, url)
                resp = s.post(url, json=json_body, timeout=(30, 300), **kwargs)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.info("POST rate limited url=%s retry_after=%ss", url, retry_after)
                    time.sleep(retry_after)
                    self._adjust_concurrency(True)
                    continue
                if resp.status_code in (401, 403):
                    logger.info("POST auth expired status_code=%s url=%s; re-authenticating", resp.status_code, url)
                    print("[brain_api] Session expired, re-authenticating...", flush=True)
                    self.connect()
                    s = self.session  # type: ignore
                    continue
                self._adjust_concurrency(False)
                if resp.status_code >= 400:
                    logger.info(
                        "POST returned error status_code=%s url=%s body_len=%s body_hash=%s",
                        resp.status_code,
                        url,
                        len(resp.text or ""),
                        _text_fingerprint(resp.text or ""),
                    )
                    logger.debug("POST error body url=%s body=%s", url, resp.text[:1000])
                return resp
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                logger.info("POST transient exception attempt=%s/%s url=%s error=%s", attempt + 1, retries, url, e)
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
        """Submit to /simulations, poll until complete, then fetch alpha metrics from /alphas/{alphaId}."""
        payload = self.build_payload(expression, settings)
        expr_hash = _expr_fingerprint(expression)
        logger.info(
            "Simulation submit start expr_hash=%s expr_len=%s settings_keys=%s",
            expr_hash,
            len(expression),
            sorted(payload["settings"].keys()),
        )
        resp = self.post_with_retry(f"{API_BASE}/simulations", payload)
        if resp.status_code != 201:
            logger.info(
                "Simulation submit failed status_code=%s expr_hash=%s body_len=%s body_hash=%s",
                resp.status_code,
                expr_hash,
                len(resp.text or ""),
                _text_fingerprint(resp.text or ""),
            )
            logger.debug("Simulation submit failure body expr_hash=%s body=%s", expr_hash, resp.text[:1000])
            return {"status": "ERROR", "error": resp.text[:500], "status_code": resp.status_code}

        location = resp.headers.get("Location", "")
        sim_id = location.rstrip("/").split("/")[-1]
        logger.info("Simulation created sim_id=%s expr_hash=%s", sim_id, expr_hash)

        sim_result = self._poll_simulation(sim_id)
        if sim_result.get("status") != "COMPLETE":
            logger.info(
                "Simulation finished non-complete sim_id=%s status=%s expr_hash=%s status_code=%s error_hash=%s",
                sim_id,
                sim_result.get("status"),
                expr_hash,
                sim_result.get("status_code"),
                _text_fingerprint(str(sim_result.get("error", ""))),
            )
            logger.debug("Simulation non-complete result sim_id=%s result=%s", sim_id, json.dumps(sim_result, ensure_ascii=False, default=str)[:2000])
            return sim_result

        alpha_id = sim_result.get("alpha_id", "")
        if not alpha_id:
            logger.info("Simulation complete without alpha_id sim_id=%s expr_hash=%s", sim_id, expr_hash)
            return {"status": "ERROR", "error": "No alpha ID returned", "sim_data": sim_result}

        # Fetch full alpha data to get sharpe/fitness/turnover from /alphas/{alphaId}
        alpha_data = self.get_alpha(alpha_id)
        if not alpha_data:
            # The simulation itself COMPLETED — we just couldn't retrieve the
            # metrics. Flag this as a distinct FETCH_ERROR so it is NOT recorded
            # as a SIM_ERROR (which would wrongly penalize the template) and can
            # be retried as a transient failure.
            logger.info("Could not fetch alpha metrics alpha_id=%s sim_id=%s expr_hash=%s", alpha_id, sim_id, expr_hash)
            return {
                "status": "FETCH_ERROR",
                "error": f"Could not fetch alpha {alpha_id}",
                "alpha_id": alpha_id,
            }

        is_data = alpha_data.get("is", {}) if isinstance(alpha_data, dict) else {}
        logger.info(
            "Simulation complete alpha_id=%s sharpe=%s fitness=%s turnover=%s expr_hash=%s",
            alpha_id,
            is_data.get("sharpe"),
            is_data.get("fitness"),
            is_data.get("turnover"),
            expr_hash,
        )
        return {"status": "COMPLETE", "alpha_id": alpha_id, "sim_data": alpha_data}

    def _poll_simulation(self, sim_id: str, timeout: int = 600) -> dict[str, Any]:
        start = time.time()
        logger.info("Polling simulation start sim_id=%s timeout=%ss", sim_id, timeout)
        last_status = None
        while time.time() - start < timeout:
            resp = self.get_with_retry(f"{API_BASE}/simulations/{sim_id}")
            if resp.status_code != 200:
                logger.info("Polling simulation non-200 sim_id=%s status_code=%s", sim_id, resp.status_code)
                time.sleep(8)
                continue
            data = resp.json()
            status = data.get("status", "UNKNOWN")
            if status != last_status:
                logger.info("Polling simulation status changed sim_id=%s status=%s", sim_id, status)
                last_status = status
            # COMPLETE and WARNING are both terminal SUCCESS states: the
            # simulation finished and produced an alpha record. WARNING only
            # flags non-fatal advisories (e.g. low coverage / high turnover);
            # the metrics are still real and the alpha is fetchable. Treating
            # WARNING as non-terminal made the loop spin until timeout and the
            # result was silently lost — handle it exactly like COMPLETE.
            if status in ("COMPLETE", "WARNING"):
                alpha_id = data.get("alpha", "")
                logger.info(
                    "Polling simulation terminal sim_id=%s status=%s alpha_id=%s",
                    sim_id, status, alpha_id,
                )
                return {"status": "COMPLETE", "alpha_id": alpha_id, "sim_data": data}
            if status in ("ERROR", "FAILED"):
                logger.info(
                    "Polling simulation failed sim_id=%s status=%s data_keys=%s",
                    sim_id,
                    status,
                    sorted(data.keys()) if isinstance(data, dict) else [],
                )
                logger.debug("Polling simulation failed data sim_id=%s data=%s", sim_id, json.dumps(data, ensure_ascii=False, default=str)[:2000])
                return {"status": "ERROR", "sim_data": data}
            time.sleep(5)
        logger.info("Polling simulation timeout sim_id=%s elapsed=%.1fs", sim_id, time.time() - start)
        return {"status": "TIMEOUT", "simulation_id": sim_id}

    def _is_retryable_sim_error(self, sim_result: dict[str, Any]) -> bool:
        """Return True for transient API/network failures worth retrying."""
        status = sim_result.get("status")
        status_code = sim_result.get("status_code")
        error = str(sim_result.get("error", "")).lower()

        if status == "TIMEOUT":
            # A TIMEOUT from _poll_simulation means the simulation was already
            # created; retrying would POST a duplicate simulation.
            return False
        if status == "FETCH_ERROR":
            # Simulation already COMPLETED and get_alpha has already retried the
            # metrics fetch internally (eventual consistency). Do NOT batch-retry
            # — that would re-POST a duplicate simulation and waste fuel.
            return False
        if isinstance(status_code, int) and status_code in {429, 500, 502, 503, 504}:
            return True
        transient_markers = (
            "429",
            "timeout",
            "timed out",
            "connection",
            "temporarily",
        )
        return any(marker in error for marker in transient_markers)

    def batch_simulate(
        self,
        candidates: list[dict],
        max_concurrent: int | None = None,
        max_retries: int = 2,
    ) -> list[dict[str, Any]]:
        """Simulate a batch of candidates with limited concurrency.

        Each candidate: {expression, settings, ...}
        Returns list of results with candidate info merged.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        concurrency = max_concurrent or self.max_concurrent
        results: list[dict[str, Any]] = []
        logger.info(
            "Batch simulate start candidate_count=%s concurrency=%s max_retries=%s",
            len(candidates),
            concurrency,
            max_retries,
        )

        def _run_one(idx: int, cand: dict) -> dict[str, Any]:
            expr = cand["expression"]
            expr_hash = _expr_fingerprint(expr)
            settings = cand.get("settings", {})
            for attempt in range(max_retries + 1):
                try:
                    sim_result = self.simulate(expr, settings)
                except Exception as e:
                    logger.info("Simulation raised exception batch_idx=%s attempt=%s expr_hash=%s error=%s", idx, attempt + 1, expr_hash, e)
                    sim_result = {"status": "ERROR", "error": str(e)}

                sim_result["attempts"] = attempt + 1
                retryable = self._is_retryable_sim_error(sim_result)
                if not retryable or attempt >= max_retries:
                    logger.info(
                        "Simulation candidate finished batch_idx=%s status=%s attempts=%s retryable=%s expr_hash=%s",
                        idx,
                        sim_result.get("status"),
                        sim_result.get("attempts"),
                        retryable,
                        expr_hash,
                    )
                    return {**cand, "sim_result": sim_result, "batch_idx": idx}

                sleep_s = min(30, 5 * (attempt + 1))
                logger.info(
                    "Simulation candidate retry batch_idx=%s attempt=%s/%s sleep=%ss status=%s status_code=%s error=%s expr_hash=%s",
                    idx,
                    attempt + 1,
                    max_retries,
                    sleep_s,
                    sim_result.get("status"),
                    sim_result.get("status_code"),
                    sim_result.get("error"),
                    expr_hash,
                )
                print(
                    f"  [retry] transient simulation error; retrying {attempt + 1}/{max_retries} "
                    f"after {sleep_s}s — {expr[:50]}",
                    flush=True,
                )
                time.sleep(sleep_s)

            return {**cand, "sim_result": sim_result, "batch_idx": idx}

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
        status_counts: dict[str, int] = {}
        for result in results:
            status = result.get("sim_result", {}).get("status", "?")
            status_counts[status] = status_counts.get(status, 0) + 1
        logger.info("Batch simulate complete candidate_count=%s status_counts=%s", len(candidates), status_counts)
        return results

    def batch_simulate_stream(
        self,
        candidates: list[dict],
        max_concurrent: int | None = None,
        max_retries: int = 2,
    ):
        """Stream simulation results as they complete (generator).

        Yields each result immediately when a simulation finishes,
        enabling streaming submit during batch processing.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        concurrency = max_concurrent or self.max_concurrent
        logger.info(
            "Batch simulate stream start candidate_count=%s concurrency=%s max_retries=%s",
            len(candidates),
            concurrency,
            max_retries,
        )

        def _run_one(idx: int, cand: dict) -> dict[str, Any]:
            expr = cand["expression"]
            expr_hash = _expr_fingerprint(expr)
            settings = cand.get("settings", {})
            for attempt in range(max_retries + 1):
                try:
                    sim_result = self.simulate(expr, settings)
                except Exception as e:
                    logger.info(
                        "Stream simulation raised exception batch_idx=%s attempt=%s expr_hash=%s error=%s",
                        idx,
                        attempt + 1,
                        expr_hash,
                        e,
                    )
                    sim_result = {"status": "ERROR", "error": str(e)}

                sim_result["attempts"] = attempt + 1
                retryable = self._is_retryable_sim_error(sim_result)
                if not retryable or attempt >= max_retries:
                    logger.info(
                        "Stream simulation candidate finished batch_idx=%s status=%s attempts=%s retryable=%s expr_hash=%s",
                        idx,
                        sim_result.get("status"),
                        sim_result.get("attempts"),
                        retryable,
                        expr_hash,
                    )
                    return {**cand, "sim_result": sim_result, "batch_idx": idx}

                sleep_s = min(30, 5 * (attempt + 1))
                logger.info(
                    "Stream simulation candidate retry batch_idx=%s attempt=%s/%s sleep=%ss status=%s status_code=%s error=%s expr_hash=%s",
                    idx,
                    attempt + 1,
                    max_retries,
                    sleep_s,
                    sim_result.get("status"),
                    sim_result.get("status_code"),
                    sim_result.get("error"),
                    expr_hash,
                )
                print(
                    f"  [retry] transient simulation error; retrying {attempt + 1}/{max_retries} "
                    f"after {sleep_s}s — {expr[:50]}",
                    flush=True,
                )
                time.sleep(sleep_s)

            return {**cand, "sim_result": sim_result, "batch_idx": idx}

        completed = 0
        total = len(candidates)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_run_one, i, cand): i
                for i, cand in enumerate(candidates)
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                status = result.get("sim_result", {}).get("status", "?")
                expr_short = result["expression"][:50]
                print(f"  [{completed}/{total}] {status} — {expr_short}", flush=True)
                yield result
        logger.info("Batch simulate stream complete candidate_count=%s", len(candidates))

    # ----------------------------------------------------------------- #
    # Metrics & PnL
    # ----------------------------------------------------------------- #
    def list_user_alphas(self, user_id: str = "self", limit: int = 100) -> list[dict[str, Any]]:
        """Fetch all user alphas from BRAIN with pagination."""
        alphas: list[dict[str, Any]] = []
        offset = 0
        logger.info("Fetching remote alpha list user_id=%s limit=%s", user_id, limit)
        while True:
            resp = self.get_with_retry(
                f"{API_BASE}/users/{user_id}/alphas",
                params={"limit": limit, "offset": offset},
            )
            if resp.status_code != 200:
                logger.info(
                    "Fetch remote alpha list failed user_id=%s offset=%s status_code=%s body_len=%s body_hash=%s",
                    user_id,
                    offset,
                    resp.status_code,
                    len(resp.text or ""),
                    _text_fingerprint(resp.text or ""),
                )
                raise RuntimeError(f"remote alpha list failed: status={resp.status_code} offset={offset}")
            try:
                data = resp.json()
            except Exception as e:
                logger.info(
                    "Fetch remote alpha list JSON parse failed user_id=%s offset=%s error=%s body_len=%s body_hash=%s",
                    user_id,
                    offset,
                    e,
                    len(resp.text or ""),
                    _text_fingerprint(resp.text or ""),
                )
                raise RuntimeError(f"remote alpha list JSON parse failed: offset={offset}") from e

            batch = data.get("results", data.get("alphas", [])) if isinstance(data, dict) else []
            if not isinstance(batch, list):
                logger.info(
                    "Fetch remote alpha list returned unexpected batch type user_id=%s offset=%s batch_type=%s",
                    user_id,
                    offset,
                    type(batch).__name__,
                )
                raise RuntimeError(f"remote alpha list unexpected batch type: {type(batch).__name__}")
            alphas.extend(a for a in batch if isinstance(a, dict))
            logger.info("Fetched remote alpha page user_id=%s offset=%s count=%s", user_id, offset, len(batch))
            if len(batch) < limit:
                break
            offset += limit

        status_counts: dict[str, int] = {}
        for alpha in alphas:
            status = str(alpha.get("status", "UNKNOWN"))
            status_counts[status] = status_counts.get(status, 0) + 1
        logger.info("Remote alpha list complete total=%s status_counts=%s", len(alphas), status_counts)
        return alphas

    def refresh_alpha_db_from_remote(self, db: dict[str, Any]) -> list[dict[str, Any]]:
        """Refresh local alpha DB statuses from the remote user alpha list.

        Returns the remote ACTIVE alpha objects so callers can build correlation
        baselines from the authoritative BRAIN state instead of stale local DB.
        """
        remote_alphas = self.list_user_alphas()
        db.setdefault("alphas", {})
        now = datetime.now(timezone.utc).isoformat()
        updated = 0
        created = 0
        for alpha in remote_alphas:
            alpha_id = alpha.get("id")
            if not alpha_id:
                continue
            is_data = alpha.get("is", {}) if isinstance(alpha.get("is"), dict) else {}
            existing = db["alphas"].setdefault(alpha_id, {})
            if not existing:
                created += 1
            before = {
                "status": existing.get("status"),
                "sharpe": existing.get("sharpe"),
                "fitness": existing.get("fitness"),
                "turnover": existing.get("turnover"),
            }
            existing.update(
                {
                    "status": alpha.get("status"),
                    "sharpe": is_data.get("sharpe", existing.get("sharpe")),
                    "fitness": is_data.get("fitness", existing.get("fitness")),
                    "turnover": is_data.get("turnover", existing.get("turnover")),
                    "remote_refreshed_at": now,
                }
            )
            if "expression" not in existing and alpha.get("regular"):
                existing["expression"] = alpha.get("regular")
            after = {
                "status": existing.get("status"),
                "sharpe": existing.get("sharpe"),
                "fitness": existing.get("fitness"),
                "turnover": existing.get("turnover"),
            }
            if before != after:
                updated += 1

        active = [a for a in remote_alphas if a.get("status") == "ACTIVE"]
        logger.info(
            "Refreshed alpha DB from remote remote_total=%s remote_active=%s created=%s updated=%s db_alpha_count=%s",
            len(remote_alphas),
            len(active),
            created,
            updated,
            len(db.get("alphas", {})),
        )
        return active

    def get_alpha(self, alpha_id: str, retries: int = 3, retry_sleep: float = 2.0) -> dict:
        """Fetch full alpha metrics.

        The simulation may report COMPLETE slightly before the alpha record is
        queryable (eventual consistency), so retry on empty/non-200 before
        giving up. Returns {} only after exhausting retries — callers treat that
        as a *fetch* failure (FETCH_ERROR), NOT a simulation error.
        """
        logger.info("Fetching alpha details alpha_id=%s retries=%s", alpha_id, retries)
        for attempt in range(retries + 1):
            try:
                resp = self.get_with_retry(f"{API_BASE}/alphas/{alpha_id}")
            except Exception as e:
                logger.info("Fetch alpha details exception alpha_id=%s attempt=%s/%s error=%s", alpha_id, attempt + 1, retries + 1, e)
                if attempt >= retries:
                    return {}
                time.sleep(retry_sleep * (attempt + 1))
                continue
            if resp.status_code == 200:
                data = resp.json()
                is_data = data.get("is", {}) if isinstance(data, dict) else {}
                logger.info(
                    "Fetched alpha details alpha_id=%s status=%s sharpe=%s fitness=%s turnover=%s",
                    alpha_id,
                    data.get("status") if isinstance(data, dict) else None,
                    is_data.get("sharpe"),
                    is_data.get("fitness"),
                    is_data.get("turnover"),
                )
                return data
            logger.info(
                "Fetch alpha details failed alpha_id=%s attempt=%s/%s status_code=%s body_len=%s body_hash=%s",
                alpha_id,
                attempt + 1,
                retries + 1,
                resp.status_code,
                len(resp.text or ""),
                _text_fingerprint(resp.text or ""),
            )
            logger.debug("Fetch alpha details failed body alpha_id=%s body=%s", alpha_id, resp.text[:1000])
            if attempt >= retries:
                return {}
            time.sleep(retry_sleep * (attempt + 1))
        return {}

    def fetch_pnl(self, alpha_id: str, retries: int = 3, retry_sleep: float = 2.0) -> list[float]:
        logger.info("Fetching alpha PnL alpha_id=%s retries=%s", alpha_id, retries)
        resp: requests.Response | None = None
        for attempt in range(retries + 1):
            try:
                resp = self.get_with_retry(f"{API_BASE}/alphas/{alpha_id}/recordsets/pnl")
            except Exception as e:
                logger.info("Fetch alpha PnL exception alpha_id=%s attempt=%s/%s error=%s", alpha_id, attempt + 1, retries + 1, e)
                if attempt >= retries:
                    return []
                time.sleep(retry_sleep * (attempt + 1))
                continue
            if resp.status_code == 200 and resp.text.strip():
                break
            logger.info(
                "Fetch alpha PnL empty/non-200 alpha_id=%s attempt=%s/%s status_code=%s text_chars=%s",
                alpha_id,
                attempt + 1,
                retries + 1,
                resp.status_code,
                len(resp.text or ""),
            )
            if attempt >= retries:
                return []
            time.sleep(retry_sleep * (attempt + 1))

        if resp is None:
            return []
        try:
            data = resp.json()
        except Exception as e:
            logger.info(
                "Fetch alpha PnL JSON parse failed alpha_id=%s error=%s body_len=%s body_hash=%s",
                alpha_id,
                e,
                len(resp.text or ""),
                _text_fingerprint(resp.text or ""),
            )
            logger.debug("Fetch alpha PnL JSON parse failed body alpha_id=%s body=%s", alpha_id, resp.text[:1000])
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
        logger.info("Fetched alpha PnL alpha_id=%s raw_records=%s parsed_records=%s", alpha_id, len(records), len(out))
        return out

    def submit_alpha(self, alpha_id: str) -> dict[str, Any]:
        """Submit alpha and poll for result."""
        logger.info("Submit alpha start alpha_id=%s", alpha_id)
        resp = self.post_with_retry(f"{API_BASE}/alphas/{alpha_id}/submit", json_body={})
        if resp.status_code not in (200, 201):
            logger.info(
                "Submit alpha failed alpha_id=%s status_code=%s body_len=%s body_hash=%s",
                alpha_id,
                resp.status_code,
                len(resp.text or ""),
                _text_fingerprint(resp.text or ""),
            )
            logger.debug("Submit alpha failed body alpha_id=%s body=%s", alpha_id, resp.text[:1000])
            return {"submitted": False, "status_code": resp.status_code, "text": resp.text[:300]}

        logger.info("Submit alpha accepted alpha_id=%s status_code=%s; polling status", alpha_id, resp.status_code)
        for poll_idx in range(30):
            time.sleep(10)
            alpha = self.get_alpha(alpha_id)
            status = alpha.get("status")
            checks = alpha.get("is", {}).get("checks", [])
            self_corr = next((c for c in checks if c.get("name") == "SELF_CORRELATION"), {})
            logger.info(
                "Submit alpha poll alpha_id=%s poll=%s status=%s self_corr_result=%s",
                alpha_id,
                poll_idx + 1,
                status,
                self_corr.get("result"),
            )
            if status == "ACTIVE":
                logger.info("Submit alpha active alpha_id=%s poll=%s", alpha_id, poll_idx + 1)
                return {"submitted": True, "status": "ACTIVE", "alpha": alpha}
            if self_corr.get("result") == "FAIL":
                logger.info("Submit alpha self-correlation failed alpha_id=%s status=%s", alpha_id, status)
                return {"submitted": True, "status": status, "self_correlation": "FAIL", "alpha": alpha}
        logger.info("Submit alpha still pending after polling alpha_id=%s polls=30", alpha_id)
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


def _align_returns(
    a: list[float], b: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Align two daily-return series for correlation.

    PnL series are cumulative and sorted ascending by date. Different alphas
    are simulated against the same data end date but may start on different
    dates (data availability), so the series share their most-recent tail but
    differ in length at the head. We align on the common overlapping tail
    rather than demanding identical lengths — the old exact-length check made
    the correlation gate fire almost never.
    """
    n = min(len(a), len(b))
    if n == 0:
        return np.array([]), np.array([])
    return np.array(a[-n:]), np.array(b[-n:])


def compute_correlation(
    new_pnl: list[float],
    db: dict[str, Any],
    min_records: int = 50,
    min_overlap: int = 50,
) -> list[dict[str, Any]]:
    if len(new_pnl) < min_records + 1:
        return []
    new_ret_full = daily_returns(new_pnl)
    results: list[dict[str, Any]] = []
    for old_id, old in db.get("alphas", {}).items():
        if old.get("status") != "ACTIVE" or not old.get("pnl"):
            continue
        old_ret_full = daily_returns(old["pnl"])
        new_ret, old_ret = _align_returns(new_ret_full, old_ret_full)
        # Need enough overlapping points for a meaningful correlation.
        if len(new_ret) < min_overlap:
            logger.info(
                "Skipping correlation: insufficient overlap old_alpha_id=%s overlap=%s new_len=%s old_len=%s",
                old_id, len(new_ret), len(new_ret_full), len(old_ret_full),
            )
            continue
        # corrcoef is undefined when either series is constant.
        if np.std(new_ret) == 0 or np.std(old_ret) == 0:
            logger.info("Skipping correlation: constant series old_alpha_id=%s", old_id)
            continue
        corr = float(np.corrcoef(new_ret, old_ret)[0, 1])
        if np.isnan(corr):
            logger.info("Skipping NaN correlation old_alpha_id=%s records=%s", old_id, len(old_ret))
            continue
        results.append({"alpha_id": old_id, "correlation": corr, "sharpe": old.get("sharpe"), "fitness": old.get("fitness")})
    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
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


_FIELD_VALIDATOR_CACHE: Any = None


def _get_field_categories() -> dict[str, str]:
    """Lazily build (once) the field-id -> data-category map used to classify
    expression fields into field_classes. Empty dict if unavailable."""
    global _FIELD_VALIDATOR_CACHE
    if _FIELD_VALIDATOR_CACHE is None:
        if FieldValidator is None or _FIELDS_PATH is None:
            _FIELD_VALIDATOR_CACHE = {}
        else:
            try:
                _FIELD_VALIDATOR_CACHE = FieldValidator(_FIELDS_PATH).field_categories
            except Exception:
                _FIELD_VALIDATOR_CACHE = {}
    return _FIELD_VALIDATOR_CACHE


def _fingerprint_candidate(candidate: dict) -> dict[str, Any]:
    """Structure fingerprint for a candidate, robust to missing infra.

    Falls back to a minimal shape (concept_id as ast_hash) when
    structure_fingerprint is unavailable so the write path never crashes.
    """
    expr = candidate.get("expression", "")
    if isinstance(expr, dict):  # remote alpha records store {"code": ...}
        expr = expr.get("code") or ""
    expr = str(expr)
    if structure_fingerprint is not None and expr:
        try:
            return structure_fingerprint(expr, _get_field_categories())
        except Exception:
            pass
    cid = candidate.get("concept_id") or candidate.get("template_id") or "unknown"
    return {"ast_hash": str(cid), "ops": [], "fields": [], "field_classes": [], "depth": 0}


def _verdict_to_failure_mode(
    verdict: str,
    sharpe: float | None,
    fitness: float | None,
    turnover: float | None,
    max_corr: float | None,
) -> str | None:
    """Classify why a non-SUBMIT result fell short (None for SUBMIT)."""
    if verdict == "SUBMIT":
        return None
    if sharpe is None:
        return "SIM_ERROR"
    if turnover is not None and turnover > 0.7:
        return "HIGH_TURNOVER"
    if max_corr is not None and abs(max_corr) >= 0.7:
        return "HIGH_CORR"
    if sharpe < 1.0:
        return "LOW_SHARPE"
    if fitness is not None and fitness < 1.0:
        return "LOW_FITNESS"
    return "OTHER"


def _new_rollup() -> dict[str, Any]:
    return {
        "tested": 0, "submit": 0, "observe": 0, "discard": 0,
        "sharpe_count": 0, "sum_sharpe": 0.0, "avg_sharpe": 0.0,
        "best_sharpe": None, "failure_modes": {}, "action": "explore",
        "ops": [], "field_classes": [], "examples": [],
    }


def _apply_to_rollup(roll: dict[str, Any], exp: dict[str, Any]) -> None:
    """Fold a single experiment record into a rollup bucket (in place)."""
    roll["tested"] += 1
    verdict = exp.get("verdict")
    if verdict == "SUBMIT":
        roll["submit"] += 1
    elif verdict == "OBSERVE":
        roll["observe"] += 1
    else:
        roll["discard"] += 1
    sharpe = (exp.get("is") or {}).get("sharpe")
    if sharpe is not None:
        roll["sharpe_count"] += 1
        roll["sum_sharpe"] += sharpe
        roll["avg_sharpe"] = roll["sum_sharpe"] / roll["sharpe_count"]
        if roll["best_sharpe"] is None or sharpe > roll["best_sharpe"]:
            roll["best_sharpe"] = sharpe
    fm = exp.get("failure_mode")
    if fm:
        roll["failure_modes"][fm] = roll["failure_modes"].get(fm, 0) + 1
    # Union of structural metadata for human readability.
    if exp.get("ops"):
        roll["ops"] = sorted(set(roll["ops"]) | set(exp["ops"]))
    if exp.get("field_classes"):
        roll["field_classes"] = sorted(set(roll["field_classes"]) | set(exp["field_classes"]))
    # Keep a few example expressions for the LLM prompt / debugging.
    ex = exp.get("expr")
    if ex and ex not in roll["examples"] and len(roll["examples"]) < 3:
        roll["examples"].append(ex)


def _finalize_rollup_action(roll: dict[str, Any]) -> None:
    """Derive a consume-side action once a bucket has enough evidence."""
    tested = roll["tested"]
    if tested < 5:
        roll["action"] = "explore"
        return
    pass_rate = (roll["submit"] + roll["observe"]) / tested if tested else 0.0
    if roll["submit"] == 0 and roll["observe"] == 0:
        roll["action"] = "skip"
    elif pass_rate < 0.2:
        roll["action"] = "deprioritize"
    else:
        roll["action"] = "explore"


def recompute_rollups(lessons: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the derived `rollups` cache from the append-only `experiments`.

    rollups is ALWAYS exactly derived(experiments) — this is the single function
    that establishes that invariant. Called after every append; can also be run
    standalone to repair the cache. Buckets: by_ast (structure), by_field_class
    (data category), by_decay (the main tunable param).
    """
    by_ast: dict[str, Any] = {}
    by_fc: dict[str, Any] = {}
    by_decay: dict[str, Any] = {}
    for exp in lessons.get("experiments", []):
        ast = exp.get("ast_hash")
        if ast:
            _apply_to_rollup(by_ast.setdefault(ast, _new_rollup()), exp)
        for fc in exp.get("field_classes", []) or []:
            _apply_to_rollup(by_fc.setdefault(fc, _new_rollup()), exp)
        decay = (exp.get("settings") or {}).get("decay")
        if decay is not None:
            _apply_to_rollup(by_decay.setdefault(str(decay), _new_rollup()), exp)
    for roll in by_ast.values():
        _finalize_rollup_action(roll)
    for roll in by_fc.values():
        _finalize_rollup_action(roll)
    for roll in by_decay.values():
        _finalize_rollup_action(roll)
    lessons["rollups"] = {"by_ast": by_ast, "by_field_class": by_fc, "by_decay": by_decay}
    return lessons["rollups"]


def update_lessons_from_result(
    lessons: dict[str, Any],
    candidate: dict,
    sim_result: dict,
    max_corr: float | None = None,
) -> None:
    """Update lessons.json with results from a simulation."""
    template_id = candidate.get("template_id", "unknown")
    # A FETCH_ERROR means the simulation COMPLETED but its metrics couldn't be
    # retrieved (infra/eventual-consistency issue). It says nothing about the
    # template's quality, so do not record it — counting it would wrongly inflate
    # the template's sim_errors and bias its action toward skip/deprioritize.
    if sim_result.get("status") == "FETCH_ERROR":
        logger.info("Skipping lessons update for FETCH_ERROR template_id=%s alpha_id=%s", template_id, sim_result.get("alpha_id"))
        return
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
            "sharpe_count": 0, "fitness_count": 0, "sim_errors": 0,
            "best": None, "failure_modes": {}, "action": "expand",
            "notes": "",
        }

    p = patterns[template_id]
    # Backward-compat: older lessons.json may lack the per-metric counters.
    p.setdefault("sharpe_count", 0)
    p.setdefault("fitness_count", 0)
    p.setdefault("sim_errors", 0)
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

    # Update averages over VALID samples only. `tested` counts every result
    # (incl. SIM_ERROR with sharpe=None); using it as the divisor would
    # systematically bias avg_sharpe/avg_fitness downward. Track per-metric
    # counts instead.
    if sharpe is None:
        p["sim_errors"] += 1
    if sharpe is not None:
        old_count = p["sharpe_count"]
        p["sharpe_count"] += 1
        if old_count > 0:
            p["avg_sharpe"] = (p["avg_sharpe"] * old_count + sharpe) / p["sharpe_count"]
        else:
            p["avg_sharpe"] = sharpe

    if fitness is not None:
        old_count = p["fitness_count"]
        p["fitness_count"] += 1
        if old_count > 0:
            p["avg_fitness"] = (p["avg_fitness"] * old_count + fitness) / p["fitness_count"]
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
            "avg_sharpe": 0.0, "verdict": "neutral", "notes": "", "count": 0,
            "sharpe_count": 0,
        })
        entry.setdefault("sharpe_count", 0)
        entry["count"] += 1
        if sharpe is not None:
            old_count = entry["sharpe_count"]
            entry["sharpe_count"] += 1
            if old_count > 0:
                entry["avg_sharpe"] = (entry["avg_sharpe"] * old_count + sharpe) / entry["sharpe_count"]
            else:
                entry["avg_sharpe"] = sharpe
            if entry["sharpe_count"] >= 3:
                if entry["avg_sharpe"] >= 1.5:
                    entry["verdict"] = "prefer"
                elif entry["avg_sharpe"] < 0.8:
                    entry["verdict"] = "deprioritize"

    # --- v2: append an immutable experiment fact + recompute derived rollups ---
    # `action` here is the quality_filter verdict (SUBMIT / OBSERVE / DISCARD).
    fp = _fingerprint_candidate(candidate)
    expr_text = candidate.get("expression", "")
    if isinstance(expr_text, dict):
        expr_text = expr_text.get("code") or ""
    experiment = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "alpha_id": sim_result.get("alpha_id"),
        "ast_hash": fp["ast_hash"],
        "concept_id": candidate.get("concept_id") or candidate.get("template_id"),
        "ops": fp["ops"],
        "field_classes": fp["field_classes"],
        "depth": fp["depth"],
        "source": candidate.get("source", "template"),
        "expr": str(expr_text)[:200],
        "settings": {
            "decay": candidate.get("settings", {}).get("decay"),
            "neutralization": candidate.get("settings", {}).get("neutralization"),
            "universe": candidate.get("settings", {}).get("universe"),
        },
        "is": {"sharpe": sharpe, "fitness": fitness, "turnover": turnover},
        "max_corr": max_corr,
        "verdict": action,
        "failure_mode": _verdict_to_failure_mode(action, sharpe, fitness, turnover, max_corr),
    }
    lessons.setdefault("experiments", []).append(experiment)
    recompute_rollups(lessons)

    lessons["last_updated"] = datetime.now(timezone.utc).isoformat()
