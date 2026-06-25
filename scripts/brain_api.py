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


def load_lessons() -> dict[str, Any]:
    if LESSONS_PATH.exists():
        lessons = json.loads(LESSONS_PATH.read_text(encoding="utf-8"))
        logger.info("Loaded lessons path=%s pattern_count=%s", LESSONS_PATH, len(lessons.get("patterns", {})))
        return lessons
    logger.info("Lessons not found; initializing empty lessons path=%s", LESSONS_PATH)
    return {"patterns": {}, "param_insights": {}, "version": 1}


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
            logger.info("Could not fetch alpha metrics alpha_id=%s sim_id=%s expr_hash=%s", alpha_id, sim_id, expr_hash)
            return {"status": "ERROR", "error": f"Could not fetch alpha {alpha_id}", "alpha_id": alpha_id}

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
            if status == "COMPLETE":
                alpha_id = data.get("alpha", "")
                logger.info("Polling simulation complete sim_id=%s alpha_id=%s", sim_id, alpha_id)
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

    def get_alpha(self, alpha_id: str) -> dict:
        logger.info("Fetching alpha details alpha_id=%s", alpha_id)
        resp = self.get_with_retry(f"{API_BASE}/alphas/{alpha_id}")
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
            "Fetch alpha details failed alpha_id=%s status_code=%s body_len=%s body_hash=%s",
            alpha_id,
            resp.status_code,
            len(resp.text or ""),
            _text_fingerprint(resp.text or ""),
        )
        logger.debug("Fetch alpha details failed body alpha_id=%s body=%s", alpha_id, resp.text[:1000])
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
