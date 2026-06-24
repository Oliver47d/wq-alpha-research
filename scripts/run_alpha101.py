"""Alpha101-inspired batch simulation and submit for WQ BRAIN.
v2: Focus on fundamental variants (based on v1 results: only roe_trend passed).
    Drop pure technical signals. Increase timeout. Better retry logic.
Usage:
    python3 scripts/run_alpha101.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
CREDENTIAL_PATH = SKILL_DIR / "credential.txt"
API_BASE = "https://api.worldquantbrain.com"

HEADERS = {
    "Accept": "application/json;version=2.0",
    "Content-Type": "application/json",
}

# ============================================================
# v2: Focus on fundamental alpha variants
# Key insight from v1: only group_rank(ts_rank(operating_income/equity,126), subindustry) passed.
# Pure technical signals all failed. Mix signals also underperformed.
# ============================================================
ALPHAS = [
    # --- ROE/ROA variants ---
    {
        "name": "roe_trend_252",
        "expression": "group_rank(ts_rank(return_equity, 252), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "roa_trend",
        "expression": "group_rank(ts_rank(return_assets, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "ebitda_margin_trend",
        "expression": "group_rank(ts_rank(ebitda / assets, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "ebitda_yield",
        "expression": "group_rank(ts_rank(ebitda / equity, 126), industry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    # --- Analyst EPS/Sales variants ---
    {
        "name": "analyst_eps_yield_252",
        "expression": "group_rank(ts_rank(est_eps / close, 252), industry)",
        "decay": 4,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "analyst_sales_yield",
        "expression": "group_rank(ts_rank(est_sales / close, 126), industry)",
        "decay": 4,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "analyst_bookvalue_yield",
        "expression": "group_rank(ts_rank(est_bookvalue_ps / close, 252), industry)",
        "decay": 4,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "eps_growth_trend",
        "expression": "group_rank(ts_rank(eps / ts_delay(eps, 252) - 1, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    # --- FCF variants ---
    {
        "name": "fcf_yield",
        "expression": "group_rank(ts_rank(free_cash_flow_reported_value / equity, 126), industry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "fcf_per_share",
        "expression": "group_rank(ts_rank(free_cash_flow_per_share / close, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "operating_cf_yield",
        "expression": "group_rank(ts_rank(cash_flow_from_operations / equity, 126), industry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    # --- Quality/Growth ---
    {
        "name": "sales_growth",
        "expression": "group_rank(ts_rank(sales_growth, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "bookvalue_ps_trend",
        "expression": "group_rank(ts_rank(bookvalue_ps / close, 252), industry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "earnings_yield",
        "expression": "group_rank(ts_rank(eps / close, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    # --- Multi-factor combinations (proven to work) ---
    {
        "name": "roe_eps_combo",
        "expression": "0.5 * group_rank(ts_rank(operating_income / equity, 126), subindustry) + 0.5 * group_rank(ts_rank(est_eps / close, 126), industry)",
        "decay": 2,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "quality_value_combo",
        "expression": "0.5 * group_rank(ts_rank(return_equity, 126), subindustry) + 0.5 * group_rank(ts_rank(ebitda / equity, 126), industry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    # --- Analyst sentiment (with careful decay) ---
    {
        "name": "eps_revision",
        "expression": "group_rank(ts_rank(est_eps - ts_delay(est_eps, 20), 60), industry)",
        "decay": 4,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "sales_revision",
        "expression": "group_rank(ts_rank(est_sales - ts_delay(est_sales, 20), 60), industry)",
        "decay": 4,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    # --- Liabilities/Quality (leverage) ---
    {
        "name": "leverage_inverse",
        "expression": "group_rank(ts_rank(-liabilities / assets, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
]


def load_credentials() -> tuple[str, str]:
    env_user = os.getenv("WQ_BRAIN_USERNAME")
    env_password = os.getenv("WQ_BRAIN_PASSWORD")
    if env_user and env_password:
        return env_user, env_password
    for p in [CREDENTIAL_PATH, Path.cwd() / "credential.txt"]:
        if p.exists():
            username, password = json.loads(p.read_text(encoding="utf-8"))
            return str(username), str(password)
    raise FileNotFoundError("BRAIN credentials not found.")


def create_session() -> requests.Session:
    username, password = load_credentials()
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
    session.headers.update(HEADERS)
    resp = session.post(f"{API_BASE}/authentication")
    if resp.status_code != 201:
        raise RuntimeError(f"Auth failed: {resp.status_code} {resp.text}")
    print(f"Auth OK ({resp.status_code})")
    return session


def build_payload(alpha: dict) -> dict:
    return {
        "type": "REGULAR",
        "settings": {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": alpha["decay"],
            "neutralization": alpha["neutralization"],
            "truncation": alpha["truncation"],
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": alpha.get("nanHandling", "ON"),
            "maxTrade": "OFF",
            "maxPosition": "OFF",
            "language": "FASTEXPR",
            "visualization": False,
        },
        "regular": alpha["expression"],
    }


def simulate(session: requests.Session, alpha: dict, retries: int = 2) -> dict:
    payload = build_payload(alpha)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            resp = session.post(f"{API_BASE}/simulations", json=payload, timeout=(15, 120))
            if resp.status_code != 201:
                last_error = f"post_failed: {resp.status_code}"
                time.sleep(3)
                continue
            sim_id = resp.headers["Location"].rstrip("/").split("/")[-1]
            print(f"    sim_id={sim_id}", end="", flush=True)

            start = time.time()
            while time.time() - start < 1200:  # 20 min timeout
                data = session.get(f"{API_BASE}/simulations/{sim_id}", timeout=(15, 60)).json()
                status = data.get("status", "UNKNOWN")
                if status == "COMPLETE":
                    alpha_id = data["alpha"]
                    print(f" -> {alpha_id}", end="", flush=True)
                    return {"sim_id": sim_id, "alpha_id": alpha_id, "status": "COMPLETE"}
                if status in ("ERROR", "FAILED"):
                    return {"sim_id": sim_id, "status": "ERROR", "detail": str(data.get("message", ""))[:200]}
                time.sleep(8)
                print(".", end="", flush=True)
            # Timeout on polling
            last_error = "timeout_polling"
            time.sleep(5)
        except Exception as e:
            last_error = str(e)
            time.sleep(5)
    return {"error": f"simulate_failed_after_retries: {last_error}"}


def get_metrics(session: requests.Session, alpha_id: str) -> dict:
    resp = session.get(f"{API_BASE}/alphas/{alpha_id}", timeout=(15, 60))
    if resp.status_code != 200:
        return {}
    return resp.json()


def submit_alpha(session: requests.Session, alpha_id: str) -> dict:
    sub = session.post(f"{API_BASE}/alphas/{alpha_id}/submit", timeout=(15, 60))
    print(f"    submit -> {sub.status_code}", end="", flush=True)
    if sub.status_code not in (200, 201):
        return {"submitted": False, "status_code": sub.status_code}

    for _ in range(30):
        time.sleep(10)
        resp = session.get(f"{API_BASE}/alphas/{alpha_id}", timeout=(15, 60))
        if resp.status_code != 200:
            continue
        alpha = resp.json()
        status = alpha.get("status")
        print(f" -> {status}", end="", flush=True)
        if status == "ACTIVE":
            return {"submitted": True, "status": "ACTIVE", "alpha": alpha}
        checks = alpha.get("is", {}).get("checks", [])
        sc = next((c for c in checks if c.get("name") == "SELF_CORRELATION"), {})
        if sc.get("result") == "FAIL":
            return {"submitted": True, "status": status, "self_correlation": "FAIL"}
    return {"submitted": True, "status": "PENDING"}


def main():
    session = create_session()
    results = []

    for i, alpha in enumerate(ALPHAS, 1):
        name = alpha["name"]
        expr_short = alpha["expression"][:60]
        print(f"\n[{i}/{len(ALPHAS)}] {name}: {expr_short}...")

        try:
            sim_result = simulate(session, alpha)
            if sim_result.get("status") != "COMPLETE":
                print(f"\n    SKIP: {sim_result.get('status')} {sim_result.get('detail', '')}")
                results.append({"name": name, "expression": alpha["expression"], "sim": sim_result})
                continue

            alpha_id = sim_result["alpha_id"]
            metrics = get_metrics(session, alpha_id)
            is_ = metrics.get("is", {})
            sharpe = is_.get("sharpe", 0) or 0
            fitness = is_.get("fitness", 0) or 0
            turnover = is_.get("turnover", 1) or 1
            returns_val = is_.get("returns", 0) or 0
            drawdown = is_.get("drawdown", 0) or 0

            print(f"\n    Sharpe={sharpe:.2f} Fitness={fitness:.2f} TO={turnover*100:.1f}% Ret={returns_val:.3f} DD={drawdown:.3f}")

            # Check IS checks
            checks = is_.get("checks", [])
            failed = [c["name"] for c in checks if c.get("result") == "FAIL"]
            if failed:
                print(f"    FAILS: {failed}")

            entry = {
                "name": name,
                "expression": alpha["expression"],
                "decay": alpha["decay"],
                "neutralization": alpha["neutralization"],
                "alpha_id": alpha_id,
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover": turnover,
                "returns": returns_val,
                "drawdown": drawdown,
                "checks_failed": failed,
            }

            # Submit if passes thresholds
            if fitness >= 1.0 and sharpe >= 1.25 and turnover <= 0.50:
                sub_result = submit_alpha(session, alpha_id)
                entry["submission"] = sub_result
                print("")
            else:
                entry["submission"] = {"submitted": False, "reason": "metrics_threshold"}
                print("    SKIP submit (below threshold)")

            results.append(entry)

        except Exception as e:
            print(f"\n    ERROR: {e}")
            results.append({"name": name, "expression": alpha["expression"], "error": str(e)})

        time.sleep(3)

    # Save results
    out_path = SKILL_DIR / "alpha101_v2_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")

    # Summary
    active = [r for r in results if r.get("submission", {}).get("status") == "ACTIVE"]
    submitted = [r for r in results if r.get("submission", {}).get("submitted") and r.get("submission", {}).get("status") != "ACTIVE"]
    skipped = [r for r in results if not r.get("submission", {}).get("submitted", True)]
    errors = [r for r in results if "error" in r or r.get("sim", {}).get("status") == "ERROR"]
    good_metrics = [r for r in results if r.get("fitness", 0) >= 1.0 and r.get("sharpe", 0) >= 1.25]

    print(f"\n=== Summary ===")
    print(f"Total: {len(results)}")
    print(f"ACTIVE: {len(active)}")
    print(f"Submitted (not ACTIVE): {len(submitted)}")
    print(f"Skipped (below threshold): {len(skipped)}")
    print(f"Errors: {len(errors)}")
    print(f"Passed metrics: {len(good_metrics)}")

    if active:
        print("\n=== ACTIVE Alphas ===")
        for r in active:
            print(f"  {r['name']}: {r['alpha_id']} Sharpe={r['sharpe']:.2f} Fitness={r['fitness']:.2f} TO={r['turnover']*100:.1f}%")

    if good_metrics:
        print("\n=== Passed Metrics (all) ===")
        for r in good_metrics:
            sub = r.get("submission", {})
            status = sub.get("status", "not_submitted")
            print(f"  {r['name']}: Sharpe={r['sharpe']:.2f} Fitness={r['fitness']:.2f} TO={r['turnover']*100:.1f}% -> {status}")


if __name__ == "__main__":
    main()
