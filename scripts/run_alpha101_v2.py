"""Alpha101-inspired batch simulation v2 — focused on fundamentals, fixed timeouts.
Usage:
    python3 scripts/run_alpha101_v2.py
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
# Fundamental-focused alphas (v2 — skip pure tech, expand fundamentals)
# Based on lessons: roe_trend (zq9Wm0kK) worked: Sharpe=2.01, Fitness=1.32, TO=6.3%
# All pure tech failed. Focus on group_rank + ts_rank templates.
# ============================================================
ALPHAS = [
    # --- ROE/Profitability variations ---
    {
        "name": "roe_trend_v2_252",
        "expression": "group_rank(ts_rank(operating_income / equity, 252), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "roe_trend_short_win",
        "expression": "group_rank(ts_rank(operating_income / equity, 63), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "ebitda_margin_trend",
        "expression": "group_rank(ts_rank(ebitda / sales, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "gross_margin_trend",
        "expression": "group_rank(ts_rank(gross_profit / sales, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "revenue_growth",
        "expression": "group_rank(ts_rank(revenue_growth, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "oi_assets_trend",
        "expression": "group_rank(ts_rank(operating_income / assets, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    # --- Analyst expectations ---
    {
        "name": "est_eps_yeild_252",
        "expression": "group_rank(ts_rank(est_eps / close, 252), industry)",
        "decay": 2,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "est_revenue_yield",
        "expression": "group_rank(ts_rank(est_revenue / close, 126), industry)",
        "decay": 2,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "est_ebitda_yield",
        "expression": "group_rank(ts_rank(est_ebitda / close, 126), industry)",
        "decay": 2,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "est_fcf_yield",
        "expression": "group_rank(ts_rank(est_fcf / close, 126), industry)",
        "decay": 4,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    # --- FCF / Cash flow ---
    {
        "name": "fcf_equity_252",
        "expression": "group_rank(ts_rank(free_cash_flow_reported_value / equity, 252), industry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "cashflow_revenue",
        "expression": "group_rank(ts_rank(cashflow_from_operations / revenue, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    # --- Balance sheet / Quality ---
    {
        "name": "equity_growth",
        "expression": "group_rank(ts_rank(book_value_per_share_growth, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "debt_equity_ratio",
        "expression": "group_rank(ts_rank(-(total_liabilities / total_equity), 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "asset_turnover",
        "expression": "group_rank(ts_rank(sales / total_assets, 126), subindustry)",
        "decay": 0,
        "truncation": 0.08,
        "neutralization": "SUBINDUSTRY",
        "nanHandling": "ON",
    },
    # --- Mixed (tech + fundamental, with high decay to control TO) ---
    {
        "name": "short_term_reversal_mix",
        "expression": "0.3 * rank(-(close / open - 1)) + 0.7 * rank(ts_rank(operating_income / equity, 126))",
        "decay": 25,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "tech_value_mix_v2",
        "expression": "0.3 * rank(-ts_std_dev(returns, 20)) + 0.7 * group_rank(ts_rank(operating_income / equity, 126), subindustry)",
        "decay": 10,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
        "nanHandling": "ON",
    },
    {
        "name": "volume_signal_mix",
        "expression": "0.3 * rank(-ts_delta(volume, 5) / ts_std_dev(volume, 20)) + 0.7 * group_rank(ts_rank(est_eps / close, 126), industry)",
        "decay": 15,
        "truncation": 0.08,
        "neutralization": "INDUSTRY",
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
    # Increase timeout
    session.timeout = (30, 120)
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


def simulate_with_retry(session: requests.Session, alpha: dict, max_retries: int = 3) -> dict:
    payload = build_payload(alpha)
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = session.post(f"{API_BASE}/simulations", json=payload)
            if resp.status_code != 201:
                return {"error": f"simulate_failed: {resp.status_code}", "detail": resp.text[:300]}
            sim_id = resp.headers["Location"].rstrip("/").split("/")[-1]
            print(f"    sim_id={sim_id} (attempt {attempt+1})", end="", flush=True)

            # Poll with longer timeout
            start = time.time()
            while time.time() - start < 900:  # 15 min per alpha
                data = session.get(f"{API_BASE}/simulations/{sim_id}").json()
                status = data.get("status", "UNKNOWN")
                if status == "COMPLETE":
                    alpha_id = data["alpha"]
                    print(f" -> {alpha_id}", end="", flush=True)
                    return {"sim_id": sim_id, "alpha_id": alpha_id, "status": "COMPLETE"}
                if status in ("ERROR", "FAILED"):
                    return {"sim_id": sim_id, "status": "ERROR", "detail": str(data.get("message", ""))[:200]}
                time.sleep(8)
                print(".", end="", flush=True)
            return {"sim_id": sim_id, "status": "TIMEOUT"}

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
            last_error = str(e)
            print(f"\n    Retry {attempt+1}/{max_retries} after error: {last_error[:80]}")
            time.sleep(5 * (attempt + 1))
            continue

    return {"error": f"all_retries_failed: {last_error}"}


def get_metrics(session: requests.Session, alpha_id: str) -> dict:
    try:
        resp = session.get(f"{API_BASE}/alphas/{alpha_id}")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def submit_alpha(session: requests.Session, alpha_id: str) -> dict:
    try:
        sub = session.post(f"{API_BASE}/alphas/{alpha_id}/submit")
        print(f"    submit -> {sub.status_code}", end="", flush=True)
        if sub.status_code not in (200, 201):
            return {"submitted": False, "status_code": sub.status_code}

        for _ in range(30):
            time.sleep(10)
            resp = session.get(f"{API_BASE}/alphas/{alpha_id}")
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
    except Exception as e:
        return {"submitted": False, "error": str(e)}


def main():
    session = create_session()
    results = []

    for i, alpha in enumerate(ALPHAS, 1):
        name = alpha["name"]
        expr_short = alpha["expression"][:60]
        print(f"\n[{i}/{len(ALPHAS)}] {name}: {expr_short}...")

        try:
            sim_result = simulate_with_retry(session, alpha)
            if sim_result.get("status") != "COMPLETE":
                print(f"\n    SKIP: {sim_result.get('status')} {sim_result.get('detail', '') or sim_result.get('error', '')}")
                results.append({"name": name, "expression": alpha["expression"], "sim": sim_result})
                time.sleep(5)
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

            # Submit threshold: fitness >= 1.0 and sharpe >= 1.25
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

        time.sleep(5)

    # Save
    out_path = SKILL_DIR / "alpha101_v2_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")

    # Summary
    active = [r for r in results if r.get("submission", {}).get("status") == "ACTIVE"]
    submitted = [r for r in results if r.get("submission", {}).get("submitted") and r.get("submission", {}).get("status") != "ACTIVE"]
    good_metrics = [r for r in results if r.get("fitness", 0) >= 1.0 and r.get("sharpe", 0) >= 1.25]
    errors = [r for r in results if "error" in r or (r.get("sim", {}).get("status") in ("ERROR", "TIMEOUT"))]

    print(f"\n=== Summary ===")
    print(f"Total: {len(results)}")
    print(f"ACTIVE: {len(active)}")
    print(f"Submitted (not ACTIVE): {len(submitted)}")
    print(f"Passed metrics (not submitted): {len(good_metrics) - len(active)}")
    print(f"Errors/timeouts: {len(errors)}")

    if active:
        print("\n=== ACTIVE Alphas ===")
        for r in active:
            print(f"  {r['name']}: {r['alpha_id']} Sharpe={r['sharpe']:.2f} Fitness={r['fitness']:.2f} TO={r['turnover']*100:.1f}%")

    if good_metrics:
        print("\n=== Passed Metrics ===")
        for r in good_metrics:
            sub = r.get("submission", {})
            status = sub.get("status", "not_submitted")
            print(f"  {r['name']}: Sharpe={r['sharpe']:.2f} Fitness={r['fitness']:.2f} TO={r['turnover']*100:.1f}% -> {status}")


if __name__ == "__main__":
    main()
