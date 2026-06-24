#!/usr/bin/env python3
"""submit_batch.py — Simplified batch submit for pre-filtered candidates.

Accepts a JSON list of candidates (with alpha_id and quality verdict) and
submits those marked SUBMIT to the BRAIN API.

Usage:
    python3 submit_batch.py --input candidates.json
    python3 submit_batch.py --alpha-id <alpha_id>
    python3 submit_batch.py --input candidates.json --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
ALPHA_DB_PATH = SKILL_DIR / "alpha_db.json"


def load_alpha_db() -> dict[str, Any]:
    if ALPHA_DB_PATH.exists():
        return json.loads(ALPHA_DB_PATH.read_text(encoding="utf-8"))
    return {"alphas": {}, "last_update": None, "version": 1}


def save_alpha_db(db: dict[str, Any]) -> None:
    from datetime import datetime, timezone
    db["last_update"] = datetime.now(timezone.utc).isoformat()
    ALPHA_DB_PATH.write_text(json.dumps(db, indent=2, default=str), encoding="utf-8")


def submit_from_list(client, candidates: list[dict], dry_run: bool = False) -> list[dict]:
    """Submit candidates that have quality verdict SUBMIT.

    Each candidate should have: alpha_id, quality (SUBMIT/OBSERVE/DISCARD)
    """
    db = load_alpha_db()
    results = []

    for cand in candidates:
        quality = cand.get("quality", "DISCARD")
        alpha_id = cand.get("alpha_id")

        if quality != "SUBMIT" or not alpha_id:
            results.append({**cand, "submit_result": "skipped"})
            continue

        if dry_run:
            print(f"  [dry-run] Would submit {alpha_id}: {cand.get('expression', '')[:60]}", flush=True)
            results.append({**cand, "submit_result": "dry_run"})
            continue

        print(f"  Submitting {alpha_id}...", flush=True)
        result = client.submit_alpha(alpha_id)
        results.append({**cand, "submit_result": result})

        # Update alpha_db
        if alpha_id in db.get("alphas", {}):
            db["alphas"][alpha_id]["status"] = result.get("status", "UNKNOWN")
            db["alphas"][alpha_id]["submit_result"] = result

        # Log
        if result.get("submitted"):
            status = result.get("status", "?")
            print(f"    → Submitted, status: {status}", flush=True)
            if result.get("self_correlation") == "FAIL":
                print(f"    → SELF_CORRELATION FAIL", flush=True)
        else:
            print(f"    → Submit failed: {result.get('text', '?')[:100]}", flush=True)

    if not dry_run:
        save_alpha_db(db)

    return results


def submit_single(client, alpha_id: str, dry_run: bool = False) -> dict:
    """Submit a single alpha by ID."""
    if dry_run:
        print(f"  [dry-run] Would submit {alpha_id}", flush=True)
        return {"alpha_id": alpha_id, "submit_result": "dry_run"}

    print(f"  Submitting {alpha_id}...", flush=True)
    result = client.submit_alpha(alpha_id)

    db = load_alpha_db()
    if alpha_id in db.get("alphas", {}):
        db["alphas"][alpha_id]["status"] = result.get("status", "UNKNOWN")
        db["alphas"][alpha_id]["submit_result"] = result
        save_alpha_db(db)

    if result.get("submitted"):
        print(f"    → Submitted, status: {result.get('status', '?')}", flush=True)
    else:
        print(f"    → Submit failed: {result.get('text', '?')[:100]}", flush=True)

    return {"alpha_id": alpha_id, "submit_result": result}


def main():
    parser = argparse.ArgumentParser(description="Submit pre-filtered alpha candidates")
    parser.add_argument("--input", type=str, help="JSON file with candidate list")
    parser.add_argument("--alpha-id", type=str, help="Submit a single alpha by ID")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be submitted without actually submitting")
    args = parser.parse_args()

    if not args.input and not args.alpha_id:
        parser.error("Must provide --input or --alpha-id")

    # Connect to BRAIN
    from brain_api import BrainClient
    client = BrainClient()
    client.connect()

    try:
        if args.alpha_id:
            result = submit_single(client, args.alpha_id, dry_run=args.dry_run)
            print(json.dumps(result, indent=2, default=str))
        else:
            candidates = json.loads(Path(args.input).read_text(encoding="utf-8"))
            print(f"Loaded {len(candidates)} candidates from {args.input}", flush=True)

            submit_count = sum(1 for c in candidates if c.get("quality") == "SUBMIT")
            print(f"  {submit_count} marked SUBMIT, {len(candidates) - submit_count} skipped", flush=True)

            results = submit_from_list(client, candidates, dry_run=args.dry_run)

            # Save results
            output_path = Path(args.input).with_suffix(".submitted.json")
            output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            print(f"\nResults saved to {output_path}", flush=True)

            # Summary
            submitted = sum(1 for r in results if r.get("submit_result", {}).get("submitted"))
            failed = sum(1 for r in results if r.get("submit_result") == "skipped")
            print(f"Summary: {submitted} submitted, {failed} skipped", flush=True)
    finally:
        if client.session:
            client.session.close()


if __name__ == "__main__":
    main()
