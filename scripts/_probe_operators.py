#!/usr/bin/env python3
"""One-off probe: fetch BRAIN's official /operators list and diff against our
hardcoded KNOWN_OPERATORS whitelist. Read-only, no simulations spent."""
import json
from brain_api import BrainClient, API_BASE
from generate_candidates import KNOWN_OPERATORS

c = BrainClient()
s = c._ensure_session() if c.session else None
if s is None:
    # force auth
    c.__enter__() if hasattr(c, "__enter__") else None
    s = c._ensure_session()

resp = s.get(f"{API_BASE}/operators")
print("HTTP", resp.status_code, "len", len(resp.text))
data = resp.json()
# data may be a list of dicts or a dict
if isinstance(data, dict):
    items = data.get("results") or data.get("operators") or []
else:
    items = data

official = set()
meta = {}
for it in items:
    if isinstance(it, dict):
        name = it.get("name") or it.get("operator")
        if name:
            official.add(name)
            meta[name] = {k: it.get(k) for k in ("category", "scope", "definition", "level") if k in it}
    elif isinstance(it, str):
        official.add(it)

print("official operator count:", len(official))
print("our KNOWN_OPERATORS count:", len(KNOWN_OPERATORS))

in_ours_not_official = sorted(KNOWN_OPERATORS - official)
in_official_not_ours = sorted(official - KNOWN_OPERATORS)

print("\n=== IN OUR WHITELIST BUT NOT IN OFFICIAL (illegal / drifted) ===")
for op in in_ours_not_official:
    print("  ", op)

print("\n=== IN OFFICIAL BUT NOT IN OUR WHITELIST (missed opportunities) ===")
for op in in_official_not_ours:
    print("  ", op)

# save full official list for reference
with open("_official_operators.json", "w") as f:
    json.dump(sorted(official), f, indent=1)
print("\nsaved full official list -> scripts/_official_operators.json")
