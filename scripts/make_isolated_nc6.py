"""Create stage3_1_isolated_no_check6.jsonl.

Combines:
  1. stage3_1_isolated_kept.jsonl   — all 7 checks passed
  2. rows from stage3_1_isolated_audit.jsonl that passed checks 1–5
     but failed check6 (naturalness / lexical artifacts) only.

The intent is to evaluate on a slightly larger isolated set by relaxing
the naturalness check, since check6 removes a large fraction of isolated
candidates.
"""

import json
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
KEPT_FILE   = DATA / "stage3_1_isolated_kept.jsonl"
AUDIT_FILE  = DATA / "stage3_1_isolated_audit.jsonl"
OUT_FILE    = DATA / "stage3_1_isolated_no_check6.jsonl"

CHECK6_FIELD = "check6_naturalness_and_lexical_artifacts"
OTHER_CHECKS = [
    "check1_intent_consistency",
    "check2_policy_order",
    "check3_parallelism_cover_framing_consistency",
    "check4_task_consistency",
    "check5_harm_domain_consistency",
]


def load_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    kept = load_jsonl(KEPT_FILE)
    kept_ids = {row["triplet_record_id"] for row in kept}
    print(f"Kept rows (all checks passed): {len(kept)}")

    audit = load_jsonl(AUDIT_FILE)
    check6_only = []
    for row in audit:
        tid = row.get("triplet_record_id")
        if tid in kept_ids:
            continue                              # already included
        s31 = row.get("stage3_1", {})
        if s31.get("accepted", False):
            continue                              # accepted rows are in kept
        # Passes checks 1–5?
        if not all(s31.get(c, False) for c in OTHER_CHECKS):
            continue
        # Fails check6?
        if s31.get(CHECK6_FIELD, True):
            continue
        check6_only.append(row)

    combined = kept + check6_only
    print(f"Check6-only rejected (re-included): {len(check6_only)}")
    print(f"Combined total: {len(combined)}")

    with open(OUT_FILE, "w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
