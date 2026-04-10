import argparse
from typing import Any, Dict, List

from pipeline_common import (
    STAGE2_POLICY,
    add_common_io_args,
    deterministic_rng,
    ensure_stage_metadata,
    load_jsonl,
    progress_iter,
    resolve_harm_domain,
    resolve_task_type,
    resolve_topic_summary,
    stable_record_id,
    weighted_pick,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 deterministic dual-use assignment.")
    add_common_io_args(parser)
    parser.add_argument("--task-type-field", default="stage1_1.task_type", help="Dot path for task type.")
    parser.add_argument("--preferred-weight", type=float, default=0.7, help="Probability mass for the first two options.")
    parser.add_argument("--limit", type=int, help="Only process the first N input rows.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    output_rows: List[Dict[str, Any]] = []
    for row in progress_iter(rows, desc="stage2", disable=args.no_progress):
        base_record_id = stable_record_id(row)
        topic_summary = str(resolve_topic_summary(row))
        harm_domain = str(resolve_harm_domain(row))
        task_type = str(resolve_task_type(row, args.task_type_field))
        if task_type not in STAGE2_POLICY:
            raise ValueError(f"Unsupported task type '{task_type}'. Allowed: {sorted(STAGE2_POLICY)}")

        policy = STAGE2_POLICY[task_type]
        for target_a_level in ("A2", "A3"):
            rng = deterministic_rng(base_record_id, task_type, target_a_level, "stage2")
            benign_cover = weighted_pick(
                policy["benign_cover_categories"],
                rng,
                preferred_count=2,
                preferred_weight=args.preferred_weight,
            )
            framing_style = weighted_pick(
                policy["framing_styles"],
                rng,
                preferred_count=2,
                preferred_weight=args.preferred_weight,
            )

            enriched = ensure_stage_metadata(
                row,
                topic_summary=topic_summary,
                harm_domain=harm_domain,
                task_type=task_type,
            )
            enriched["dual_use_record_id"] = f"{base_record_id}_{target_a_level.lower()}"
            enriched["stage2"] = {
                "target_a_level": target_a_level,
                "benign_cover_category": benign_cover,
                "framing_style": framing_style,
            }
            output_rows.append(enriched)

    write_jsonl(args.output, output_rows)


if __name__ == "__main__":
    main()
