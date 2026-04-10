import argparse
import csv
import json
from typing import Dict, Iterable, List

from pipeline_common import write_jsonl


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("data", "rows", "items", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        raise ValueError(f"Unsupported JSON structure in {path}")
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def load_csv(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def detect_prompt_field(row: Dict, candidates: Iterable[str]) -> str:
    for field in candidates:
        value = row.get(field)
        if value is not None and str(value).strip():
            return field
    raise ValueError(f"Could not find a prompt field in row: {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an AdvBench export into seeds.jsonl format.")
    parser.add_argument("--input", required=True, help="CSV or JSON file containing AdvBench rows.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--prompt-fields",
        nargs="+",
        default=["goal", "prompt", "instruction", "text"],
        help="Candidate source fields for the unsafe prompt text.",
    )
    parser.add_argument("--id-prefix", default="advbench", help="Prefix for generated ids when source ids are missing.")
    args = parser.parse_args()

    if args.input.endswith(".csv"):
        rows = load_csv(args.input)
    elif args.input.endswith(".json"):
        rows = load_json(args.input)
    else:
        raise ValueError("Unsupported input format. Use .csv or .json")

    output_rows = []
    for index, row in enumerate(rows, start=1):
        prompt_field = detect_prompt_field(row, args.prompt_fields)
        source_id = row.get("id") or row.get("idx") or row.get("uuid") or f"{args.id_prefix}-{index:05d}"
        output_rows.append(
            {
                "id": str(source_id),
                "source_dataset": "advbench",
                "source_split": row.get("split", "unknown"),
                "unsafe_seed": str(row[prompt_field]).strip(),
            }
        )

    write_jsonl(args.output, output_rows)


if __name__ == "__main__":
    main()
