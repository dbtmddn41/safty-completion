# Safety Completion Pipeline

This repository contains Python scripts for the staged dataset construction flow:

- `scripts/stage1_metadata_generation.py`
- `scripts/stage1_1_metadata_verification.py`
- `scripts/stage2_dual_use_assignment.py`
- `scripts/stage3_triplet_generation.py`
- `scripts/stage3_1_triplet_filtering.py`

All scripts use JSONL input and JSONL output.

## Expected Input Shape

At minimum, each seed row should contain:

```json
{"id": "seed-001", "unsafe_seed": "original unsafe prompt text"}
```

`unsafe_seed` is configurable with `--seed-field`.

## Environment

Stage 1 uses Claude on Vertex AI:

- `VERTEX_ACCESS_TOKEN`, or Application Default Credentials, or an authenticated `gcloud` session
- `--project-id`
- `--location`
- `--model` using the Vertex model ID, not the display name

Example valid model ID:

- `claude-sonnet-4@20250514`

Stage 1.1 uses Gemini API:

- `GEMINI_API_KEY`
- `--model` if you want something other than the default

Both model IDs are passed through directly, so use the exact identifier available in your environment for Claude on Vertex and Gemini API.

All stage scripts show progress by default. If `tqdm` is installed, they use a proper progress bar. Otherwise they fall back to a simple stderr counter. Use `--no-progress` to disable it.
Use `--limit N` to do a short trial run on only the first `N` rows.

## Example

If you are starting from the original seed prompts and want to run the full pipeline:

```bash
python scripts/stage1_metadata_generation.py \
  --input data/seeds.jsonl \
  --output data/stage1.jsonl \
  --project-id YOUR_PROJECT \
  --location us-central1 \
  --model YOUR_VERTEX_CLAUDE_MODEL
```

```bash
python scripts/stage1_1_metadata_verification.py \
  --input data/stage1.jsonl \
  --output data/stage1_1_kept.jsonl \
  --audit-output data/stage1_1_audit.jsonl
```

```bash
python scripts/stage2_dual_use_assignment.py \
  --input data/stage1_1_kept.jsonl \
  --output data/stage2_dual_use.jsonl
```

```bash
python scripts/stage3_triplet_generation.py \
  --input data/stage2_dual_use.jsonl \
  --output data/stage3_triplets.jsonl \
  --project-id YOUR_PROJECT \
  --location us-central1 \
  --model YOUR_VERTEX_CLAUDE_MODEL
```

```bash
python scripts/stage3_1_triplet_filtering.py \
  --input data/stage3_triplets.jsonl \
  --output data/stage3_1_kept.jsonl \
  --audit-output data/stage3_1_audit.jsonl
```

If you already received externally prepared Stage 1 / 1.1 outputs in the newer flat schema, you can start directly from Stage 2:

```bash
python scripts/stage2_dual_use_assignment.py \
  --input data/stage1_gemini_verified.jsonl \
  --output data/stage2_dual_use.jsonl
```

`stage2`, `stage3`, and `stage3.1` now accept both:

- the original nested schema with `stage1.*` and `stage1_1.*`
- the newer flat schema with fields such as `topic_summary`, `gemini_harm_domain`, and `gemini_task_type`

When Stage 2 reads the flat schema, it writes compatibility `stage1` and `stage1_1` objects into its output so downstream stages continue to see a stable shape.

## Temp Seeds

`data/seeds.jsonl` is scaffolded as a temporary AdvBench-shaped input file.

To replace it with a real AdvBench export:

```bash
python scripts/convert_advbench_to_seeds.py \
  --input path/to/advbench.csv \
  --output data/seeds.jsonl
```

## Notes

- The current recommendation is:
  - keep `stage1_metadata_generation.py` and `stage1_1_metadata_verification.py` as the legacy "raw-seed to metadata" path
  - use `data/stage1_gemini_verified.jsonl` as the handoff artifact when Stage 1 / 1.1 was run elsewhere
  - begin local processing from Stage 2 when working with the new delivered data
- The allowed task types default to:
  - `Explanation`
  - `Planning and Procedure`
  - `Troubleshooting`
  - `Optimization`
  - `Transformation`
- The allowed harm domains default to:
  - `Hazardous Agent Use`
  - `Physical Harm and Weapons`
  - `Hate and Harassment`
  - `Cyber Compromise`
  - `Fraud and Deception`
  - `Privacy and Data Misuse`
  - `Evasion and Circumvention`
- The allowed harm domains can be overridden with `--harm-domains-file`.
- Stage 2 is deterministic. Each accepted row produces two outputs: one `A2` copy and one `A3` copy.
- Stage 2 includes an `Explanation` policy branch using:
  - benign covers: `Education / learning`, `Research / analysis`, `Journalistic / historical / documentary interest`, `Defensive protection / safety`
  - framing styles: `Direct information request`, `Reflective / analytical request`, `Scenario-based / hypothetical request`
- Stage 3 generates `k=5` triplets per Stage 2 row by default.
- Stage 3 uses the base temperature for the first 2 generations and the high temperature for the remaining 3 generations by default.
- Stage 3.1 keeps only rows whose judge output marks the full triplet as accepted, and can also emit a full audit file.
