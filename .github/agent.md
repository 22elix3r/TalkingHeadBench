# TalkingHeadBench — Project Context for Claude

## What This Is
A three-sub-environment OpenEnv benchmark for evaluating AI agents on
talking-head LoRA pipelines. No live generation anywhere. All signals are
pre-extracted from user artifacts. All graders are deterministic.

For the full environment spec, use the `talkingheadbench` skill — it contains
all Pydantic schemas, grader logic, and worked examples.

## Tech Stack
- Python 3.10+
- PyTorch (canonical weight decomposition)
- Pydantic v2 (all data models)
- scipy (entropy calculations)
- OpenCV + MediaPipe + ArcFace (signal extraction)

## Conventions
- All Pydantic models live in `src/schemas/` — one file per sub-environment
- Shared grader utilities live in `src/utils/grader_utils.py` — never duplicate
  `set_f1()` or `jaccard_similarity()` inline
- `canonicalize_lora_factors()` lives in `src/utils/canonical.py` — all weight
  signal extraction must call this first, never operate on raw A/B matrices
- Type annotations are mandatory everywhere — no bare `dict` in function signatures
- `Optional[X]` means the field can be absent; always handle the None case
- Node files follow the naming convention: `nodeN_descriptive_name.py`

## Critical Constraints
- NEVER add live model inference or generation to any node — the whole design
  premise is pre-extracted signals only
- NEVER hardcode thresholds — all thresholds come from the test set annotations
  in `tests/test_set/`
- `token_position_to_phoneme` is loaded from tokenizer config, NOT derived from
  weights — if you see code trying to infer this from weight statistics, flag it
- `override_decision` is a 3-way Literal ("not_applicable" | "declined" |
  "applied"), NOT a bool — reject any refactor that changes this

## Testing
Run unit tests: `pytest tests/unit/ -v`
Run grader smoke test: `python src/evaluate.py --dry-run`

## Build Order (if starting from scratch)
1. `src/utils/grader_utils.py` — shared utilities first
2. `src/utils/canonical.py` — canonical decomposition
3. `src/schemas/` — all Pydantic models (define the contract before implementations)
4. `src/envs/` — node implementations, one sub-env at a time
5. `src/pipeline.py` — wire nodes together
6. `src/evaluate.py` — scoring harness last