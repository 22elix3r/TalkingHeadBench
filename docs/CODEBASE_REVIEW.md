# TalkingHeadBench Codebase Review

## Scope
- Workspace: `talkingheadbench`
- Review mode: static architecture/code audit + test/readiness audit
- Ordered targets reviewed: 53 files (52 present, 1 missing)
- Snapshot from prior verified run state in this session:
  - Schema validation: subenv1/subenv2/subenv3 passed
  - Test suite: `361 passed`

## Executive Summary
- Core benchmark architecture is strong: deterministic graders, explicit handoff coupling, rich schema contracts, and broad smoke/unit coverage.
- The biggest risks are integration-facing rather than algorithmic:
  - stale public usage docs in `README.md`
  - `--subenv all` auto-detection mismatch with supported wrapped Sub-env 1 payloads
  - evaluator display logic drift from canonical grader behavior for one Sub-env 2 dimension
- Tooling/dependency hygiene can be improved with a small pass (`scripts/export_test_set.py` presence contract and optional dependency extras).

## Findings (Ordered by Severity)

| Severity | ID | Finding | Evidence | Impact | Recommendation |
|---|---|---|---|---|---|
| High | THB-001 | README usage snippet calls the wrong public API and assumes a float return. | `README.md:51`, `README.md:69`, `README.md:70` vs `src/pipeline.py:31`, `src/pipeline.py:43`, `src/pipeline.py:686`, `src/pipeline.py:1064` | New users copy-pasting the README sample can hit runtime errors (`run_episode` expects typed args, and returns `EpisodeResult`, not float). | Update README usage to either: (1) call `run_episode_from_bundle(artifact_bundle)`, or (2) show typed `run_episode(...)` call and print `result.final_score`. |
| High | THB-002 | Sub-environment auto-detection is inconsistent with wrapped Sub-env 1 schema support. | `src/evaluate.py:41`, `src/evaluate.py:43`, `src/evaluate.py:132`, `src/evaluate.py:135`, `src/evaluate.py:152`; `scripts/validate_annotations.py:47`, `scripts/validate_annotations.py:49`, `scripts/validate_annotations.py:64` | `--subenv all` can fail on valid wrapped Sub-env 1 cases (`{"image_obs": ..., "proposed_config": ...}`), despite coercers already supporting that shape. | Extend `detect_subenv` in both validator/evaluator to detect wrapped Sub-env 1 (`image_obs` key). |
| Medium | THB-003 | Sub-env 2 reasoning breakdown display in evaluator can diverge from grader truth when `expected_reasoning_elements` is empty. | `src/evaluate.py:493`, `src/evaluate.py:497`, `src/evaluate.py:499` vs `src/envs/subenv2/node6_grader.py:119`, `src/envs/subenv2/node6_grader.py:123` | Final score is correct, but printed dimension diagnostics can be misleading during analysis/debugging. | Mirror Node 6 logic exactly in evaluator breakdown path (remove `if kw_elements` gate or explicitly emulate grader behavior). |
| Medium | THB-004 | Requested target `scripts/export_test_set.py` is missing from workspace. | Ordered inventory item #31 | Automation or team expectations based on that script name can break. | Either add the script, or remove/update references/contracts that imply it exists. |
| Low | THB-005 | Optional extraction dependencies are used but not declared in `requirements.txt` as optional extras. | `requirements.txt:1`, `requirements.txt:8`; `src/envs/subenv2/node4_clip_extractor.py:330`; `scripts/extract_subenv2_signals.py:246`; `scripts/extract_subenv3_signals.py:152` | Clean environments may silently fall back to reduced-fidelity extraction paths, creating reproducibility variance. | Add `extras` documentation (e.g., `[full-extraction]`) for `insightface`, `transformers`, and any related optional stack. |

## LLM/Agent Swap Readiness Checklist

| Item | Status | Notes |
|---|---|---|
| Strong typed interfaces at env boundaries | Pass | Pydantic schemas are comprehensive across subenv1/2/3 and ground truth. |
| Deterministic grader behavior | Pass | Rule-based scoring and utility functions are explicit and tested. |
| Explicit inter-subenv coupling | Pass | Handoffs are documented and implemented in `pipeline.py`. |
| Evaluator/validator schema gatekeeping | Pass with caveat | Works well for explicit subenv mode; `all` mode has wrapped Sub-env 1 detection gap (THB-002). |
| Public onboarding docs match runtime API | Fail | README entrypoint example is stale (THB-001). |
| Dependency contract clear for full-fidelity extraction | Partial | Optional dependency behavior exists, but package contract is implicit (THB-005). |

## Test Coverage Gaps

| Gap | Current Coverage | Risk | Suggested Test |
|---|---|---|---|
| `--subenv all` with wrapped Sub-env 1 payload | Not explicitly covered | Regression risk for mixed-format datasets | Add evaluator + validator smoke tests using wrapped Sub-env 1 observation (`image_obs`). |
| Evaluator breakdown parity with Node 6 | Not explicitly covered | Misleading diagnostics despite correct final score | Add assertion test comparing `_run_subenv2` dimension display logic against grader branch behavior. |
| Script CLIs (`convert/extract/generate worksheet`) | Mostly untested end-to-end | Data-prep workflow regressions may go unnoticed | Add minimal CLI smoke tests with synthetic fixtures and temporary directories. |
| README runnable example validity | Not tested | New-user onboarding breakage | Add a doctest-like smoke check for README snippet API shape. |

## Dependency Audit

### Declared Core Dependencies
- `numpy`
- `torch`
- `pydantic`
- `scipy`
- `safetensors`
- `opencv-python`
- `mediapipe`
- `pytest`

### Observed Optional Imports in Runtime/Script Paths
- `insightface` (ArcFace-based embedding extraction path)
- `transformers` (tokenizer reconstruction path in Sub-env 3 extraction script)

### Assessment
- Core benchmark/evaluator/test functionality is installable from current `requirements.txt`.
- Full extraction quality paths are partly conditional on optional packages.
- Recommendation: formalize optional extras and document expected behavior in fallback mode.

## Technical Debt Notes

| Area | Observation | Impact | Recommendation |
|---|---|---|---|
| Duplicated coercion/validation logic | Similar subenv detection and coercion logic appears in both evaluator and validator scripts. | Fixes can drift between files. | Consolidate shared detection/coercion utilities in one module imported by both scripts. |
| Large script surface without direct tests | Extraction/worksheet scripts are substantial and carry many branches. | Harder to refactor safely. | Add lightweight fixture-based CLI tests for core happy/error paths. |
| Memory/context drift risk | Repository memory had at least one stale statement about missing validator script. | Future automation/review confusion. | Keep repository memory synced whenever tooling files are added/removed. |

## Ordered File Audit (Requested Sequence)

| # | File | Status | LOC | Notes |
|---:|---|---|---:|---|
| 1 | src/utils/__init__.py | PRESENT | 0 | Runtime benchmark code |
| 2 | src/utils/grader_utils.py | PRESENT | 57 | Runtime benchmark code |
| 3 | src/utils/canonical.py | PRESENT | 151 | Runtime benchmark code |
| 4 | src/schemas/__init__.py | PRESENT | 0 | Runtime benchmark code |
| 5 | src/schemas/subenv1.py | PRESENT | 165 | Runtime benchmark code |
| 6 | src/schemas/subenv2.py | PRESENT | 191 | Runtime benchmark code |
| 7 | src/schemas/subenv3.py | PRESENT | 255 | Runtime benchmark code |
| 8 | src/schemas/ground_truth.py | PRESENT | 86 | Runtime benchmark code |
| 9 | src/envs/__init__.py | PRESENT | 0 | Runtime benchmark code |
| 10 | src/envs/subenv1/__init__.py | PRESENT | 0 | Runtime benchmark code |
| 11 | src/envs/subenv1/node1_image_diagnostician.py | PRESENT | 121 | Runtime benchmark code |
| 12 | src/envs/subenv1/node2_param_anomaly.py | PRESENT | 172 | Runtime benchmark code |
| 13 | src/envs/subenv1/node3_grader.py | PRESENT | 281 | Runtime benchmark code |
| 14 | src/envs/subenv2/__init__.py | PRESENT | 0 | Runtime benchmark code |
| 15 | src/envs/subenv2/node4_clip_extractor.py | PRESENT | 484 | Runtime benchmark code |
| 16 | src/envs/subenv2/node5_disposition.py | PRESENT | 100 | Runtime benchmark code |
| 17 | src/envs/subenv2/node6_grader.py | PRESENT | 149 | Runtime benchmark code |
| 18 | src/envs/subenv3/__init__.py | PRESENT | 0 | Runtime benchmark code |
| 19 | src/envs/subenv3/node7_weight_extractor.py | PRESENT | 501 | Runtime benchmark code |
| 20 | src/envs/subenv3/node8_phoneme_risk.py | PRESENT | 266 | Runtime benchmark code |
| 21 | src/envs/subenv3/node9_grader.py | PRESENT | 165 | Runtime benchmark code |
| 22 | src/__init__.py | PRESENT | 0 | Runtime benchmark code |
| 23 | src/pipeline.py | PRESENT | 1300 | Runtime benchmark code |
| 24 | src/evaluate.py | PRESENT | 816 | Runtime benchmark code |
| 25 | scripts/convert_captions.py | PRESENT | 352 | CLI/data-prep utility |
| 26 | scripts/extract_subenv1_signals.py | PRESENT | 828 | CLI/data-prep utility |
| 27 | scripts/extract_subenv2_signals.py | PRESENT | 928 | CLI/data-prep utility |
| 28 | scripts/extract_subenv3_signals.py | PRESENT | 584 | CLI/data-prep utility |
| 29 | scripts/generate_annotation_worksheet.py | PRESENT | 828 | CLI/data-prep utility |
| 30 | scripts/validate_annotations.py | PRESENT | 371 | CLI/data-prep utility |
| 31 | scripts/export_test_set.py | MISSING | - | Listed target missing in workspace |
| 32 | conftest.py | PRESENT | 8 | Project configuration/documentation |
| 33 | tests/unit/test_canonical.py | PRESENT | 272 | Test coverage artifact |
| 34 | tests/unit/test_graders.py | PRESENT | 235 | Test coverage artifact |
| 35 | tests/unit/test_schemas.py | PRESENT | 935 | Test coverage artifact |
| 36 | tests/unit/test_node4_extractor.py | PRESENT | 280 | Test coverage artifact |
| 37 | tests/unit/test_node7_extractor.py | PRESENT | 155 | Test coverage artifact |
| 38 | tests/unit/test_subenv1.py | PRESENT | 162 | Test coverage artifact |
| 39 | tests/unit/test_subenv2.py | PRESENT | 283 | Test coverage artifact |
| 40 | tests/unit/test_subenv3.py | PRESENT | 164 | Test coverage artifact |
| 41 | tests/smoke/test_grader_arithmetic.py | PRESENT | 466 | Test coverage artifact |
| 42 | tests/smoke/test_node1_boundaries.py | PRESENT | 408 | Test coverage artifact |
| 43 | tests/smoke/test_node2_boundaries.py | PRESENT | 506 | Test coverage artifact |
| 44 | tests/smoke/test_node5_boundaries.py | PRESENT | 221 | Test coverage artifact |
| 45 | tests/smoke/test_node7_deep.py | PRESENT | 136 | Test coverage artifact |
| 46 | tests/smoke/test_node8_boundaries.py | PRESENT | 247 | Test coverage artifact |
| 47 | tests/smoke/test_pipeline_e2e.py | PRESENT | 294 | Test coverage artifact |
| 48 | tests/smoke/test_schema_roundtrip.py | PRESENT | 756 | Test coverage artifact |
| 49 | tests/smoke/test_evaluate_cli.py | PRESENT | 299 | Test coverage artifact |
| 50 | tests/smoke/test_pipeline_bundle.py | PRESENT | 162 | Test coverage artifact |
| 51 | requirements.txt | PRESENT | 8 | Project configuration/documentation |
| 52 | .gitignore | PRESENT | 65 | Project configuration/documentation |
| 53 | README.md | PRESENT | 95 | Project configuration/documentation |

## Pre-Submission Checklist

| Check | Status | Notes |
|---|---|---|
| Subenv1/2/3 schema validation commands | Pass | Verified in-session before this review stage. |
| Full test sweep (`pytest tests/ -q`) | Pass | `361 passed` in-session. |
| Critical runtime scoring paths deterministic | Pass | Rule-based graders and utility tests present. |
| Public usage docs consistent with current API | Fail | See THB-001. |
| Auto-detect interoperability for mixed case formats | Fail | See THB-002. |
| Tooling completeness for expected script inventory | Partial | `scripts/export_test_set.py` missing. |

## Overall Assessment
- **Readiness**: Strong core benchmark implementation with excellent deterministic testing posture.
- **Primary blockers before external-facing submission**:
  1. fix README API mismatch (THB-001)
  2. fix auto-detection behavior for wrapped Sub-env 1 in `all` mode (THB-002)
- **Recommended near-term hardening**:
  1. align evaluator diagnostics with canonical grader behavior (THB-003)
  2. formalize optional dependency extras and document fallback modes (THB-005)
  3. either add or formally de-scope `scripts/export_test_set.py` (THB-004)

## Document Stats
- Line count: 158
