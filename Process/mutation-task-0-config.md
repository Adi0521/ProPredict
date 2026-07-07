# Mutation Master Plan — Task 0: Config fixes

**Date:** 2026-07-06
**Source:** `mutation-plans/Process-plan-master.md` Task 0 (merges hardening + mutation-scan docs)
**Scope:** config-only; no code imports these vars yet.

## What was done

### 0a — `AGENT_MODEL` default (`config.py`)
`"claude-opus-4-6"` → `"claude-sonnet-5"`. The old string is not a valid current model,
so any agent-loop call falling back to the default (i.e. `AGENT_MODEL` unset) was hitting
a model-not-found error.

**Decision — `.env.example` left unchanged.** The plan said to also fix `.env.example`
"if it references the old string." It does not: `.env.example` has
`AGENT_MODEL=deepseek-v4-flash`, which matches the DeepSeek testing backend documented in
`CLAUDE.local.md`. The code default is the sane fallback when the var is unset; the
example intentionally stays on DeepSeek for local dev.

### 0b — `AGENT_MAX_MUTATIONS` (`config.py` + `.env.example`)
Added `AGENT_MAX_MUTATIONS = int(os.getenv("AGENT_MAX_MUTATIONS", 3))` beside the other
`AGENT_*` flags. Caps `apply_mutation` calls per agent session, separate from
`AGENT_MAX_ITERATIONS` (which bounds all tool calls). Consumed later in Task 3. Mirrored
in `.env.example` with a comment.

### 0c — `PROTEINMPNN_PATH` / `PROTEINMPNN_MODEL_NAME` (`config.py` + `.env.example`)
Added beside the Stage F external-tool paths (`INSANE_PATH`, `GNINA_BIN`), since
ProteinMPNN is a git-clone tool, not pip-installable — same pattern as `insane.py`.
Defaults: path `""`, model `v_48_020` (ProteinMPNN's own default). Consumed later in
Task 1 (`orchestrator/mutation_scan.py`).

## Verification
- `python -c "import config"` succeeds; all four vars resolve
  (`AGENT_MODEL`, `AGENT_MAX_MUTATIONS=3`, `PROTEINMPNN_PATH=''`,
  `PROTEINMPNN_MODEL_NAME=v_48_020`).
- `pytest tests/test_boltz.py` in the `ProPredict` conda env: **11 passed, 2 skipped**
  (skips are GPU integration tests). No regression.

## Next
Task 1 — `orchestrator/mutation_scan.py` (ProteinMPNN scorer, standalone), then Task 2
validation gate before any `apply_mutation` work.
