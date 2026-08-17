# Plan — Hardening Pass (apply_mutation, stale-clash bug, ligand/membrane tests)

**Context for whoever picks this up (human or Claude Code):** This plan was written after
directly cloning and reading the current `main` branch of
https://github.com/Adi0521/ProPredict — not from `ROADMAP.md` or `nextsteps.md`, both of
which are stale in places. Specifically:

- `nextsteps.md` Phase 4.1–4.3 (requirements split, docker platform lock, gitignore
  cleanup) are **already done** on `main`. Do not redo them. (`pip install --dry-run -r
  requirements.txt` resolves cleanly; conda-only deps already live in
  `environment-conda.yml`; `docker-compose.yml` has no `platform:` lock; `.gitignore`
  already excludes `__pycache__`/artifacts.)
- Phases 4.4–4.6 are done and documented in `Process/`.
- Phase 4.7 (`apply_mutation` tool) is **not done** — confirmed by grep, no such tool
  exists in `orchestrator/agent.py`.

Per `CLAUDE.md`'s own workflow rules: do these **one at a time**, propose the diff, get
sign-off, then move to the next. Don't batch all three into one commit.

## Decisions (confirmed, no longer open)

- **Schema:** add `mutations_applied` to `PostProcessingResult` (see Task 0).
- **Mutation cap:** separate explicit limit, not just `AGENT_MAX_ITERATIONS` (see Task 0
  and the cap-check in Task 1's `apply_mutation` branch).
- **`AGENT_MODEL` default:** fix to `"claude-sonnet-5"` (see Task 0).
- **Order:** Task 0 (config fix, trivial) → Task 1 (apply_mutation) → Task 2 (clash bug)
  → Task 3 (tests).

---

## Task 0 — Config fixes (do first, small, unblocks nothing else but is trivial)

### 0a. Fix `AGENT_MODEL` default
In `config.py`:

```python
# Before
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-4-6")

# After
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5")
```

`"claude-opus-4-6"` is not a valid current model string — any agent-loop call relying on
the default (i.e. `AGENT_MODEL` unset in `.env`) is currently hitting a model-not-found
error from the Anthropic API. Also update `.env.example`'s `AGENT_MODEL=` comment/example
value if it references the old string.

### 0b. Add `AGENT_MAX_MUTATIONS` config flag
In `config.py`, alongside the other `AGENT_*` flags:

```python
AGENT_MAX_MUTATIONS = int(os.getenv("AGENT_MAX_MUTATIONS", 3))
```

Add to `.env.example`:

```
# Max number of apply_mutation calls the agent may make in one refinement session,
# independent of AGENT_MAX_ITERATIONS (which bounds total tool calls of any kind).
AGENT_MAX_MUTATIONS=3
```

Import it in `agent.py`:

```python
from config import (
    ROSETTA_ENABLED,
    GROMACS_ENABLED,
    OPENMM_ENABLED,
    BOLTZ_ENABLED,
    MD_PRODUCTION_NS,
    AGENT_API_KEY,
    AGENT_BASE_URL,
    AGENT_MODEL,
    AGENT_MAX_ITERATIONS,
    AGENT_MAX_MUTATIONS,  # add this
)
```

---

## Task 1 — Implement the `apply_mutation` agent tool

### Where
`orchestrator/agent.py` — add to `_AGENT_TOOLS` list and `_execute_agent_tool()`.

### Why this design
- `models/schemas.py` `Context.mutations` already uses the shape
  `{"pos": int, "from": str, "to": str}` for the *request* schema. The tool's
  `input_schema` below uses `position` / `from_aa` / `to_aa` instead — deliberately
  different names, since these are Claude's tool-call arguments, not the request body,
  and `from` is a Python keyword (can't use it as a dict key comfortably in code you'll
  read later). Flagging this naming difference so it's a conscious choice, not an
  inconsistency someone "fixes" later without realizing why.
- Re-prediction should **not commit the sequence mutation if re-prediction fails** —
  design below keeps `state["sequence"]` unchanged on failure so the agent's other tools
  keep operating on a consistent sequence/structure pair.
- Which backend to re-run: mirror the existing pattern in `orchestrator/tasks.py`
  (`if BOLTZ_ENABLED: call_boltz(...) else: call_esmfold_api(...)`), not always ESMFold —
  otherwise a Boltz-driven session would silently downgrade every mutation to ESMFold.

### Add this tool definition to `_AGENT_TOOLS`

```python
{
    "name": "apply_mutation",
    "description": (
        "Mutate the sequence at a given position and re-predict the structure. "
        "Use to test whether a point mutation resolves a low-confidence region or "
        "matches a requested mutation in context.mutations. Re-runs the active "
        "prediction backend (Boltz-2 if enabled, else ESMFold) on the mutated sequence. "
        "Limited to AGENT_MAX_MUTATIONS calls per session — use them deliberately."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "position": {
                "type": "integer",
                "description": "1-indexed residue position to mutate.",
            },
            "from_aa": {
                "type": "string",
                "description": (
                    "Expected current amino acid (1-letter code) at `position`, for "
                    "verification. Optional but recommended."
                ),
            },
            "to_aa": {
                "type": "string",
                "description": "Target amino acid (1-letter code) to mutate to.",
            },
        },
        "required": ["position", "to_aa"],
    },
},
```

### Add this branch to `_execute_agent_tool()`

Add near the top of the file (module-level constant, alongside the other module
constants — reuses the same 20-code alphabet already used in
`models/schemas.py::PredictionRequest.validate_sequence`, duplicated here rather than
imported to avoid a schemas->agent coupling; flag if you'd rather share one constant):

```python
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
```

Add the import at the top of `agent.py` (currently only `call_boltz` is imported):

```python
from orchestrator.backends.esmfold import call_esmfold_api
from config import BOLTZ_ENABLED  # already imported? check — if not, add
```

Then in `_execute_agent_tool()`:

```python
if tool_name == "apply_mutation":
    applied_so_far = len(state.get("mutations_applied", []))
    if applied_so_far >= AGENT_MAX_MUTATIONS:
        return json.dumps({
            "error": (
                f"mutation limit reached ({applied_so_far}/{AGENT_MAX_MUTATIONS} "
                "AGENT_MAX_MUTATIONS) — no further mutations this session"
            )
        })

    try:
        position = int(tool_input["position"])
        to_aa = str(tool_input["to_aa"]).upper()
    except (KeyError, ValueError, TypeError):
        return json.dumps({"error": "position and to_aa are required and must be valid"})

    from_aa = tool_input.get("from_aa")
    seq = state["sequence"]

    if position < 1 or position > len(seq):
        return json.dumps({
            "error": f"position {position} out of range (sequence length {len(seq)})"
        })
    if to_aa not in _VALID_AA:
        return json.dumps({"error": f"'{to_aa}' is not a standard amino acid code"})

    idx = position - 1
    actual_from = seq[idx]
    if from_aa and str(from_aa).upper() != actual_from:
        return json.dumps({
            "error": (
                f"from_aa mismatch: sequence has '{actual_from}' at position "
                f"{position}, not '{from_aa}'"
            )
        })

    mutated_seq = seq[:idx] + to_aa + seq[idx + 1:]

    try:
        if BOLTZ_ENABLED:
            pred = call_boltz(mutated_seq, context=state["context"], seed=0)
        else:
            pred = call_esmfold_api(mutated_seq, seed=0)
    except Exception as e:
        # Do NOT commit the mutation to state if re-prediction failed —
        # keep sequence/structure consistent for subsequent tool calls.
        return json.dumps({"error": f"re-prediction failed, mutation not applied: {e}"})

    # Commit only on success
    state["sequence"] = mutated_seq
    state["current_pdb"] = pred.structure_pdb
    state["plddt_scores"] = pred.plddt_scores
    state["mean_plddt"] = pred.mean_plddt
    state["num_clashes"] = count_clashes(pred.structure_pdb)
    state.setdefault("mutations_applied", []).append(f"{actual_from}{position}{to_aa}")

    result = {
        "status": "completed",
        "mutation": f"{actual_from}{position}{to_aa}",
        "model_name": pred.model_name,
        "mean_plddt": round(pred.mean_plddt, 2),
        "num_clashes": state["num_clashes"],
    }
    if pred.affinity_score is not None:
        result["affinity_kcal_mol"] = round(pred.affinity_score, 3)
    return json.dumps(result)
```

Note this branch already recomputes `num_clashes` correctly — see Task 2, which fixes
the same bug in the *existing* `run_rosetta_relax` / `run_simulation` branches.

### Schema addition — `models/schemas.py`

Add to `PostProcessingResult` (alongside `agent_reasoning`):

```python
class PostProcessingResult(BaseModel):
    """Post-processing and scoring result."""
    num_clashes: int
    rosetta_energy: Optional[float] = None
    gromacs_potential_energy: Optional[float] = None
    simulation_metrics: Optional[Dict[str, Any]] = None
    agent_reasoning: Optional[str] = None
    mutations_applied: Optional[List[str]] = None   # e.g. ["A12V", "G45S"] — new
    validation_reason: Optional[str] = None
    score: float
    decision: str
```

### Update `run_agent_refinement()`
- Initialize `state["mutations_applied"] = []` alongside the other state keys (this list
  also doubles as the counter the cap check in `apply_mutation` reads via `len(...)`).
- When building the final `PostProcessingResult`, add:
  ```python
  post_proc.mutations_applied = state["mutations_applied"] or None
  ```

### Update `_AGENT_SYSTEM` prompt
Add a line under "Available prediction backends" section, e.g.:

```
If context.mutations lists specific mutations, use apply_mutation to test them and
compare mean_pLDDT before/after. Only mutate when context requests it or when analysis
suggests a mutation would resolve a specific low-confidence region — don't mutate
speculatively without a stated reason in your final reasoning.
```

### Tests — `tests/test_agent.py` (new file; none exists yet — confirmed by `ls tests/`)
Follow the mocking style of `tests/test_boltz.py` (mock the backend calls, don't
require a live API key or GPU). Minimum cases:
- valid mutation, backend mocked to return a `StructurePrediction` → state updated,
  `mutations_applied` logged
- `from_aa` mismatch → error, state unchanged
- `to_aa` invalid code → error, state unchanged
- `position` out of range (0, negative, > len(seq)) → error
- backend raises → error returned, `state["sequence"]` and `state["current_pdb"]`
  unchanged (this is the rollback behavior — test it explicitly, it's easy to regress)
- `BOLTZ_ENABLED=True` routes to `call_boltz`, `False` routes to `call_esmfold_api`
  (patch both, assert only the expected one was called)
- cap enforcement: pre-populate `state["mutations_applied"]` with `AGENT_MAX_MUTATIONS`
  entries, assert the next `apply_mutation` call returns the limit error and does not
  call the backend at all (patch `call_esmfold_api`/`call_boltz` and assert not called)

---

## Task 2 — Fix stale `num_clashes` after refinement

### The bug
In `run_agent_refinement()`, `num_clashes` is computed once before the agent loop
starts and never updated. But `run_rosetta_relax` and `run_simulation` (in
`_execute_agent_tool`) both mutate `state["current_pdb"]` without recomputing clashes.
Consequences:
1. The final `score = state["mean_plddt"] - (state["num_clashes"] * 5.0)` can be wrong
   after refinement — using pre-relax clash count on a post-relax structure.
2. The agent itself never sees updated clash counts unless it happens to call
   `analyze_structure` again (which only reports pLDDT regions, not clashes) — so it can
   accept a structure with new clashes introduced during relax/simulation, unaware.

### Fix
In `_execute_agent_tool()`, after `state["current_pdb"]` is reassigned in both the
`run_rosetta_relax` and `run_simulation` branches, add:

```python
state["num_clashes"] = count_clashes(state["current_pdb"])
```

...and include it in both branches' returned JSON (e.g. add `"num_clashes":
state["num_clashes"]` to the result dict) so the agent's own transcript reflects it —
otherwise the fix is invisible to the model reasoning about whether to accept.

### Tests
Extend `tests/test_orchestrator.py::TestRunPredictionCoreProgress` or add a small new
test in the new `tests/test_agent.py` from Task 1: mock `run_rosetta_relax` to return a
PDB string with more clashes than the original, call `_execute_agent_tool("run_rosetta_relax", ...)`,
assert `state["num_clashes"]` reflects the new structure, not the pre-relax one.

### Note on benchmark reproducibility
Since this changes `score` computation for any run that goes through Rosetta relax or
simulation with the Claude agent enabled (`AGENT_ENABLED=True`), any existing benchmark
runs in `benchmarks/results.jsonl` that used the agent path (check: none currently do —
all logged runs are `boltz-2` direct backend calls, not agent-mediated) are unaffected.
Confirm this stays true before merging if you add agent-mediated runs to the benchmark
suite later.

---

## Task 3 — Tests for `orchestrator/ligands.py` and `orchestrator/membrane.py`

Zero coverage today (confirmed — grep for `def test_` across `tests/` returns nothing
for either module). These functions all shell out to external binaries (GNINA, Vina,
ACPYPE, insane.py) or optional heavy imports (RDKit, OpenFF, OpenMM) that won't be
installed in most CI environments — so these need to be **fully mocked unit tests**,
matching the style already established in `tests/test_boltz.py` (mock subprocess /
filesystem, no real binaries invoked). Do not attempt real GNINA/Vina/ACPYPE calls in
CI.

### `tests/test_ligands.py` (new)
- `smiles_to_3d`: valid SMILES → SDF written (mock `rdkit.Chem`/`AllChem`, or if RDKit
  is actually available in the dev env, use a trivial real molecule like ethanol
  `"CCO"` for a lightweight real test); invalid SMILES → `ValueError`; RDKit not
  installed → `RuntimeError` with install instructions in the message (mock the import
  to raise `ImportError`)
- `_ca_centroid` / `_all_ca_coords`: pure-Python PDB parsing, no mocking needed — feed a
  small synthetic PDB string, assert centroid math
- `dock_gnina`: mock `shutil.which` → None → `RuntimeError`; mock `shutil.which` → path
  and `subprocess.run` → mock success/failure, assert command construction differs for
  binding-site-provided vs. blind docking (check `--center_x` etc. appear only in the
  binding-site branch)
- `dock_vina`: mock `vina.Vina` and RDKit; test the meeko-available and
  meeko-unavailable fallback paths separately
- `parameterize_ligand_acpype` / `parameterize_ligand_openff`: mock `shutil.which` /
  `subprocess.run` / the openff imports; test both the not-installed error path and the
  success path (assert returned dict keys)
- `prepare_ligands`: integration-style test of the orchestration logic only, with
  every step-function mocked — test the GNINA→Vina→undocked fallback chain explicitly
  (this fallback chain is the most complex logic in the file and has no coverage at all)

### `tests/test_membrane.py` (new)
- `_resolve_insane`: mock `os.path.isfile` / `shutil.which`, test the 3-tier lookup order
- `_lipid_name`: pure function, test known lipids, unknown lipid passthrough, `None` →
  default POPC
- `embed_in_membrane_gromacs`: mock `_resolve_insane` → None → `RuntimeError`; mock
  found + `subprocess.run` success/failure; test that `-center` flag is only added when
  `span` is provided in `membrane_context`
- `embed_in_membrane_openmm`: mock `openmm`/`openmmforcefields` imports (both the
  not-installed and installed paths); mock `modeller.addMembrane` raising → wrapped in
  `RuntimeError` with the lipid name in the message

---

## Two small remaining calls (not blocking, pick whenever)

1. `_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")` is duplicated between
   `models/schemas.py::PredictionRequest.validate_sequence` and the new constant in
   `agent.py`. Fine to leave duplicated (avoids a schemas→agent import), but if you'd
   rather have one source of truth, it should probably live in a small shared
   `constants.py` rather than either module importing the other. Low priority — flag,
   don't block on it.
2. The exact wording I drafted for the `_AGENT_SYSTEM` prompt addition (Task 1) is a
   first pass — once Claude Code wires the tool in, worth actually running a session
   with `context.mutations` populated and reading the transcript to see if the model's
   mutation behavior matches what you want (deliberate, reasoned, not speculative) before
   locking the prompt wording in.

## After Task 0–3 land

This clears the "harden what exists" bucket. Natural next conversation: which of the
research directions from before (PoseBusters pose validation, PDBbind/CASF-2016 binding
affinity benchmarking, or a proper full-CASP15 MSA-ablation write-up) to pick up next,
now that `apply_mutation` gives the agent a real lever to pull during refinement sessions
worth benchmarking.
