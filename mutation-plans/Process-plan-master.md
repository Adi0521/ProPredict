# Master Plan — ProPredict Hardening + Mutation Scoring (resequenced)

**This supersedes the task ordering in `Process-plan-hardening.md` and
`Process-plan-mutation-scan.md`.** Those two docs still have the correct *content* for
their respective tasks — this file only changes the **order**: the algorithmic
mutation scorer (`mutation_scan.py`, ProteinMPNN-based) must land and pass a validation
check *before* any work touches the agent's `apply_mutation` tool. Rationale: once the
agent tool exists, it's tempting to start using/testing it with blind agent-picked
mutations; better to have the real scorer in place first so `apply_mutation`'s design
and system-prompt wording can be written with the scorer's existence in mind from the
start, even though (per earlier decision) `apply_mutation` itself will not call the
scorer internally yet — that wiring is still a distinct, later decision.

**Repo state this plan is based on:** re-verified fresh via `git pull` immediately
before writing this — `main` is unchanged since the prior two docs were written
(commit `a1fa688`, "added comprehensive testing suite for progress reporting"). No new
drift to account for.

## Resequenced order

| # | Task | Depends on | Source doc |
|---|------|-----------|------------|
| 0 | Config fixes (`AGENT_MODEL`, `AGENT_MAX_MUTATIONS`, `PROTEINMPNN_PATH`) | none | hardening + mutation-scan, merged below |
| 1 | `orchestrator/mutation_scan.py` — ProteinMPNN scorer, standalone | Task 0 | mutation-scan |
| 2 | **Validation gate** — sanity-check scorer against ProteinGym before proceeding | Task 1 | mutation-scan |
| 3 | `apply_mutation` agent tool | Task 0 (not Task 1/2 — still standalone per earlier decision, see note in Task 3) | hardening |
| 4 | Fix stale `num_clashes` after refinement | Task 3 (same file) | hardening |
| 5 | Tests for `orchestrator/ligands.py` / `orchestrator/membrane.py` | none — independent, can run in parallel with anything above | hardening |

Per `CLAUDE.md`'s own workflow rules, still do these **one at a time** with sign-off
between each, in the order above.

---

## Task 0 — Config fixes

### 0a. Fix `AGENT_MODEL` default
```python
# config.py — before
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-4-6")
# after
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5")
```
`"claude-opus-4-6"` is not a valid current model string — any agent-loop call relying
on the default is currently hitting a model-not-found error. Also fix `.env.example`'s
`AGENT_MODEL=` example value if it references the old string.

### 0b. Add `AGENT_MAX_MUTATIONS`
```python
AGENT_MAX_MUTATIONS = int(os.getenv("AGENT_MAX_MUTATIONS", 3))
```
`.env.example`:
```
# Max apply_mutation calls per agent session, independent of AGENT_MAX_ITERATIONS.
AGENT_MAX_MUTATIONS=3
```

### 0c. Add `PROTEINMPNN_PATH` / `PROTEINMPNN_MODEL_NAME`
```python
PROTEINMPNN_PATH = os.getenv("PROTEINMPNN_PATH", "")
PROTEINMPNN_MODEL_NAME = os.getenv("PROTEINMPNN_MODEL_NAME", "v_48_020")
```
`.env.example`:
```
# Path to a local clone of https://github.com/dauparas/ProteinMPNN (MIT license).
# Model weights are included in the clone itself (~26MB) — no separate download.
#   git clone https://github.com/dauparas/ProteinMPNN.git
PROTEINMPNN_PATH=
PROTEINMPNN_MODEL_NAME=v_48_020
```

---

## Task 1 — `orchestrator/mutation_scan.py` (standalone; build and validate before Task 3)

**Verification note:** this design isn't just read from ProteinMPNN's docs — I cloned
`https://github.com/dauparas/ProteinMPNN` (MIT), installed CPU-only PyTorch, and ran
`--conditional_probs_only` end-to-end on the repo's own sample PDB (`6MRR.pdb`, 68
residues) in a sandbox. Every shape, `.npz` key, and the alphabet string below came
from that live run plus reading `protein_mpnn_run.py` source directly — not inferred.
Live result: `log_p` shape `(1, 68, 21)`; scoring position 5 (wild-type `E`, in a
helix) ranked the wild-type residue highest of all 20 options, with hydrophobic/
similar substitutions (`L`, `I`, `Y`, `V`) next-best and `W`/`P`/`C` worst — chemically
sensible, not garbage output.

### Why ProteinMPNN over ESM-IF1
- Benchmarked more performant than ESM-IF1 for sequence design (FoldingDiff paper's
  direct comparison); still the most widely used inverse folding model per current
  literature.
- Pure PyTorch + NumPy — no `torch_geometric` (which ESM-IF1 needs and which is
  notoriously version-fragile against PyTorch/CUDA combos).
- Weights are tiny and ship in the git clone: 26MB total for all four noise-level
  checkpoints, confirmed by `du -sh` on the actual clone (not the ~2GB some blog posts
  claim).
- HERMES (Visani et al.) already validates exactly this use case: scoring mutational
  effects from ProteinMPNN's per-site probabilities via log-likelihood-ratio.

### Verified facts used below
- Alphabet (`protein_mpnn_utils.py:50`): `"ACDEFGHIKLMNPQRSTVWYX"` (21 symbols, `X` last).
- `protein_mpnn_run.py --pdb_path <single.pdb>` works directly on one PDB — no need for
  the `helper_scripts/parse_multiple_chains.py` jsonl step (that's only for folders of
  many PDBs). Confirmed by reading the `args.pdb_path` branch (~line 164).
- `--conditional_probs_only 1` writes `<out_folder>/conditional_probs_only/<name>.npz`
  with keys `log_p` (shape `[1, L, 21]`), `S`, `mask`, `design_mask`. Confirmed by
  source and live run.
- This mode gives `log P(s_i | backbone, rest of sequence)` — accounts for the real
  sequence context at every other position, stronger than `--unconditional_probs_only`
  (backbone alone) for scoring a mutation on a sequence you already have.

### Module

```python
"""
Structure-aware mutation scoring via ProteinMPNN (Dauparas et al., 2022, Science;
MIT license, https://github.com/dauparas/ProteinMPNN).

Scores candidate single-point substitutions using the log-likelihood-ratio method
validated in HERMES (Visani et al.) and related zero-shot mutation-effect literature:

    score(pos, wt -> mut) = log P(mut | backbone, rest of sequence)
                           - log P(wt  | backbone, rest of sequence)

Positive score = ProteinMPNN considers the substitution more structurally compatible
than the wild-type at this position. This is a STRUCTURAL COMPATIBILITY score, not a
direct proxy for function, stability, or fitness — say so in any output/UI that
surfaces these numbers.
"""
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Verified against orchestrator's cloned ProteinMPNN source, protein_mpnn_utils.py:50
_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_STANDARD_AA = _ALPHABET[:20]  # exclude X (unknown) from candidate substitutions


def _run_proteinmpnn_conditional_probs(
    pdb_string: str,
    tmpdir: str,
    proteinmpnn_dir: str,
    model_name: str = "v_48_020",
) -> np.ndarray:
    """
    Run ProteinMPNN --conditional_probs_only on a single-chain PDB and return the
    [L, 21] log-probability matrix (first/only batch element already sliced out).
    """
    pdb_path = os.path.join(tmpdir, "structure.pdb")
    with open(pdb_path, "w") as fh:
        fh.write(pdb_string)

    out_dir = os.path.join(tmpdir, "mpnn_out")
    os.makedirs(out_dir, exist_ok=True)

    run_script = os.path.join(proteinmpnn_dir, "protein_mpnn_run.py")
    weights_dir = os.path.join(proteinmpnn_dir, "vanilla_model_weights")

    if not os.path.isfile(run_script):
        raise RuntimeError(
            f"protein_mpnn_run.py not found at {run_script}. Check PROTEINMPNN_PATH "
            "points at a clone of https://github.com/dauparas/ProteinMPNN."
        )

    cmd = [
        sys.executable, run_script,
        "--pdb_path", pdb_path,
        "--out_folder", out_dir,
        "--path_to_model_weights", weights_dir,
        "--model_name", model_name,
        "--conditional_probs_only", "1",
        "--num_seq_per_target", "1",
        "--seed", "0",
        "--batch_size", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=proteinmpnn_dir)
    if result.returncode != 0:
        raise RuntimeError(
            f"ProteinMPNN failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-800:]}\nstderr: {result.stderr[-800:]}"
        )

    npz_dir = os.path.join(out_dir, "conditional_probs_only")
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")] if os.path.isdir(npz_dir) else []
    if not npz_files:
        raise RuntimeError(f"ProteinMPNN completed but no .npz output found in {npz_dir}")

    data = np.load(os.path.join(npz_dir, npz_files[0]))
    return data["log_p"][0]  # [L, 21] — first (only) batch element


def score_candidate_mutations(
    pdb_string: str,
    sequence: str,
    positions: Optional[List[int]] = None,
    top_k: int = 10,
    proteinmpnn_dir: str = "",
    model_name: str = "v_48_020",
) -> List[Dict[str, Any]]:
    """
    Score candidate single-point substitutions using ProteinMPNN structural log-odds.

    Parameters
    ----------
    pdb_string : the current predicted structure (PDB format)
    sequence : the current sequence (must match the structure's residue count/order)
    positions : 1-indexed residue positions to consider; None = scan every position
    top_k : return at most this many candidates, sorted by score descending
    proteinmpnn_dir : path to a clone of https://github.com/dauparas/ProteinMPNN
    model_name : one of v_48_002 / v_48_010 / v_48_020 / v_48_030
                 (lower noise = higher native-sequence recovery; higher noise =
                 designs that fold more reliably when re-predicted — v_48_020 is
                 ProteinMPNN's own default)

    Returns
    -------
    List of dicts, sorted by score descending:
        {"position": int, "from_aa": str, "to_aa": str, "score": float}
    """
    if not proteinmpnn_dir:
        raise RuntimeError(
            "proteinmpnn_dir not provided. Set PROTEINMPNN_PATH in .env to a clone of "
            "https://github.com/dauparas/ProteinMPNN (weights are included in the clone)."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        log_p = _run_proteinmpnn_conditional_probs(pdb_string, tmpdir, proteinmpnn_dir, model_name)

    candidates: List[Dict[str, Any]] = []
    scan_positions = positions if positions is not None else range(1, len(sequence) + 1)
    for pos in scan_positions:
        idx = pos - 1
        if idx < 0 or idx >= len(sequence) or idx >= log_p.shape[0]:
            continue
        wt_aa = sequence[idx]
        if wt_aa not in _STANDARD_AA:
            continue
        wt_idx = _ALPHABET.index(wt_aa)
        for mut_idx, mut_aa in enumerate(_STANDARD_AA):
            if mut_aa == wt_aa:
                continue
            score = float(log_p[idx, mut_idx] - log_p[idx, wt_idx])
            candidates.append({
                "position": pos, "from_aa": wt_aa, "to_aa": mut_aa,
                "score": round(score, 4),
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, help="Path to a PDB file")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--proteinmpnn-dir", default=os.getenv("PROTEINMPNN_PATH", ""))
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    with open(args.pdb) as fh:
        pdb_str = fh.read()
    results = score_candidate_mutations(
        pdb_str, args.sequence, top_k=args.top_k, proteinmpnn_dir=args.proteinmpnn_dir
    )
    for r in results:
        print(f"{r['from_aa']}{r['position']}{r['to_aa']}: {r['score']:+.4f}")
```

Run manually via: `python -m orchestrator.mutation_scan --pdb myprotein.pdb --sequence MKT...`

### Setup note
Git-clone tool, not pip-installable — same pattern as `insane.py` in `membrane.py`.
Add to `README.md`/`CLAUDE.md` setup: `git clone
https://github.com/dauparas/ProteinMPNN.git`, point `PROTEINMPNN_PATH` at it.

### Tests — `tests/test_mutation_scan.py` (new; fully mocked, matching `test_boltz.py` style)
- log-odds formula correctness against a synthetic `log_p` array with hand-checked
  numbers; sort order; `top_k` truncation
- `positions` filtering: subset passed, only those appear
- out-of-range positions silently skipped (not an error) — add a `logger.warning` for
  this case if not already planned, since a mismatch here likely means a stale
  sequence/structure pair upstream
- missing `proteinmpnn_dir` → `RuntimeError` before any subprocess call
- missing `protein_mpnn_run.py` at given path → `RuntimeError`
- subprocess non-zero exit → `RuntimeError` with stdout/stderr in message
- missing `.npz` output → `RuntimeError`
- one real integration test, skipped by default via env-var gate (mirrors
  `test_boltz.py`'s GPU-gated integration tests), that actually shells out to a real
  ProteinMPNN clone if `PROTEINMPNN_PATH` is set — this is the test that would catch
  drift if a future ProteinMPNN version changes the CLI/output format

---

## Task 2 — Validation gate (must pass before Task 3 starts)

Before this scorer is trusted for anything downstream, sanity-check it against real
data: **ProteinGym** (Notin et al., 2023/2024) — the standard benchmark for zero-shot
mutation-effect prediction, 2.5M+ real deep-mutational-scanning measurements across
217 assays.

### Data: do NOT clone the ProteinGym GitHub repo

That repo is just code (their own evaluation/scoring scripts) — the actual DMS
mutation data isn't stored there. Verified live in a sandbox (downloaded and queried
it directly, not just read the docs):

**Reference metadata** — small, fetchable directly with no auth, no clone:
```bash
curl -sL https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv -o DMS_substitutions_reference.csv
```
217 rows, one per assay, with `DMS_id`, `seq_len`, `coarse_selection_type`
(Stability/Activity/Binding/Expression/OrganismalFitness), `pdb_file`, `MSA_Neff_L_category`,
etc. — this is how the three assays below were picked.

**Mutation/fitness data** — use the AWS Open Data mirror, not the 1GB
`DMS_ProteinGym_substitutions.zip` some docs point to. Verified: this is a single
89MB parquet with all 217 assays already combined, no account/credentials needed:
```bash
pip install awscli --break-system-packages   # if not already installed
aws s3 cp --no-sign-request s3://proteingym/DMS_substitutions.parquet .
```
Confirmed columns: `DMS_score`, `DMS_score_bin`, `mutated_sequence`, `target_seq`,
`mutant`, `DMS_id`. Filter to the three assays below in pandas:
```python
import pandas as pd
df = pd.read_parquet("DMS_substitutions.parquet")
for did in ["TCRG1_MOUSE_Tsuboyama_2023_1E0L", "ESTA_BACSU_Nutschel_2020", "CCDB_ECOLI_Adkar_2012"]:
    df[df["DMS_id"] == did].to_csv(f"{did}.csv", index=False)
```
Row counts confirmed by actually loading it: 1058 / 2172 / 1176 respectively, matching
the reference file exactly.

**Structures** — either grab ProteinGym's own AF2-predicted structures (smaller,
already matches what ProteinGym's own baselines used):
```bash
wget https://marks.hms.harvard.edu/proteingym/ProteinGym_AF2_structures.zip
```
or re-predict the three sequences with this pipeline's own `call_esmfold_api`
(`ESMFOLD_LOCAL=False` — free public API, no GPU/Modal needed, see the "local testing"
note two tasks up). Pick one and be consistent — mixing structure sources across the
three assays would confound the comparison.

### The three assays (deliberately not all best-case)

Structure-based scorers are expected to do best on stability assays specifically — so
testing only a stability assay would bias the check optimistically. Picked one
best-case, one more-realistic-size best-case, and one contrast category:

| DMS_id | Category | Length | Mutants | Why |
|---|---|---|---|---|
| `TCRG1_MOUSE_Tsuboyama_2023_1E0L` | Stability | 37 | 1,058 | Tiny — fast, best-case sanity check |
| `ESTA_BACSU_Nutschel_2020` | Stability | 212 | 2,172 | Realistic protein size, not part of the Tsuboyama mega-scale study (avoids over-indexing on one lab's assay design) |
| `CCDB_ECOLI_Adkar_2012` | Activity | 101 | 1,176 | Contrast category — tests whether the scorer only works for stability or generalizes at all |

### What to compute
For each assay: run `score_candidate_mutations` on the assay's structure/sequence,
join on `mutant` (format like `A14D` = position 14, A→D) to get a ProteinMPNN score
per row, compute Spearman correlation against `DMS_score`. Write up the three
correlations (plus a brief note on structure source used) in `Process/`, following the
existing dated write-up convention.

This doesn't need to beat state-of-the-art — ProteinMPNN alone typically lands
mid-pack on ProteinGym versus ensemble methods like SaProt/Tranception. It only needs
to confirm the scorer does something sane on *this pipeline's* actual structures
before Task 3 references it in the agent's system prompt at all.

**This is a hard gate, not a formality** — if the correlation is near zero or negative
on your own predicted structures (as opposed to crystal structures ProteinMPNN was
more likely validated on), that's worth knowing before Task 3 moves forward. If the
Activity-category assay (`CCDB_ECOLI`) correlates much worse than the two Stability
assays, that's an expected and worth-noting result, not a failure — it tells you the
scorer's blind spot, which is useful information for the eventual paper either way.

---

## Task 3 — `apply_mutation` agent tool

**Sequencing note:** built *after* Tasks 1-2 land, but still standalone per the
earlier decision — `apply_mutation` does **not** call `score_candidate_mutations`
internally yet, and no `scan_mutations` tool is added to `_AGENT_TOOLS` yet. The only
change from doing this after Task 1-2 rather than before: the system prompt below can
now truthfully mention that a validated structural-compatibility scorer exists in the
codebase, without yet exposing it as a callable tool — sets up the next task cleanly
without conflating "exists" with "agent can call it."

### Where
`orchestrator/agent.py` — add to `_AGENT_TOOLS` list and `_execute_agent_tool()`.

### Why this design
- `models/schemas.py` `Context.mutations` already uses the shape
  `{"pos": int, "from": str, "to": str}` for the *request* schema. The tool's
  `input_schema` below uses `position` / `from_aa` / `to_aa` instead — deliberately
  different names: these are Claude's tool-call arguments, not the request body, and
  `from` is a Python keyword.
- Re-prediction should **not commit the sequence mutation if re-prediction fails** —
  keeps `state["sequence"]` unchanged on failure so other tools keep operating on a
  consistent sequence/structure pair.
- Which backend to re-run: mirror the existing pattern in `orchestrator/tasks.py`
  (`if BOLTZ_ENABLED: call_boltz(...) else: call_esmfold_api(...)`), not always
  ESMFold — otherwise a Boltz-driven session would silently downgrade every mutation.

### Add to `_AGENT_TOOLS`

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

### Module constant + import
```python
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")  # duplicated from models/schemas.py's
                                          # PredictionRequest.validate_sequence — kept
                                          # separate to avoid a schemas->agent coupling
```
```python
from orchestrator.backends.esmfold import call_esmfold_api
from config import (
    ROSETTA_ENABLED, GROMACS_ENABLED, OPENMM_ENABLED, BOLTZ_ENABLED,
    MD_PRODUCTION_NS, AGENT_API_KEY, AGENT_BASE_URL, AGENT_MODEL,
    AGENT_MAX_ITERATIONS, AGENT_MAX_MUTATIONS,
)
```

### `_execute_agent_tool()` branch

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
        return json.dumps({"error": f"re-prediction failed, mutation not applied: {e}"})

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

### Schema addition — `models/schemas.py`
```python
class PostProcessingResult(BaseModel):
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

### `run_agent_refinement()`
- Initialize `state["mutations_applied"] = []` alongside other state keys (also
  doubles as the counter the cap check reads via `len(...)`).
- Before returning, set `post_proc.mutations_applied = state["mutations_applied"] or None`.

### `_AGENT_SYSTEM` prompt addition
```
If context.mutations lists specific mutations, use apply_mutation to test them and
compare mean_pLDDT before/after. Only mutate when context requests it or when analysis
suggests a mutation would resolve a specific low-confidence region — don't mutate
speculatively without a stated reason in your final reasoning.

Note: a structure-aware mutation scorer (orchestrator/mutation_scan.py, ProteinMPNN-
based) exists in the codebase for ranking candidate substitutions algorithmically, but
is not yet available to you as a tool — for now, choose mutation targets from
analyze_structure's output and context.mutations only.
```
That last paragraph is deliberate scaffolding for the next task (wiring
`scan_mutations` in as a real tool) — it's honest about current capability without
overpromising, and makes the eventual wiring a smaller, well-telegraphed change.

### Tests — `tests/test_agent.py` (new)
Mocking style matches `tests/test_boltz.py`. Minimum cases:
- valid mutation, backend mocked → state updated, `mutations_applied` logged
- `from_aa` mismatch → error, state unchanged
- `to_aa` invalid code → error, state unchanged
- `position` out of range (0, negative, > len(seq)) → error
- backend raises → error, `state["sequence"]`/`state["current_pdb"]` unchanged
  (rollback behavior — test explicitly, easy to regress)
- `BOLTZ_ENABLED=True` routes to `call_boltz`, `False` to `call_esmfold_api`
- cap enforcement: pre-populate `mutations_applied` to the limit, assert the next call
  errors and does not call either backend

---

## Task 4 — Fix stale `num_clashes` after refinement

### The bug
`num_clashes` is computed once before the agent loop starts and never updated, but
`run_rosetta_relax`/`run_simulation` both mutate `state["current_pdb"]` without
recomputing it. Final `score` can be wrong post-refinement, and the agent never sees
updated clash counts unless it re-calls `analyze_structure` (which only reports pLDDT,
not clashes) — it can accept a structure with new clashes, unaware.

### Fix
After `state["current_pdb"]` is reassigned in both the `run_rosetta_relax` and
`run_simulation` branches of `_execute_agent_tool()`:
```python
state["num_clashes"] = count_clashes(state["current_pdb"])
```
Include it in both branches' returned JSON too, so the agent's own transcript
reflects it.

### Tests
In `tests/test_agent.py`: mock `run_rosetta_relax` to return a PDB with more clashes
than the original, call the tool, assert `state["num_clashes"]` reflects the new
structure.

### Benchmark reproducibility note
Only affects runs with `AGENT_ENABLED=True` that go through Rosetta relax or
simulation. Current `benchmarks/results.jsonl` entries are all direct `boltz-2`
backend calls, not agent-mediated — unaffected. Confirm this stays true if
agent-mediated runs are added to the benchmark suite later.

---

## Task 5 — Tests for `orchestrator/ligands.py` and `orchestrator/membrane.py`

Independent of everything above — can be done in parallel any time. Zero coverage
today. Fully mocked (subprocess/filesystem), no real GNINA/Vina/ACPYPE/insane.py
binaries in CI, matching `test_boltz.py` style.

### `tests/test_ligands.py`
- `smiles_to_3d`: valid SMILES → SDF (mock RDKit, or use a trivial real molecule like
  `"CCO"` if RDKit is available); invalid SMILES → `ValueError`; RDKit missing →
  `RuntimeError`
- `_ca_centroid`/`_all_ca_coords`: pure-Python PDB parsing, synthetic PDB string, no
  mocking needed
- `dock_gnina`: `shutil.which` → None → `RuntimeError`; found + `subprocess.run`
  success/failure; binding-site vs. blind-docking command construction differs
- `dock_vina`: mock `vina.Vina` + RDKit; meeko-available and meeko-unavailable paths
- `parameterize_ligand_acpype`/`parameterize_ligand_openff`: mock `shutil.which`/
  `subprocess.run`/openff imports; not-installed and success paths
- `prepare_ligands`: mock every step function, test the GNINA→Vina→undocked fallback
  chain explicitly — most complex logic in the file, zero coverage today

### `tests/test_membrane.py`
- `_resolve_insane`: mock `os.path.isfile`/`shutil.which`, 3-tier lookup order
- `_lipid_name`: known lipids, unknown passthrough, `None` → POPC default
- `embed_in_membrane_gromacs`: not-found → `RuntimeError`; success/failure; `-center`
  flag only added when `span` provided
- `embed_in_membrane_openmm`: mock `openmm`/`openmmforcefields` imports (both paths);
  `modeller.addMembrane` raising → wrapped `RuntimeError` with lipid name in message

---

## Still-open questions (carried over, unanswered)

1. ~~**ProteinMPNN `model_name` default**~~ **RESOLVED (2026-07-15): keep `v_48_020`.**
   Benchmarked all four checkpoints against the three ProteinGym assays
   (`benchmarks/benchmark_proteinmpnn_checkpoints.py`; see
   `Process/mutation-task-7-checkpoint-benchmark.md`). The choice is within noise — all
   four means fall in a 0.023-wide band and no checkpoint wins more than one assay. The
   "higher-noise-helps-because-we-re-fold" hypothesis held on only one of three assays
   and reversed on the other two. `v_48_020` has the best mean ρ (+0.5059) and is the
   middle-noise generalist, so the existing default stands. No code change.
2. **Where does `PROTEINMPNN_PATH` live?** — baked into `Dockerfile.celery` (clone
   during image build, ~26MB, trivial) vs. host-mounted path for local dev? Decides
   whether Task 1 needs a Dockerfile change too.
3. Combinatorial/multi-site search (AdaLead vs. Bayesian optimization vs. others) is
   still Phase 2 territory — not scheduled in this plan at all, deliberately, since it
   depends on Task 2's validation result and your answers to the oracle/budget
   questions raised in the original mutation-scan doc.