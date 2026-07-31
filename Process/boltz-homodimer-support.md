# Homo-oligomer support in call_boltz (the homodimer fix)

**Date:** 2026-07-29
**Source:** `research_plan/rowA-boltz-affinity-invariance.md` Bug 2. The last of the two
blocking bugs for the HIV-PR affinity experiment (Bug 1, the affinity key, was
`Process/boltz-affinity-key-fix.md`).
**Status: done.** 181 unit tests pass (7 new).

## The bug

`call_boltz` always built a single protein chain (`id: "A"`). HIV-1 protease is an **obligate
homodimer** — the active site forms at the dimer interface, each monomer contributing one
catalytic aspartate (D25/D25'). A 99-residue monomer has no inhibitor pocket, so every HIV-PR
affinity number the backend could produce would be meaningless.

There was also a **latent chain-ID collision**: ligands were numbered `chr(ord("B") + i)`,
always starting at "B". Add a second protein chain (also "B") and the first ligand collides
with it. Invisible while only monomers were built; a correctness bug the moment multiplicity
exists.

## What was built

Scope (agreed up front): fix `call_boltz` only. Refolding the standalone HIV-PR benchmark onto
it is a separate follow-up.

### 1. Extracted the YAML builder — `_build_boltz_input()`

The input-assembly logic was inline in `call_boltz` **and independently reimplemented** in the
test file's `_build_boltz_yaml` helper. That duplication is the same failure mode that hid the
affinity-key bug for months: the test asserted against a *copy* of the logic, so it could stay
green while the real `call_boltz` diverged. Extracting one shared function
(`_build_boltz_input(sequence, context, protein_copies, use_msa)`) that both `call_boltz` and
the tests call was therefore a prerequisite, not cleanup — without it the feature is untestable
for real. The test helper now just forwards to it.

### 2. `protein_copies`

- **Monomer (`protein_copies=1`, default) emits scalar `id: "A"`** — byte-identical YAML to
  the pre-homomer version, so none of the reproducibility work shifts.
- **`≥2` emits `id: ["A", "B", …]`** — Boltz's homomer form, verified in the research plan
  against `boltz schema.py:1094-1097` (accepts `id` as str or list).
- **Ligand chains now start after the protein chains**: `chr(ord("A") + protein_copies + i)`.
  For `protein_copies=1` that is still "B" (backward compatible); for a dimer it is "C", fixing
  the collision. HIV-PR: proteins A+B, inhibitor C, `affinity: {binder: C}`.

### 3. Surface — schema field + kwarg

- `Context.protein_copies: int = Field(default=1, ge=1, le=26)` — validated, first-class API
  field, flows through `/predict` → pipeline. All four existing `call_boltz` call sites already
  pass `context=`, so nothing else needed changing for it to reach the backend.
- `call_boltz(..., protein_copies: Optional[int] = None)` kwarg for direct/benchmark calls.
  **Precedence: explicit kwarg > `context["protein_copies"]` > 1.** The kwarg default is `None`
  (not 1) so `call_boltz(..., protein_copies=1)` is distinguishable from "unset" and still
  overrides a context value.

### 4. Validation

- `protein_copies >= 1` (rejects 0 / negative) — in the builder, and again at the schema layer.
- `protein_copies + len(ligands) <= 26` — single-letter chain IDs. Lives in the builder, where
  the ligand count is known (the schema can't cross-validate the two). Clear error beyond.

## Tests (7 new, all mocked)

`tests/test_boltz.py`:
- **Deleted** the duplicate `_build_boltz_yaml` reimplementation; the existing YAML tests now
  exercise the real `_build_boltz_input`. `test_yaml_protein_only` additionally pins the scalar
  `"A"` monomer form.
- `test_yaml_homodimer_uses_id_list` — `id == ["A","B"]`
- `test_yaml_homodimer_with_ligand_shifts_ligand_chain` — the collision fix: ligand is "C", not
  "B"; binder "C"
- `test_yaml_trimer_chain_ids` — proteins A/B/C, ligand D
- `test_build_boltz_input_rejects_zero_copies`, `..._rejects_too_many_chains`
- `test_call_boltz_reads_protein_copies_from_context` — `context["protein_copies"]` reaches the
  written YAML with no kwarg (captures the actual YAML the subprocess would receive)
- `test_call_boltz_kwarg_overrides_context` — precedence: kwarg 1 beats context 3

## Out of scope (deliberate)

- **Refolding `benchmarks/benchmark_affinity_invariance.py`** onto the fixed `call_boltz` — that
  is the actual HIV-PR run, now unblocked. Separate task.
- **Hetero-oligomers** (different sequences per chain) — a different feature; `protein_copies`
  is homo-oligomer only.
- **>26 chains / multi-character chain IDs** — rejected with a clear error rather than handled.

## Verification

```
pytest tests/ --ignore=tests/test_api.py -k "not integration"   # 181 passed
```

Not exercised on GPU here (mocked, like the rest of the boltz unit tests). The natural
end-to-end check is the refold-and-run follow-up: predict HIV-PR (`protein_copies=2`) with a
known inhibitor and confirm a physically meaningful pocket/affinity, rather than the
pocket-less monomer number the backend used to produce.
