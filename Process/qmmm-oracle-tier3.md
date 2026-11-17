# QM/MM oracle — the tier-3 (physics) rung of the mutation-search funnel

**Date:** 2026-10-02
**Commits:** `74d0f09` (module), `34a4296` (tests)
**Deliverable:** a tier-3 oracle that conforms to the AdaLead oracle contract, with the
interface, gating, budget rule and WT-caching **final and test-locked**, and the two heavy QM
internals left as an explicit scaffold.
**Reproduce:** `pytest tests/test_qmmm_oracle.py` → **12 passed in 0.57s** (no GPU, no Boltz,
no xtb/ORCA).

## Why this exists now

`orchestrator/mutation_search.py` defines a cheap→expensive oracle funnel:

| tier | oracle | cost | epistasis |
|---|---|---|---|
| 1 | `additive_oracle` | ~free | none by construction |
| 2 | `score_only_oracle` | 1 subprocess | yes (ProteinMPNN NLL) |
| 3 | **this module** | minutes–hours | real geometry + energetics |

The module was written with its own activation condition in the docstring: wire it in **only
if the affinity-invariance experiment returns `H_invariant`** — i.e. only if Boltz-2's affinity
head is blind to mutations, so no cheap oracle can rank binding. If invariance had come back
`H_responsive`, the correct action was to leave this parked.

**That gate has now fired.** Row A returned `H_invariant` (pooled slope +0.013, predicted
spread 26% of experimental — see `Process/rowA-affinity-invariance-result.md`). The physics
tier now has a *measured* gap to fill rather than an assumed one, which is the whole reason
the experiment was run before the tier was built.

## What is final (and locked by tests)

The seam is deliberately complete even though the physics is not, so tier-3 can be filled in
without redesigning anything around it.

- **Oracle contract.** `make_qmmm_oracle(...)` returns `Callable[[List[str]], List[float]]`,
  higher = better — the identical signature AdaLead already consumes. Drop-in for the top-M
  validation pass.
- **Budget cap enforced in code, not in docs.** The returned oracle *raises* if handed more
  sequences than `max_candidates`. This is the load-bearing design decision: tier-3 at
  minutes-per-candidate must never run once-per-round on a full AdaLead batch, and a comment
  saying so would eventually be ignored. The guard makes the misuse impossible rather than
  discouraged.
- **WT folded exactly once, at build time, and cached.** Every candidate's
  `ΔΔ_int = E_int(mut) − E_int(WT)` is therefore protocol-matched against one reference, and
  the WT cost is paid once. `oracle.wt_fitness` exposes the reference for reporting.
- **WT/mutant protocol symmetry.** Same engine, same region rule, same capping, same
  `protein_copies`, same `seed` on every fold — asymmetry anywhere manufactures a fake Δ. A
  test asserts every fold call (WT included) carries identical `(smiles, copies, seed)`.
- **Fitness = −E_int**, so stronger predicted binding sorts higher.
- **Injected `fold_fn`.** Default wires to `orchestrator.backends.boltz.call_boltz` (lazily
  imported, so the module stays importable without a GPU/Boltz install) and passes
  `protein_copies` through — the homodimer support that Row A's Bug-2 fix added. Injection is
  what makes the whole interface testable with a recording fake.
- **Substitution-only guard.** Unequal-length sequences raise; candidates are equal-length
  substitutions of WT, matching tiers 1–2.

## What is deliberately not implemented

`_carve_qm_region` and `_engine_energy` raise `NotImplementedError` carrying the exact
decisions they require — QM set definition (ligand + residues within `QMMM_QM_RADIUS` of the
ligand or a mutated-site Cα, whole residues only), cutting only non-polar C–C bonds with
hydrogen link atoms, recording charge/spin via the existing PropKa path, and using the same
engine/method across all three subsystems and across WT/mutant.

**Two tripwire tests assert these still raise.** Filling them in therefore forces an
intentional test update instead of silently changing behavior — the scaffold cannot rot into
accidentally-live code.

The test suite mocks at the expensive boundary (`_interaction_energy`) with a deterministic
fake, so all 12 interface tests run in 0.57s with no heavy dependency present.

**Honest status:** none of the xtb/ORCA specifics in the module are live-verified. They are
design intent to confirm against the tools, not validated behavior.

## Follow-ups found while documenting

1. **Config bypasses `config.py` — violates a stated convention.** Six `QMMM_*` vars are read
   by module-level `os.getenv` in `qmmm_oracle.py` and appear in **neither `config.py` nor
   `.env.example`** (verified by grep). CLAUDE.md is explicit that `config.py` is the single
   source of truth and that new config must go through it plus `.env.example`. Worth fixing
   before the tier goes live, while there are no call sites to migrate.
   *Subtlety if you do:* `engine`, `radius` and `max_candidates` are **default arguments**,
   bound at import time, so patching the module attribute will not change them. Only
   `QMMM_ENABLED` is read inside the function body (which is why the tests can patch it).
2. **Dead imports.** `apply_mutations` and `mutations_from_sequences` are imported from
   `mutation_search` in a `try/except` and never used — mutated positions are computed inline
   with a list comprehension. The stated intent ("reuse the funnel's own sequence-threading so
   a candidate here is defined exactly as it is for tiers 1–2") is therefore not actually
   honored. Either use them or drop the import.
3. **`protein_copies` defaults to 1, which is wrong for the lead system.** HIV-1 protease is an
   obligate homodimer whose active site exists only at `copies=2`; a monomer has no inhibitor
   pocket. Nothing stops a caller from forgetting the argument, and the failure is silent —
   it returns plausible-looking energies for a structure with no binding site. Consider making
   it required, or validating it against the ligand contact count.
4. **The static-energy objection is not yet answered.** Row A's own reference list flags
   Taguchi et al. (2022), who did QM/MM on HIV-PR at **V82T/I84V** — the headline residues —
   and recovered the experimental 2.5–3.0 kcal/mol shift *only by combining QM/MM with
   conformational sampling via long MD*, because static QM/MM energies are not free energies.
   This oracle computes a static single-point `E_int`. That is the cheapest thing that could
   work, but the closest prior art says it is the wrong axis. This should be confronted before
   the internals are built, not after — it is the main risk to the tier being worth its cost.

## Artifacts

- `orchestrator/qmmm_oracle.py` — the oracle: gating, budget cap, WT cache, fold path (real);
  `_carve_qm_region` / `_engine_energy` (scaffold).
- `tests/test_qmmm_oracle.py` — 12 interface tests + 2 scaffold tripwires, fully mocked.
- `IDEA.md` — one-line project framing added alongside (`34a4296`); no writeup needed.
