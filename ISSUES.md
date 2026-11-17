# Open Issues & Next Steps

Running list of defects and follow-ups. Complements the other two docs rather than
duplicating them:

- **`ROADMAP.md`** — stage-level feature work (what the project is building).
- **`Process/`** — write-ups of *completed* steps (what was done and why).
- **this file** — what is currently broken, stale, or next.

Each item says what it is, where, and how to fix it. Item IDs are stable and only ever
increment — reference them in commit messages. Append new items under the matching priority;
when one is done, check it off and drop the line (completed work belongs in `Process/`, not
here).

Started 2026-10-02 from the documentation pass over `d60def0..bbef345`.

---

## P0 — Data loss (committed; recover first)

Two documented results artifacts were overwritten by lower-effort re-runs at default flags in
`bbef345`. Both are recoverable from git history. **Do I-3 or these will recur.**

- [ ] **I-1 — Restore the 3-seed ProteinMPNN checkpoint benchmark.**
      `bbef345` replaced `benchmarks/proteinmpnn_checkpoint_results.json` (`seeds: [37,38,39]`,
      set by `999ef3e`) with a single-seed run; `spearman_std` collapsed `0.0003 → 0.0`. This
      re-introduces exactly what `Process/mutation-determinism-fix.md` was written to fix — a
      0.002 checkpoint margin sitting under ~0.008 seed noise, with no way to separate
      checkpoint effect from decoding-order noise.
      ```bash
      git show 999ef3e:benchmarks/proteinmpnn_checkpoint_results.json \
        > benchmarks/proteinmpnn_checkpoint_results.json
      ```
- [ ] **I-2 — Restore (or supersede) the 300-pair epistasis artifact.**
      `bbef345` replaced `benchmarks/score_only_epistasis.jsonl` (300 pairs, `588ca3a`) with a
      150-pair run, so the artifact no longer matches the run documented in
      `research_plan/rowB-score-only-epistasis.md`.
      ```bash
      git show 588ca3a:benchmarks/score_only_epistasis.jsonl \
        > benchmarks/score_only_epistasis.jsonl
      ```
      *Keep the 150-pair data somewhere* — it is a near-independent replicate (only 24/150
      pairs overlap the 300 draw) and is what established that the headline result replicates.
      See `Process/rowB-score-only-epistasis.md`.
- [ ] **I-3 — Stop harnesses silently overwriting documented results.**
      `benchmarks/benchmark_score_only_epistasis.py:201` writes mode `"w"` to a fixed default
      path, and `--n-doubles` defaults to 150 (`:156`), so a plain re-run destroys a
      documented 300-pair artifact with no warning.
      `benchmarks/benchmark_proteinmpnn_checkpoints.py` has the same shape.
      Fix: parameter-stamp the output filename (e.g. `..._n300_k8.jsonl`), **or** refuse to
      overwrite an existing file without `--force`. Prefer the stamp — it makes the
      provenance self-describing.

## P1 — Code defects (small, independent)

- [ ] **I-4 — `QMMM_*` config bypasses `config.py`.** Six vars read by module-level
      `os.getenv` at `orchestrator/qmmm_oracle.py:72-77`, present in **neither `config.py` nor
      `.env.example`**. CLAUDE.md requires `config.py` as single source of truth plus an
      `.env.example` entry. Cheapest now, while there are no call sites to migrate.
      *Trap when fixing:* `engine`, `radius` and `max_candidates` are **default arguments**
      (`:160-162`), bound at import, so patching the module attribute will not change them.
      Only `QMMM_ENABLED` is read inside the function body (`:179`), which is why the tests can
      patch it.
- [ ] **I-5 — Dead imports in `qmmm_oracle.py:65`.** `apply_mutations` and
      `mutations_from_sequences` are imported in a `try/except` and never used — mutated
      positions are computed inline at `:210`. The module docstring's claim to "reuse the
      funnel's own sequence-threading so a candidate here is defined exactly as it is for
      tiers 1-2" is therefore **not honored**. Either use them or drop the import and the claim.
- [ ] **I-6 — `protein_copies` defaults to 1 and fails silently.**
      `orchestrator/qmmm_oracle.py:163`. HIV-1 protease is an obligate homodimer whose active
      site exists only at `copies=2`; a monomer has no inhibitor pocket. A caller who forgets
      the argument gets plausible-looking energies for a structure that cannot bind. Make it
      required, or validate it against the ligand contact count.
- [ ] **I-7 — `analyze_affinity_invariance.py` docstring over-promises.** Line 17 states
      *"Censored rows are reported both included (rank-safe) and excluded"*; only the excluded
      branch exists (`:99`). Implement the included-censored column or correct the docstring.
      (Numbers are unaffected — the included-censored figures in
      `Process/rowA-affinity-invariance-result.md` were computed ad hoc, and the verdict is
      the same either way.)
- [ ] **I-8 — `check_determinism.py` docstring names the wrong structure.** The NOTE at `:28`
      says *"one chain of HIV-PR"*, but `DEFAULT_PDB` (`:41`) is the AMFR_HUMAN `4G3O` chain
      cached for Row B — verified single chain A, 53 residues. The harness works; the comment
      is wrong. Consider pointing `DEFAULT_PDB` at an actual HIV-PR chain instead, since that
      is the affinity-invariance system.
- [ ] **I-22 — `run_gromacs_em` fails on any net-charged protein, and is dead code.**
      `orchestrator/simulation.py:198`. It goes pdb2gmx → editconf → solvate → grompp with no
      `genion` step, so the system is never neutralized. On GROMACS 2023.3 (current worker image)
      and 2025.2 (previous image), `grompp` turns the net-charge Ewald warning into a fatal "Too many warnings (1)" — verified
      on `myprotein.pdb` (+4 charge). It has **no callers**: the only reference was an unused
      import in `orchestrator/tasks.py`, removed in the ruff pass (`Process/pyproject.md`).
      `run_gromacs_md` is the live path and does run `genion`. Fix: delete it, **or** add the
      `genion` step (copy from `run_gromacs_md`, `:~655`) if a standalone EM entry point is
      wanted. Do not paper over it with `-maxwarn` — a non-neutral Ewald system gives
      artifactual energies.

## P1 — Doc corrections (small)

- [ ] **I-9 — Correct Row B's contact-dependence claim.**
      `research_plan/rowB-score-only-epistasis.md` states distal pairs (CA > 8 Å) show larger
      median `|eps|` (0.029) than contacting (0.022). The 300-pair run reproduces that, but
      the near-independent 150-pair draw **inverts it** (contacting 0.0219 > distal 0.0188).
      The difference is small against an `eps` spread of ~0.029 and the distal bin is the
      smaller one, so it was never well-powered. Change to **"no resolvable contact
      dependence in either direction."** The sequence-context-not-contacts inference may still
      be right, but this data does not establish it.
- [ ] **I-10 — Add the two missing caveats to Row B's noise table.** It runs on
      `doubles[:60]` (`benchmark_score_only_epistasis.py:225`), so **n=60, not 300**; and the
      K=32 row required a second invocation at `--decoding-orders 32`. Neither is stated.

## P1 — Config change (backed by two independent measurements)

- [ ] **I-11 — Raise `PROTEINMPNN_NUM_DECODING_ORDERS` to ≥32.** Two unrelated measurements
      agree that N=8 leaves real noise: Row B's SNR table (1.7 → 3.0 going 8 → 32) and the
      determinism gate's cross-seed spread (mean per-residue std 1.15e-02 at N=8). Cost is
      linear in K; the cheap-search savings vs re-folding leave room. Decide whether this
      becomes the default in `.env.example` or only applies under `MUTATION_SEARCH_ENABLED`.

## P2 — Verification gaps (medium)

- [ ] **I-12 — Run the determinism gate on the Modal GPU image.**
      `python -m benchmarks.check_determinism` is green on M3/CPU, but the failure mode most
      worth fearing (torch GPU nondeterminism) is exactly what CPU cannot cover.
- [ ] **I-13 — Wire determinism gate 0 (and ideally gate 1) into CI.** Gate 0 — the `seed=0`
      rejection preflight — needs **no weights** and catches the highest-severity regression
      for free. The script already exits non-zero on failure.
- [ ] **I-14 — Run the Row A MSA-on arm.** Only MSA-off ran, so the mechanistic question
      (does invariance *relax* without an MSA, implicating MSA-washout as the cause?) is
      unanswered — the result is a phenomenon without a mechanism. Resumable, ~164 more jobs:
      `modal run benchmark_affinity_invariance_modal.py --both-msa-arms`
- [ ] **I-15 — Measure Row A within-model sampling noise.** Every job ran at a single seed
      (37) with `diffusion_samples=1`, so Boltz-2's own sampling spread is unmeasured. A
      handful of repeat seeds on a subset would bound it.

## P3 — Builds (large; sequenced)

- [ ] **I-16 — Settle the static-vs-free-energy objection. Blocks I-17.**
      Row A's own bibliography cites Taguchi et al. (2022), who did QM/MM on HIV-PR at
      **V82T/I84V** — the headline residues — and recovered the experimental 2.5–3.0 kcal/mol
      shift *only* by combining QM/MM with conformational sampling via long MD, because static
      QM/MM energies are not free energies. `qmmm_oracle.py` computes a **static single-point**
      `E_int`. This decides whether I-17 is even aimed at the right quantity, so it is cheaper
      to confront before the internals are built than after.
- [ ] **I-17 — Build the QM/MM tier-3 internals.** `_carve_qm_region`
      (`orchestrator/qmmm_oracle.py:110`) and `_engine_energy` (`:130`) are scaffolds raising
      `NotImplementedError` with the decisions they require. The interface, gating, budget cap,
      WT-caching and fold path are final and test-locked (12 tests, `tests/test_qmmm_oracle.py`),
      and two tripwire tests assert these two still raise — so filling them in forces an
      intentional test update. The `H_invariant` gate this was parked behind **has fired**
      (`Process/rowA-affinity-invariance-result.md`). None of the xtb/ORCA specifics in the
      module are live-verified — treat them as design intent to confirm against the tools.
      **Do I-16 first.**
- [ ] **I-18 — Row B sub-question B: does model epistasis track *experimental* epistasis?**
      Correlate `eps_model` against `eps_exp = DMS(m12) − DMS(m1) − DMS(m2) + DMS(WT)` from the
      same ProteinGym parquet. Stability arm (Tsuboyama ΔG doubles, this domain + 2 more) and
      one functional DMS arm, expecting weaker correlation there. Readout: Spearman/Pearson and
      slope. Run at ≥32 decoding orders (I-11). CPU-only, no GPU. A near-zero slope is still a
      publishable negative and a reason to lean on the tier-3 re-fold funnel.
- [ ] **I-19 — Confront Boltz-2's near-FEP affinity claim against the Row A null.**
      Passaro et al. (2025) claim Boltz-2 approaches FEP-level affinity performance, which sits
      in direct tension with measured mutation-blindness. Proposed reconciliation — their
      benchmarks measure *cross-ligand* ranking against a fixed target, Row A measures
      *cross-mutant* ranking against a fixed ligand, and the model is only trained for one —
      is **plausible and untested**. Either test it or state it as a hypothesis, not a finding.

## Inherited (open before this pass; from `Process/mutation-determinism-fix.md`)

- [ ] **I-20 — Re-run the task-2 validation gate with the seeded + averaged scorer.** That
      write-up's verdict (PASS) stands — the effect dwarfs seed noise — but the exact ρ figures
      are from the unseeded scorer and are not reproducible. Re-run and annotate the table.
- [ ] **I-21 — Add a sequence-keyed PDB cache for benchmarks.** There is no fold caching, so
      each benchmark run re-folds every assay via ESMFold-local (~10 min/assay on M3). A cache
      would make iteration cheap and unblocks re-running I-20 and I-1 without pain.
