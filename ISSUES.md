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

One documented results artifact is still overwritten by a lower-effort re-run at default flags
in `bbef345`, and is recoverable from git history. The harnesses can no longer silently clobber
results (parameter-stamped paths + `--force` refusal, `benchmarks/_artifacts.py`), so this
cannot recur; the 300-pair epistasis artifact is restored. See
`Process/rowB-score-only-epistasis.md`.

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

## P1 — Config change (backed by two independent measurements)

- [ ] **I-11 — Raise `PROTEINMPNN_NUM_DECODING_ORDERS` to ≥32.** Two unrelated measurements
      agree that N=8 leaves real noise: Row B's SNR table (1.7 → 3.0 going 8 → 32) and the
      determinism gate's cross-seed spread (mean per-residue std 1.15e-02 at N=8). Cost is
      linear in K; the cheap-search savings vs re-folding leave room. Decide whether this
      becomes the default in `.env.example` or only applies under `MUTATION_SEARCH_ENABLED`.

- [ ] **I-23 — Pick `PROTEINMPNN_NUM_DECODING_ORDERS` from a v2 K-sweep (amends I-11).**
      The ≥32 recommendation was calibrated against scoring protocol v1's noise, and most of
      that noise was a batching artifact rather than decoding-order variance. The v2 rerun of
      the same 300-double panel gives per-double epistasis reliability **0.99 on all three
      domains at K=32** (was 0.89 / 0.40 / 0.51), and the diagnostic reached **0.977 at K=8**
      on 4G3O against v1's 0.533 at the same K. So K=32 is over-provisioned and the current
      default of 8 is defensible — but **only 4G3O has been measured at K=8**; the K at which
      3DKM / 3L1X reach a given reliability is unmeasured, and a fixed K still does not buy a
      fixed SNR across proteins. Action: sweep K ∈ {2, 4, 8, 16, 32} under v2 on all three
      domains, pick the knee, and keep it configurable. (The r = 0.28–0.92 "batch composition"
      figure previously cited here is withdrawn — not reproducible from any artifact, and it
      described the epistasis contrast rather than raw scores. See
      `Process/mutation-scoring-consistency.md`.)

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

- [ ] **I-25 — Decide whether `search_mutations` earns its keep at all.**
      Two independent observations say the combinatorial search may be buying nothing over the
      additive baseline. (a) The stability arm: the non-additive signal did not reliably beat the
      additive sum of single-site effects on measured double-mutant stability. (b) A smoke run of
      the new finalist stage on 4G3O (`--rounds 3 --candidates-per-round 6 --max-sites 2`) had the
      additive proposer and the `score_only` search return the **same five candidates** — the
      oracle only re-ranked a shortlist both proposers had already found. If that holds at a
      realistic search budget and on more backbones, the honest move is to demote the tier-2
      search to an optional experiment and make the additive scan the default proposer, which
      would also remove a ProteinMPNN subprocess per round from the agent's cost. Needs a proper
      proposal-overlap measurement first (the smoke run shared the AdaLead seed, so the two
      searches explored the same sequence neighbourhood by construction).
      See `Process/mutation-scoring-consistency.md`.

- [ ] **I-26 — Nothing in the pipeline measures experimental benefit.**
      Tracking the gap explicitly so it stops being rediscovered. `scan_mutations` and
      `search_mutations` score structural compatibility; the tier-3 funnel screens predicted
      structural plausibility (pLDDT, clashes) with a model that shares its inductive bias with
      the proposer; `apply_mutation` changes a sequence. No stage estimates ΔΔG, activity or
      binding, so "the pipeline suggested this mutation" cannot be upgraded to "this mutation is
      an improvement". Closing this means an independent oracle (a ΔΔG predictor, FoldX/Rosetta
      ddg_monomer, or a held-out DMS benchmark), not a better ProteinMPNN score. Agent prose and
      tool descriptions now state this; the capability gap remains.

- [ ] **I-24 — Choose benchmark assays for position-pair diversity, not variant count.**
      The Row B stability arm's 300 doubles per domain sit on only **7–16 distinct position
      pairs**, so the effective n for any cross-pair claim is single-digit and the block CIs are
      wide. Scoring more doubles buys almost nothing; an assay selection that maximises distinct
      position pairs is what would add power. Applies to any future epistasis arm.

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
- [ ] **I-18 — Row B sub-question B: the FUNCTION arm.** The stability arm is done
      (`Process/rowB-step2-stability-arm.md`): model epistasis tracks experimental stability
      epistasis weakly and positively on three X-ray Tsuboyama domains, almost entirely at the
      level of *which position pairs* couple. Remaining: one functional DMS assay, expecting a
      weaker/noisier correlation. It needs **domain-slice threading** — those assays'
      `target_seq` are full-length (GB1/SPG1 448 aa, GRB2 217 aa) and will not match a PDB chain
      gap-free the way the Tsuboyama domains do. CPU-only, no GPU.
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
