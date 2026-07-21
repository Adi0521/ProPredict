# Plan — Row A: Is Boltz-2's affinity head blind to resistance mutations?

**Status: two blocking bugs found and verified. The experiment cannot run today.
Both fixes are small. Dataset is built and attached; analysis is written and tested.**

Everything below was verified against source or real data in a sandbox, not from docs
prose or memory. Boltz-2 inference itself needs a GPU and could not be run here — that
part goes to Modal.

---

## 0. Two blocking bugs (fix before anything else)

### Bug 1 — the affinity value is never read. `affinity_score` is always `None`.

`orchestrator/backends/boltz.py:144`:
```python
affinity_score = aff_data.get("affinity")
```

Boltz-2 does not write a key called `affinity`. Verified directly against the source
of `boltz==2.2.1` (`src/boltz/data/write/writer.py:308-326`), which builds the JSON as:

```python
affinity_summary = {
    "affinity_pred_value": pred_affinity_value.item(),
    "affinity_probability_binary": pred_affinity_probability.item(),
}
if "affinity_pred_value1" in prediction:
    affinity_summary["affinity_pred_value1"] = ...
    affinity_summary["affinity_probability_binary1"] = ...
    affinity_summary["affinity_pred_value2"] = ...
    affinity_summary["affinity_probability_binary2"] = ...
```

So `.get("affinity")` returns `None` on **every run that has ever executed**. Consequences:
- `StructurePrediction.affinity_score` is always `None`.
- The success log's `f", affinity: {affinity_score:.3f} kcal/mol"` branch never fires.
- `apply_mutation`'s `if pred.affinity_score is not None:` never fires.
- The entire affinity capability — the thing the research plan's ligand-binding wedge
  depends on — is silently dead and has never produced a number.

**Fix:**
```python
affinity_score: Optional[float] = None
affinity_probability: Optional[float] = None
if affinity_binder:
    aff_hits = glob.glob(os.path.join(out_dir, "**", "affinity_*.json"), recursive=True)
    if aff_hits:
        with open(aff_hits[0]) as fh:
            aff_data = json.load(fh)
        # Boltz-2 writes affinity_pred_value (log10 IC50, IC50 in uM) and
        # affinity_probability_binary. There is no "affinity" key.
        affinity_score = aff_data.get("affinity_pred_value")
        affinity_probability = aff_data.get("affinity_probability_binary")
```
Also widen the glob: the current `"*affinity*.json"` would match `affinity_*.json` but
being explicit prevents a future file (e.g. a `pae_affinity_*.json`) sorting first.
Consider surfacing `affinity_probability_binary` on the schema too — it's a different
quantity trained on different data (binder-vs-decoy detection) and is the right output
for a hit-discovery framing, whereas `affinity_pred_value` is the one for SAR/Δ work.

**Add a regression test** (`tests/test_boltz.py`): mock an `affinity_*.json` with the
real key set and assert `affinity_score` is populated. A test asserting `is not None`
would have caught this on day one.

### Bug 2 — `call_boltz` cannot model the lead system.

```python
protein_entry: dict = {"id": "A", "sequence": sequence}
sequences: list = [{"protein": protein_entry}]
```

One protein chain. **HIV-1 protease is an obligate homodimer** — the active site forms
at the dimer interface, and each monomer contributes one catalytic aspartate (D25/D25').
A 99-residue monomer has no inhibitor binding pocket at all. Every HIV-PR number the
current backend could produce would be meaningless.

Boltz supports homomers via an id list — verified in
`src/boltz/data/parse/schema.py:1094-1097`, which accepts `id` as either `str` or `list`:
```yaml
sequences:
  - protein:
      id: [A, B]
      sequence: PQITLWQRPL...
  - ligand:
      id: C
      smiles: "CC(C)CN(C[C@H](..."
properties:
  - affinity:
      binder: C
```

**Fix:** give `call_boltz` a way to express chain multiplicity — e.g. a
`protein_copies: int = 1` arg (or a `context["protein_copies"]`), emitting
`{"id": ["A", "B"], ...}` when >1 and shifting ligand chain IDs accordingly (the
current `chr(ord("B") + i)` would collide with a second protein chain named B).

### Bug 3 (minor) — wrong units in the log line

`f", affinity: {affinity_score:.3f} kcal/mol"` is wrong. Boltz-2's own README (verified
in the 2.2.1 sdist) states `affinity_pred_value` reports **`log10(IC50)`, with IC50 in
μM**. It is not kcal/mol. Rowan's conversion is `(6 - affinity) * 1.364` and they
explicitly caution it's a non-standard pIC50 — don't compare it to other packages'
numbers without care.

---

## 1. Why the units are a gift

Because `affinity_pred_value = log10(IC50 [μM])`:

```
pred_Δ = affinity_pred_value(mutant) − affinity_pred_value(WT)
       = log10(IC50_mut) − log10(IC50_wt)
       = log10(IC50_mut / IC50_wt)
       = log10(predicted fold-change)
```

And **fold-change is exactly what PhenoSense/Stanford HIVDB reports**. So the comparison
is `pred_Δ` vs `log10(experimental fold-change)` — direct, no unit conversion, no fitted
scaling, no free parameters. A slope of 1.0 would mean perfect Δ-tracking; a slope of 0
means blindness. This is a much cleaner readout than the research plan's Δaffinity
correlation, which would have needed a kcal/mol conversion the code was getting wrong.

---

## 2. What the data actually supports (this changes the plan's design)

Fetched the real Stanford HIVDB PhenoSense dataset
(`https://hivdb.stanford.edu/download/GenoPhenoDatasets/PI_DataSet.txt`, 640KB, 2171
isolates, fold-change for 8 PIs + per-position mutation columns).

**The plan's single-pocket-mutation design has no data.** Of 2171 isolates, only **28**
carry exactly one mutation, and they are almost all polymorphisms with no resistance
effect:

```
mutation      FPV    ATV    IDV    LPV    NFV    SQV    TPV    DRV
L63P          0.9    0.9    1.0    1.0    1.7    1.0    1.4    0.8
R57G          1.0    1.0    1.0     NA    1.0     NA     NA     NA
I64V          0.6    0.8    0.9    0.9    1.0    0.7    0.8    0.6
A71V          0.6    0.9    1.2     NA    1.4     NA     NA     NA
I84C          1.7     NA    1.2    0.4   15.3    6.1     NA     NA   <- rare real signal
I50L          0.1    2.0    0.1     NA    0.1     NA     NA     NA
I50L          0.2    5.4    0.1     NA    0.2     NA     NA     NA   <- same mutation,
I50L          0.3    3.0    0.1     NA    0.1     NA     NA     NA      2.0/5.4/3.0 on ATV
```

Two things to absorb. First, there is essentially **no dynamic range** — a lone V82A or
I84V simply doesn't occur in clinical isolates, because resistance is accumulated under
drug pressure alongside compensatory mutations. Second, the I50L rows are the *same
mutation measured three times*: ATV fold-change 2.0, 5.4, 3.0. That ~2.7× spread is the
experimental noise floor, and it is comparable to the entire effect size of most single
mutations. Any single-mutation Δ study on this data would be fitting noise.

**The multi-mutant isolates do work, and are arguably a better test:**

| | |
|---|---|
| Clean isolates (no mixtures/indels) | **906** of 2171 |
| Dynamic range | **3 log10 units** (0.1 → 100 fold) |
| Censoring at 100 (assay ceiling) | DRV 6%, NFV 9% |
| Median mutation load, DRV fold > 10 | **19.5** |
| Median mutation load, DRV fold < 2 | 7 |

A resistant isolate carries ~19.5/99 ≈ **20% mutation load** — squarely inside the
regime where Feldman et al. showed AF3 stays invariant (they went to 40%). So this is a
*generous* test: if Boltz-2 can't separate a 20%-mutated, 100×-resistant variant from
WT, that's a clean and damning result. And it extends Feldman et al. from apo-structure
invariance into the ligand/affinity domain, which is exactly the open space your plan
identified and which they did not touch.

---

## 3. The experiment (dataset built, attached)

`build_hiv_pr_dataset.py` → `hiv_pr_resistance_dataset.json`, already generated:

```
WT reference runs: 4   isolate runs: 160   total: 164
  DRV: n= 40  log10FC -0.52..+2.00  mutation load median=8  censored=3
  NFV: n= 40  log10FC -1.00..+2.00  mutation load median=8  censored=3
  SQV: n= 40  log10FC -1.00..+2.00  mutation load median=7  censored=4
  IDV: n= 40  log10FC -1.00..+2.00  mutation load median=8  censored=2
```

- Isolates stratified across the log10 fold-change range (not sampled at random — a
  random sample clusters at the median and wastes the range).
- Mixtures (`"IV"` at a position) and indels excluded — these are mixed viral
  populations, not a single sequence, and can't be folded.
- Consensus subtype-B sequence verified against the dataset's own mutation annotations.
- Ligand SMILES pulled from PubChem **IsomericSMILES** — `CanonicalSMILES` drops
  stereochemistry, and PI stereochemistry is load-bearing (your plan flags darunavir
  chirality specifically).

**Run:** `python benchmark_affinity_invariance.py hiv_pr_resistance_dataset.json results.jsonl`
164 Boltz-2 runs on a 198-residue dimer + ligand. Rough estimate 1–3 min/run on an
A100 → **~3–8 GPU-hours**, resumable (skips completed rows on rerun). Needs the Modal
path since there's no local GPU.

**The MSA ablation is worth running (`--no-msa`).** AlphaInterp (bioRxiv, Apr 2026)
found AF3's accuracy survives heavily degraded MSAs but collapses when they're removed —
the model leans on evolutionary context over raw sequence. An HIV-protease MSA contains
both WT and every resistant variant, which is a *plausible mechanism* for invariance:
the MSA washes the point mutations out. If invariance holds MSA-on and relaxes MSA-off,
that's a mechanism and a figure, not just a phenomenon. Doubles the cost to ~6–16 GPU-h.

## 4. Analysis (written and tested)

`analyze_affinity_invariance.py` — validated end-to-end on synthetic data under both hypotheses before it
ever sees a real number:

```
SIMULATED, thesis holds:  ALL  148  spearman +0.032  slope -0.001  pred/exp spread 0.05
SIMULATED, thesis fails:  ALL  148  spearman +0.899  slope +0.826  pred/exp spread 0.88
```

**One design finding from that validation:** with pure noise, per-drug Spearman ranged
from **−0.349 (NFV) to +0.231 (SQV)** at n≈37. At this per-drug sample size, spurious
correlations of |ρ|≈0.35 appear from nothing. So per-drug ρ is not trustworthy — the
**pooled slope** and the **predicted/experimental spread ratio** are the statistics to
report. The script does this and refuses to call a verdict on ρ alone: a tiny-but-
significant ρ with slope ≈0.02 is still practical blindness, and that distinction is
the entire point.

Spearman over Pearson throughout, because fold-change is censored at 100.

---

## 5. What each outcome buys you

- **Invariant** (expected): the research plan's central premise is confirmed and, more
  importantly, *quantified in the ligand domain for the first time* — Feldman et al. did
  apo structures, nobody has shown the affinity head is blind. That's a workshop paper on
  its own, it justifies the whole physics ladder, and it cost ~8 GPU-hours instead of six
  months of ORCA.
- **Responsive**: the plan's foundation is gone, and you found out for 8 GPU-hours
  instead of after building a QM/MM tier on a false premise. Also a publishable result —
  "Boltz-2's affinity head recovers clinical resistance" is a *positive* finding people
  would cite.
- **Ambiguous / weak**: the slope and spread-ratio tell you whether weak-but-real is
  practically useful. Report honestly; don't let a p-value upgrade it.

Either way this is the cheapest decisive thing available, and it gates everything
downstream in the research plan.

## 6. Open questions

1. **Do you want the two bug fixes landed in `orchestrator/backends/boltz.py` first, or
   run the standalone script?** The script deliberately bypasses the backend so the
   experiment isn't blocked on a refactor — but the affinity-key bug should be fixed
   regardless, since it silently disables a headline capability.
2. **Train/test leakage is real and unfixable here.** HIV protease + PI co-crystals are
   all over the PDB and certainly in Boltz-2's training set. That cuts *toward* the
   invariance hypothesis (memorised WT-like complexes → same answer regardless of
   mutation), so it doesn't threaten a null result — but it does mean a *positive* result
   would need a leakage caveat. Worth stating up front either way.
3. **Should `affinity_probability_binary` be a second readout?** It's trained on
   different data with different supervision. If `affinity_pred_value` is flat but the
   binary probability moves, that's interesting and worth capturing — the runner already
   records both.

---

## 7. References

Confidence is marked explicitly. **[verified]** = I fetched the source/BibTeX/DOI
directly in this session. **[check]** = the work exists and the claim is right, but
confirm exact volume/page/DOI before it goes in a paper — don't paste these into a
bibliography unchecked.

### Core to this experiment

1. **[verified]** Passaro, S.\*, Corso, G.\*, Wohlwend, J.\*, Reveiz, M.\*, Thaler, S.\*,
   Somnath, V.R., Getz, N., Portnoi, T., Roy, J., Stark, H., Kwabi-Addo, D., Beaini, D.,
   Jaakkola, T., Barzilay, R. (2025). *Boltz-2: Towards Accurate and Efficient Binding
   Affinity Prediction.* bioRxiv. doi:10.1101/2025.06.14.659707
   — BibTeX taken verbatim from the official repo (`github.com/jwohlwend/boltz`).
   Establishes that `affinity_pred_value` is `log10(IC50)`, IC50 in μM (README, verified
   in the 2.2.1 sdist). Note for §2 of the research plan: Boltz-2 claims to be the first
   AI model to **approach FEP-level performance** on small-molecule affinity — which sits
   in direct tension with a thesis that ML affinity needs a QM tier to be useful. Worth
   confronting head-on rather than around.

2. **[verified]** Boltz-2 source, v2.2.1. `src/boltz/data/write/writer.py:308-326`
   (affinity JSON keys); `src/boltz/data/parse/schema.py:1094-1097` (`id` accepts `str`
   or `list`, enabling homodimers). These are the two ground truths behind Bugs 1 and 2.
   Pin the version — if boltz changes these, the fixes need revisiting.

3. **[verified]** Feldman, J., Brogi, M., Skolnick, J. (2026). *Adversarial Sequence
   Mutations in AlphaFold and ESMFold Reveal Nonphysical Structural Invariance,
   Confidence Failures, and Concerns for Protein Design.* bioRxiv 2026.02.25.708002.
   doi:10.64898/2026.02.25.708002 (posted 26 Feb 2026).
   — The plan's central citation, and stronger than the plan states: invariance to
   mutation of up to **40% of residues** including deliberately destabilizing ones, with
   average TM-score **≥0.63** to the original prediction, and the **AF3 ranking score
   also stays high** (confidence invariance, not just structural). Covers AF3 **and**
   ESMFold across 200 proteins, on **apo structures**. The gap this experiment fills:
   nobody has asked the same question of a **ligand-bound affinity head**.

4. **[verified]** Stanford HIV Drug Resistance Database — genotype-phenotype datasets.
   `https://hivdb.stanford.edu/pages/genopheno.dataset.html`; PI dataset fetched from
   `https://hivdb.stanford.edu/download/GenoPhenoDatasets/PI_DataSet.txt`
   (640 KB, 2171 isolates, 8 PIs, downloaded and parsed in this session).
   Susceptibility is the **PhenoSense** assay (Monogram Biosciences).
   **[check]** The canonical paper citation for the *dataset specifically* — the usual
   ones are Rhee et al., *Nucleic Acids Res.* 2003 (database) and Rhee et al., *PNAS*
   2006 (genotypic predictors / the ML datasets). Confirm which the download page asks
   for; don't guess.

5. **[check]** Shafer, R.W. (2006). *Rationale and Uses of a Public HIV Drug-Resistance
   Database.* J. Infect. Dis. 194(Suppl 1):S51–S58. — Confirms the PhenoSense/Antivirogram
   assay provenance of the phenotype data. Verify the supplement pagination.

6. **[verified]** PubChem PUG-REST, `IsomericSMILES` endpoint — SMILES for darunavir,
   nelfinavir, saquinavir, indinavir (and atazanavir, lopinavir, amprenavir, tipranavir,
   fetched but unused so far). Retrieved 2026-07-17. **Use `IsomericSMILES`, not
   `CanonicalSMILES`** — the latter silently drops stereochemistry.
   **[check]** Cite PubChem itself via Kim et al., *Nucleic Acids Res.* (most recent
   PubChem database issue paper).

7. **[check]** Mirdita, M., Schütze, K., Moriwaki, Y., Heo, L., Ovchinnikov, S.,
   Steinegger, M. (2022). *ColabFold: making protein folding accessible to all.* Nature
   Methods 19:679–682. — Required by Boltz's own citation guidance whenever
   `--use_msa_server` is used, which the MSA-on arm does.

### Motivating the framing (why row A is the right first experiment)

8. **[check]** Pak, M.A., Markhieva, K.A., Novikova, M.S., Petrov, D.S., Vorobyev, I.S.,
   Maksimova, E.S., Kondrashov, F.A., Ivankov, D.N. (2023). *Using AlphaFold to predict
   the impact of single mutations on protein stability and function.* PLOS ONE.
   — ΔpLDDT vs ΔΔG gives PCC ≈ **−0.17**; global mean pLDDT shows **no** correlation.
   This is why the agent's pLDDT-based feedback loop can't grade mutations, and why the
   plan correctly moved the target from stability to ligand binding. Verify volume/e-ID.

9. **[verified]** *AlphaInterp: Probing AlphaFold 3's Internal Representations Reveals
   Evolutionary Determinants of Predicted Structure and Confidence.* bioRxiv
   2026.04.22.720175 (Apr 2026).
   — Accuracy survives heavily degraded MSAs but **collapses when MSAs are removed**;
   the model leans on phylogenetic diversity over raw sequence. This is the rationale for
   the `--no-msa` ablation arm: an HIV-protease MSA contains WT *and* resistant variants,
   so MSA-washout is a **mechanism** for invariance, not just a restatement of it.
   **[check]** author list/venue if cited formally.

### Prior art the research plan must engage with (see §2 discussion)

10. **[verified]** Taguchi, M., Oyama, R., Kaneso, M., Hayashi, S. (2022). *Hybrid QM/MM
    Free-Energy Evaluation of Drug-Resistant Mutational Effect on the Binding of an
    Inhibitor Indinavir to HIV-1 Protease.* J. Chem. Inf. Model. **62**(5):1328–1344.
    doi:10.1021/acs.jcim.1c01193
    — QM/MM on HIV-1 protease resistance at **V82T/I84V** — the plan's own headline
    residues — recovering the experimental 2.5–3.0 kcal/mol shift. Critically, they
    combined QM/MM with **conformational sampling via long MD**, because static QM/MM
    energies are not free energies. This is both the closest prior art and the evidence
    that the plan's static ladder is on the wrong axis.

11. **[check]** *Computational Studies of a Mechanism for Binding and Drug Resistance in
    the Wild Type and Four Mutations of HIV-1 Protease with a GRL-0519 Inhibitor.* Int. J.
    Mol. Sci. 2016, 17(6):819. doi:10.3390/ijms17060819
    — MM-PBSA on **D30N, I50V, I54M, V82A**; ranking of calculated binding free energies
    accords with experiment. Fixed-charge MM already ranks the plan's headline mutations
    correctly, which undercuts the "polarization-blind MM fails here" premise.

12. **[check]** *A Contribution to the Drug Resistance Mechanism of Darunavir, Amprenavir,
    Indinavir, and Saquinavir Complexes with HIV-1 Protease Due to Flap Mutation I50V: A
    Systematic MM–PBSA and Thermodynamic Integration Study.* J. Chem. Inf. Model. 2013.
    doi:10.1021/ci4002102 — TI/MM-PBSA on the same system; FEP/TI is the real baseline
    the plan's matrix is missing.

13. **[check]** Ross, G.A., Lu, C., Albanese, S.K., Houang, E., Abel, R., Harder, E.D.,
    Wang, L. *The maximal and current accuracy of rigorous protein-ligand binding free
    energy calculations.* Commun. Chem. 2023. — For the accuracy ceiling of FEP; useful
    when arguing what a physics tier could even buy.

### Already used elsewhere in this repo (listed for a single bibliography)

14. **[check]** Dauparas, J., et al. (2022). *Robust deep learning-based protein sequence
    design using ProteinMPNN.* Science 378(6615):49–56. — `orchestrator/mutation_scan.py`.
15. **[check]** Notin, P., et al. (2023). *ProteinGym: Large-Scale Benchmarks for Protein
    Fitness Prediction and Design.* NeurIPS Datasets & Benchmarks. —
    `Process/mutation-task-2-validation-gate.md`.
16. **[check]** Buttenschoen, M., Morris, G.M., Deane, C.M. (2024). *PoseBusters:
    AI-based docking methods fail to generate physically valid poses or generalise to
    novel sequences.* Chemical Science 15:3130–3139. — relevant if the pose-validity gate
    in the research plan's §4 gets built. Reinforced by Passaro et al.'s own admission
    that Boltz-2 poses can carry incorrect bond lengths/angles, wrong stereochemistry at
    chiral centres, and non-planar aromatic rings.
