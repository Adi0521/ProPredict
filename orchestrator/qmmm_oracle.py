"""
QM/MM oracle — the tier-3 (physics) rung of the mutation-search funnel.

WHERE THIS FITS
---------------
`orchestrator/mutation_search.py` defines a cheap -> expensive oracle funnel:

    tier 1  additive_oracle       ~free      no epistasis
    tier 2  score_only_oracle     1 subproc  epistasis-aware (ProteinMPNN NLL)
    tier 3  <re-fold / physics>   minutes    real geometry + energetics

This module is a tier-3 oracle. It conforms to the SAME contract AdaLead consumes:

    oracle : Callable[[List[str]], List[float]]   # higher = better

so `make_qmmm_oracle(...)` returns a drop-in for the top-M re-fold validation pass — NOT
the per-round AdaLead loop (see BUDGET).

WHEN TO WIRE THIS IN
--------------------
Only if the affinity-invariance experiment returns **H_invariant** — i.e. Boltz-2's
affinity head is blind to mutations, so the tier-2/re-fold affinity signal can't rank
binding. In that world a physics-based interaction energy is the *fix*, not a detour. If
invariance returns H_responsive, leave this parked: the cheap oracle already ranks binding
and QM/MM only earns its keep on the covalent (Mpro/nirmatrelvir) case, where bond
formation is intrinsically quantum and no ML/MM oracle can represent it.

BUDGET (load-bearing)
---------------------
Tier-3 is minutes-to-hours per candidate. It must run ONLY on the top-M candidates AdaLead
already ranked with tier-2, never once-per-round on a full batch. `max_candidates` enforces
this in code: the returned oracle raises if handed more sequences than the cap, so it cannot
be silently dropped into the round loop.

FITNESS
-------
Per candidate we fold the mutant complex, carve a QM region around (ligand + mutated sites),
and compute a QM/MM interaction energy

    E_int(mut) = E(complex) - E(protein) - E(ligand)        [all under the same QM/MM setup]

and return fitness = -E_int  (higher = stronger predicted binding = better). The WT complex
is scored once and cached so a protocol-matched ΔΔ_int = E_int(mut) - E_int(WT) is available
for reporting — WT/mutant symmetry (same engine, same region rule, same capping) is the
methodological keystone; any asymmetry manufactures a fake Δ.

IMPLEMENTATION STATUS
---------------------
The interface, gating, budget rule, WT-caching, and the fold path are REAL and final: the
fold step is wired to orchestrator.backends.boltz.call_boltz (whose homodimer + affinity-key
handling is already fixed in the backend). The two heavy internals — `_carve_qm_region` and
`_engine_energy` — remain a SCAFFOLD, raising NotImplementedError with the exact decisions
they require. None of the xtb/ORCA specifics below are live-verified yet — treat them as
design intent to confirm against the tools, not validated behavior. Build tier-3 behind this
seam once the invariance result says you need it.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

# Reuse the funnel's own sequence-threading so a candidate here is defined exactly as it is
# for tiers 1-2 (equal-length substitutions of the WT).
try:
    from orchestrator.mutation_search import apply_mutations, mutations_from_sequences
except Exception:  # pragma: no cover - allows standalone import while scaffolding
    apply_mutations = None
    mutations_from_sequences = None


# --- config (mirrors the os.getenv convention in config.py) -----------------------------
QMMM_ENABLED = os.getenv("QMMM_ENABLED", "False") == "True"
QMMM_ENGINE = os.getenv("QMMM_ENGINE", "xtb")  # "xtb" (GFN2 rung) | "orca" (DFT rung)
QMMM_QM_RADIUS = float(os.getenv("QMMM_QM_RADIUS", "5.0"))  # Å shell around ligand + mut sites
QMMM_MAX_CANDIDATES = int(os.getenv("QMMM_MAX_CANDIDATES", "5"))  # tier-3 budget cap
QMMM_ORCA_METHOD = os.getenv("QMMM_ORCA_METHOD", "r2SCAN-3c")  # DFT rung; dispersion built in
QMMM_XTB_GFN = os.getenv("QMMM_XTB_GFN", "2")  # GFN2-xTB screening rung


# A fold function turns (sequence, ligand_smiles, protein_copies, seed) -> complex PDB
# string. Injected so the oracle is unit-testable and so it reuses the pipeline's real
# folding path (orchestrator.backends.boltz.call_boltz) rather than forking one.
# protein_copies is a homo-oligomer count: HIV-1 protease is an obligate homodimer, so its
# active site only exists at copies=2 — the affinity-invariance system uses 2.
FoldFn = Callable[[str, str, int, int], str]


def _default_fold_fn(sequence: str, ligand_smiles: str, protein_copies: int = 1, seed: int = 0) -> str:
    """Fold the holo complex via the existing Boltz backend and return the complex PDB.

    Wires to call_boltz(sequence, context={"ligands": [...], "protein_copies": N}, seed,
    protein_copies=N) -> StructurePrediction; the complex PDB is `.structure_pdb`. The
    backend already handles the homodimer YAML (id: [A, B]) and the affinity-key parsing,
    so no pre-processing is needed here. Imported lazily so this module stays importable
    (and unit-testable with an injected fold_fn) without a GPU / Boltz install present.
    """
    from orchestrator.backends.boltz import call_boltz  # noqa: WPS433 (lazy on purpose)

    ctx = {
        "ligands": [{"name": "ligand", "smiles": ligand_smiles}],
        "protein_copies": protein_copies,
    }
    pred = call_boltz(sequence, context=ctx, seed=seed, protein_copies=protein_copies)
    if not getattr(pred, "structure_pdb", None):
        raise RuntimeError("call_boltz returned no structure_pdb for the holo complex.")
    return pred.structure_pdb


def _carve_qm_region(complex_pdb: str, mutated_positions: List[int], radius: float) -> dict:
    """Select the QM subsystem and cap the boundary.

    DECISIONS THIS MUST MAKE (all standard QM/MM, reuse a published selector — e.g. a
    Kulik-style charge/CSA descriptor — rather than a hand-rolled cutoff, and cite it):
      * QM set = ligand (always) + protein residues with any atom within `radius` Å of the
        ligand OR of any mutated-site Cα. Whole residues, never partial side chains split
        mid-group.
      * Cut only non-polar C-C bonds at the boundary; cap with hydrogen link atoms.
      * Record total charge and spin of the QM region (protonation-state dependent — reuse
        the existing PropKa path so WT and mutant get identical protonation rules).
    Returns a dict describing the QM region (atom indices, link atoms, charge, spin) for the
    engine step. Region-size sensitivity is an ablation, not a fixed constant.
    """
    raise NotImplementedError(
        "QM-region carving + link-atom capping not implemented. Reuse an existing automated "
        "QM-region selector; do NOT reinvent it. Keep WT and mutant on the identical rule."
    )


def _engine_energy(region: dict, engine: str) -> float:
    """Run the QM(/MM) single-point/optimization and return the energy used for E_int.

    engine == 'xtb'  -> GFN2-xTB (fast screening rung; dispersion built in). Via tblite/ASE.
    engine == 'orca' -> DFT rung (QMMM_ORCA_METHOD, e.g. r2SCAN-3c optimize; a dispersion-
                        corrected hybrid single point for the final interaction energy).
    Same engine/method for the three subsystems (complex, protein, ligand) and for WT vs
    mutant — otherwise E_int / ΔΔ_int is meaningless.
    """
    raise NotImplementedError(
        f"Engine '{engine}' not wired. Start with xtb (GFN2) for the screening rung; add ORCA "
        f"({QMMM_ORCA_METHOD}) for the DFT rung. Confirm invocation/parse against the tools."
    )


def _interaction_energy(complex_pdb: str, mutated_positions: List[int], engine: str, radius: float) -> float:
    """E_int = E(complex) - E(protein) - E(ligand), all under one QM/MM setup."""
    region = _carve_qm_region(complex_pdb, mutated_positions, radius)
    e_complex = _engine_energy(region, engine)
    e_protein = _engine_energy({**region, "_subsystem": "protein"}, engine)
    e_ligand = _engine_energy({**region, "_subsystem": "ligand"}, engine)
    return e_complex - e_protein - e_ligand


def make_qmmm_oracle(
    wild_type: str,
    ligand_smiles: str,
    *,
    fold_fn: Optional[FoldFn] = None,
    engine: str = QMMM_ENGINE,
    radius: float = QMMM_QM_RADIUS,
    max_candidates: int = QMMM_MAX_CANDIDATES,
    protein_copies: int = 1,
    seed: int = 0,
) -> Callable[[List[str]], List[float]]:
    """Build a tier-3 QM/MM fitness oracle: List[str] -> List[float], higher = better.

    Intended use is the top-M validation pass, e.g.:

        top = result.candidates[:M]                       # ranked by tier-2 already
        qmmm = make_qmmm_oracle(wt, smiles, max_candidates=M, protein_copies=2)  # HIV-PR
        fitnesses = qmmm([c.sequence for c in top])       # protocol-matched vs cached WT

    Raises at BUILD time if QMMM_ENABLED is false (graceful degradation — same ethos as the
    rest of the optional-tool stack), and at CALL time if handed more than `max_candidates`
    sequences (the guard that keeps it out of the AdaLead round loop). `protein_copies=2`
    for obligate homodimers like HIV-1 protease.
    """
    if not QMMM_ENABLED:
        raise RuntimeError(
            "QMMM_ENABLED is not True. This tier-3 oracle is intended only after the "
            "affinity-invariance experiment returns H_invariant (see module docstring). "
            "Set QMMM_ENABLED=True in .env to build it."
        )

    fold = fold_fn or _default_fold_fn
    wt_positions: List[int] = []  # WT has no mutated sites; region = ligand + pocket only

    # Score the WT complex ONCE and cache it, so every candidate's ΔΔ_int is protocol-matched
    # against the same reference (and we never re-pay for it).
    wt_complex = fold(wild_type, ligand_smiles, protein_copies, seed)
    wt_e_int = _interaction_energy(wt_complex, wt_positions, engine, radius)

    cache: Dict[str, float] = {wild_type: -wt_e_int}

    def oracle(sequences: List[str]) -> List[float]:
        if max_candidates and len(sequences) > max_candidates:
            raise ValueError(
                f"QM/MM oracle handed {len(sequences)} candidates but the tier-3 budget cap "
                f"is {max_candidates}. Truncate to the top-M AdaLead candidates first — this "
                f"oracle must not run in the per-round loop."
            )
        out: List[float] = []
        for seq in sequences:
            if seq in cache:
                out.append(cache[seq])
                continue
            if len(seq) != len(wild_type):
                raise ValueError("QM/MM oracle requires equal-length (substitution-only) seqs.")
            positions = [i for i in range(len(seq)) if seq[i] != wild_type[i]]
            mut_complex = fold(seq, ligand_smiles, protein_copies, seed)  # thread + re-fold
            e_int = _interaction_energy(mut_complex, positions, engine, radius)
            fitness = -e_int  # higher = stronger binding = better
            cache[seq] = fitness
            out.append(fitness)
        return out

    # Expose the cached WT reference so callers can report ΔΔ_int = fitness(mut) - fitness(WT).
    oracle.wt_fitness = cache[wild_type]  # type: ignore[attr-defined]
    return oracle
