# Plan — Algorithmic Mutation Scoring (ProteinMPNN) + Combinatorial Search Sketch

**Verification note:** Unlike a pure literature/docs-based design, I actually cloned
`https://github.com/dauparas/ProteinMPNN` (MIT license), installed CPU-only PyTorch,
and **ran its `--conditional_probs_only` mode end-to-end on a real PDB** (the sample
`6MRR.pdb` shipped in the repo) in a sandbox before writing any of this. Every CLI flag,
output file location, `.npz` key name, array shape, and the alphabet string below is
taken directly from that run and from reading `protein_mpnn_run.py` source — not
inferred from README prose. Where I'm relying on documentation rather than a live run
(Phase 2), I say so explicitly.

## Why ProteinMPNN over ESM-IF1 (your Q1 answer covered both — here's the tiebreak)

- Benchmarked as **more performant** than ESM-IF1 for sequence design (FoldingDiff
  paper's direct comparison), and described as still the most widely used inverse
  folding model in the current literature (property-driven inverse folding survey,
  2026).
- **Much lighter dependency footprint**: pure PyTorch + NumPy. No `torch_geometric`,
  which ESM-IF1 requires and which is notoriously version-fragile against specific
  PyTorch/CUDA combinations — exactly the kind of dependency hell `nextsteps.md`
  was already trying to avoid elsewhere in this repo.
- **Weights are tiny and ship in the git clone**: 26MB total for all four vanilla
  noise-level checkpoints (`v_48_002/010/020/030.pt`) — no separate multi-GB download
  step, confirmed by `du -sh` on the actual clone. (Some third-party blog posts claim
  ~2GB; that's wrong, or counting something else — I measured it directly.)
- **The literature already validates this exact use case**: the HERMES paper
  (Visani et al.) explicitly uses ProteinMPNN's per-site probabilities to score
  mutational effects via a log-likelihood-ratio, on a public fork built for exactly
  this purpose (`gvisani/ProteinMPNN-copy`) — same method this spec implements.

## Live verification results

Ran on `6MRR.pdb` (a 68-residue monomer), CPU only, completed in a few seconds:

```
log_p shape: (1, 68, 21)
S shape: (68,)
recovered sequence (first 40): GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEG
```

Scored all 19 substitutions at position 5 (wild-type `E`, in a helix). Results were
chemically sensible: the wild-type residue scored highest of all options, hydrophobic/
similar substitutions (`L`, `I`, `Y`, `V`) scored next-best, and `W`/`P`/`C` (a bulky
aromatic, a helix-breaker, and a residue with very different steric/chemical
properties) scored worst — matching what you'd expect a structural compatibility
scorer to say about a buried helical position. This isn't proof it's accurate on your
actual targets, but it's evidence the plumbing works and produces non-garbage output,
not just "the script ran."

---

## Task — `orchestrator/mutation_scan.py` (new module, standalone — not wired to the agent yet per your answer)

### Verified facts (from source + live run, not assumed)

- Alphabet, from `protein_mpnn_utils.py` line 50: `"ACDEFGHIKLMNPQRSTVWYX"` (21 symbols,
  `X` = unknown, always last).
- `protein_mpnn_run.py --pdb_path <single.pdb>` works directly on one PDB file —
  **no need** for the `helper_scripts/parse_multiple_chains.py` jsonl-preprocessing
  step that all of ProteinMPNN's own example scripts use; that step is only needed for
  batches of PDBs from a folder. Confirmed by reading the `args.pdb_path` branch
  (line ~164) which calls `parse_PDB()` directly.
- `--conditional_probs_only 1` writes `<out_folder>/conditional_probs_only/<pdb_name>.npz`
  with keys `log_p` (shape `[1, L, 21]`), `S` (shape `[L]`, integer indices into the
  alphabet), `mask`, `design_mask`. Confirmed both from source and from the live run.
- This mode computes `log P(s_i | backbone, rest of sequence)` — conditioned on the
  actual sequence at every *other* position, not just the backbone. This is the
  stronger option vs. `--unconditional_probs_only` (backbone alone) for scoring a
  mutation on a sequence you already have, since it accounts for the real sequence
  context around each position.

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
```

### Config additions — `config.py`

```python
PROTEINMPNN_PATH = os.getenv("PROTEINMPNN_PATH", "")
PROTEINMPNN_MODEL_NAME = os.getenv("PROTEINMPNN_MODEL_NAME", "v_48_020")
```

### `.env.example` additions

```
# Path to a local clone of https://github.com/dauparas/ProteinMPNN (MIT license).
# Model weights are included in the clone itself (~26MB) — no separate download.
#   git clone https://github.com/dauparas/ProteinMPNN.git
PROTEINMPNN_PATH=
PROTEINMPNN_MODEL_NAME=v_48_020
```

### Setup note (this is a git-clone tool, not pip-installable — same pattern as `insane.py` in `membrane.py`)

Add a line to `README.md`/`CLAUDE.md`'s setup section: `git clone
https://github.com/dauparas/ProteinMPNN.git` somewhere convenient, point
`PROTEINMPNN_PATH` at it. No conda/pip entry needed — this mirrors how `insane.py`
(membrane embedding) is already documented, so it's a consistent pattern in this repo
rather than a new one.

### Standalone entry point (per your answer — not wired into the agent yet)

Add a small CLI wrapper so you can test this manually before any agent integration:

```python
# orchestrator/mutation_scan.py, at the bottom
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

Run it as: `python -m orchestrator.mutation_scan --pdb myprotein.pdb --sequence MKT...`

### Tests — `tests/test_mutation_scan.py` (new; fully mocked, matching `test_boltz.py` style — do not invoke the real ProteinMPNN binary/weights in CI)

- `score_candidate_mutations` with a synthetic `log_p` array (mock
  `_run_proteinmpnn_conditional_probs` directly): verify the log-odds formula is
  computed correctly against hand-checked numbers, verify sort order, verify `top_k`
  truncation
- `positions` filtering: pass a subset, assert only those positions appear
- position/sequence length mismatch: verify out-of-range positions are silently
  skipped, not an error (a mismatch here likely means a stale sequence/structure pair
  upstream — worth a `logger.warning` too, add one)
- missing `proteinmpnn_dir` → `RuntimeError` before any subprocess call
- missing `protein_mpnn_run.py` at the given path → `RuntimeError` (test the exact
  check, not just "some error")
- subprocess non-zero exit → `RuntimeError` with stdout/stderr in the message
- `.npz` file missing from expected output dir → `RuntimeError`
- **one real integration test, skipped by default** (`@pytest.mark.skipif` on an env
  var, same pattern you'd want to mirror from `test_boltz.py`'s GPU-gated integration
  tests) that actually shells out to a real ProteinMPNN clone if `PROTEINMPNN_PATH` is
  set — this is the test that would have caught if my verified design above drifts
  from a future ProteinMPNN version

---

## Phase 2 sketch — combinatorial / multi-site search (per your answer: wanted eventually)

This is a sketch for a **future decision**, not ready to implement yet — the
single-site scorer above needs to exist and be sanity-checked first (see Validation
below). Flagging the landscape now so you can start thinking about which direction
fits your paper goals, since this is a bigger design decision than Phase 1.

### The problem
Single-site scores from Phase 1 are computed independently per position — they ignore
epistasis (mutation A's effect can depend on whether mutation B is also present).
Combining the top-N single-site hits into one sequence and hoping it works is a
reasonable first pass but not principled. The actual search space for k simultaneous
mutations is up to 19^k — exhaustive search is out past k≈3-4.

### Algorithm options (all evaluated head-to-head in **FLEXS**, an open-source
sequence-design benchmarking sandbox — Sinai et al., 2020, `pip install flexs`)

| Algorithm | What it is | Tradeoff |
|---|---|---|
| **AdaLead** | Simple adaptive greedy/hill-climbing: mutate around current best, keep improvements, repeat | FLEXS's own recommended default — "simple and robust," beats more complex methods in their benchmarks. Cheapest to implement correctly (~100-150 lines, no new heavy dependency) or use directly from the `flexs` package. |
| **Bayesian Optimization** | Fit a surrogate model (e.g. GP) over sequence space, use an acquisition function (UCB/EI) to pick the next candidate to actually evaluate | Most sample-efficient when each real evaluation (a full `apply_mutation`-style re-fold) is expensive — which it is here. More implementation complexity (need an embedding + surrogate + acquisition loop). |
| **CMA-ES** | Evolutionary strategy over a continuous relaxation of one-hot sequence encoding | Reasonable middle ground, less commonly used for proteins specifically than AdaLead/BO. |
| **CbAS / DbAS** | Generative-model-based adaptive sampling (VAE) | Most complex, needs a trained generative model over your sequence family — probably overkill unless you already have MSA data to train one on. |

### My read, pending your input
Start with **AdaLead** (or the `flexs` package directly, using your ProteinMPNN
scorer or the full `apply_mutation` re-fold as the "oracle"). It's the benchmark's own
recommended default for exactly this reason: simple, robust, hard to get wrong, and
empirically competitive with fancier methods. Bayesian optimization is the natural
"phase 3" if AdaLead's sample efficiency turns out to be the bottleneck once you're
paying for real re-folds per evaluation.

### Open questions for when you're ready to scope this
1. **Oracle for combos**: score by Phase 1's structural log-odds only (cheap, no
   re-fold), or actually re-fold each candidate combo and use `mean_plddt`/clash count
   (expensive, matches what `apply_mutation` already does), or — when ligands are
   present — Boltz-2 binding affinity as the fitness target?
2. **Budget**: how many real oracle evaluations (re-folds) per search session is
   acceptable? This directly decides whether AdaLead's simplicity is enough or BO's
   sample-efficiency is worth the extra complexity.
3. **Use the `flexs` package directly, or hand-roll AdaLead?** Using `flexs` is less
   code to maintain but pulls in an academic package (last PyPI release history should
   be checked for how actively maintained it still is before depending on it).

---

## Validation plan (relevant to the paper, not just engineering)

Before trusting this scorer's rankings for anything — agent-guided or combinatorial —
worth a quick sanity check against **ProteinGym** (Notin et al., 2023/2024), the
standard benchmark for exactly this task (2.5M+ real deep-mutational-scanning
measurements across 250+ assays). Pick 2-3 small ProteinGym DMS assays with known
experimental fitness values, run them through `score_candidate_mutations`, and check
Spearman correlation between your ProteinMPNN log-odds scores and the real
measurements. This doesn't need to beat state-of-the-art (ProteinMPNN alone typically
lands mid-pack on ProteinGym vs. ensemble methods like SaProt/Tranception) — it just
needs to confirm the scorer is doing something sane on your actual pipeline's PDB
outputs (predicted structures, not crystal structures) before you build anything on
top of it.

## Questions before Claude Code starts Task 1

1. **`model_name` default** — I used ProteinMPNN's own default `v_48_020` (0.20Å
   training noise). Lower-noise checkpoints (`v_48_002`) give higher native-sequence
   recovery on crystal structures; higher-noise ones (`v_48_030`) produce designs that
   fold more reliably when re-predicted by ESMFold/Boltz downstream. Since your
   pipeline always re-folds afterward (via `apply_mutation`), a higher-noise checkpoint
   might actually be the better fit — worth a quick empirical check once Task 1 lands,
   not a blocker now.
2. **Where does `PROTEINMPNN_PATH` live in practice?** — Docker container (add to
   `Dockerfile.celery`, clone during image build, ~26MB, trivial) vs. host-mounted path
   for local dev? Affects whether this needs a `Dockerfile.celery` change in the same
   task or a separate one.
3. Should the validation-against-ProteinGym step happen now (as part of landing this
   module, to catch problems early) or after, once you've seen it run on your own
   targets first? I'd lean toward running it early since it's cheap and would tell you
   immediately if something's off, but it's your call on sequencing.


