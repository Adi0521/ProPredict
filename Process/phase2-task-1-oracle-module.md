# Phase 2 — Task P2-1: Oracle module (`orchestrator/mutation_search.py`)

**Date:** 2026-07-21
**Source:** `mutation-plans/Process-plan-phase2.md` § "P2-1 — Oracle module"
**Scope:** the two fitness oracles + mutation-representation helpers. No AdaLead search yet
(P2-2), no pipeline re-fold funnel yet (P2-3).

## What was built

New module `orchestrator/mutation_search.py` with two oracles, both returning **higher =
better**:

### Tier 1 — `additive_oracle(log_p, mutations)` (pure)
Sum over a multi-mutant's substitutions of the single-site structural log-odds
`log_p[pos, mut] - log_p[pos, wt]` — the same math as
`mutation_scan.score_candidate_mutations`, summed across sites. `log_p` is the `[L, 21]`
mean matrix from `_run_proteinmpnn_conditional_probs` on the WT structure. Independent by
construction, so it **cannot see epistasis** — that is exactly why it is a seed / pre-filter,
not the search-loop oracle (AdaLead over a pure sum is vacuous: the optimum is just the best
substitution at each site).

### Tier 2 — `score_only_oracle(pdb_string, sequences, ...)` (epistasis-aware, batched)
Threads each mutant sequence onto the WT backbone and runs ProteinMPNN
`--score_only 1 --path_to_fasta` **once for the whole list** (one model load), reading each
sequence's `global_score` (mean NLL of the whole structure-sequence, autoregressive → sees
epistasis). Returns the **negated** mean NLL so higher = better. This is the real search-loop
oracle; call it once per AdaLead round with that round's candidates.

### Mutation helpers (shared, will be reused by AdaLead)
`parse_mutation` / `format_mutation` (`"A12V" ↔ ("A",12,"V")`), `apply_mutations`
(validates the stated WT residue matches — catches sequence/mutation desync), and
`mutations_from_sequences` (diff two equal-length sequences into `["A12V", ...]`, the bridge
from AdaLead's raw sequences to a mutation description).

## Key implementation decisions
- **Reuse, not duplicate.** Imports `_ALPHABET`, `_STANDARD_AA`, and
  `_run_proteinmpnn_conditional_probs` from `mutation_scan.py` so the `[L,21]` ordering, the
  non-zero-seed guard, and decoding-order averaging can never drift between the two modules.
- **Output mapping.** ProteinMPNN writes `{name}_fasta_{N}.npz` (N = 1-indexed fasta order;
  native PDB goes to `{name}_pdb`, skipped). The parser globs `*_fasta_*.npz` and maps the
  trailing integer back to the 0-indexed input, so results are aligned to input order
  regardless of the derived PDB `name` or filesystem glob order. Missing indices raise
  (names the offending candidates) rather than silently returning a short list.
- **Length invariant.** ProteinMPNN threads a fasta seq into `S[:, :len(seq)]`, so a wrong
  length silently scores a chimera (short) or crashes (long). `score_only_oracle` rejects
  unequal-length batches up front. (Mutation search only ever produces substitutions, so
  equal length ↔ structure length holds in practice.)
- **Standalone**, params not `config.py` — same convention as `mutation_scan.py`. Wiring to
  `MUTATION_SEARCH_*` config happens when AdaLead lands.

## Verification
- `pytest tests/test_mutation_search.py` — **25 passed**. Pure helpers + `additive_oracle`
  tested directly against a synthetic `log_p` (same construction as `test_mutation_scan`);
  `score_only_oracle` mocked at the subprocess boundary with a fake `subprocess.run` that
  writes ProteinMPNN-shaped `*_fasta_N.npz` files — exercises negation, input-order mapping,
  decoding-order averaging, flag construction, and the missing-output / bad-length / seed-0 /
  missing-dir error paths.
- `pytest tests/test_mutation_scan.py tests/test_boltz.py` — 22 passed, 3 skipped (shared
  imports intact, nothing broke).
- **Real-binary check** (not just the mock): ran `score_only_oracle` against the actual
  ProteinMPNN clone on a 68-residue monomer (`6MRR`) with WT + two point mutants. Output
  parsed correctly; **WT scored best** (fitness `-1.4744`), a disruptive `E20P` proline
  substitution scored worst (`-1.5572`) — sensible, and confirms the `global_score` /
  filename assumptions against real output.

## Next
P2-2 — hand-rolled AdaLead (population + greedy-around-best + recombination, `MAX_SITES`
k-cap, per-round budget, deterministic given seed) against this oracle interface (default
tier-2 `score_only`). Tested on a synthetic non-additive landscape with a planted epistatic
pair — where the search is proven to beat naive top-N single-site stacking.
