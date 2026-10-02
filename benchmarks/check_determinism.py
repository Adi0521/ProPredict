"""
Tier-1 determinism gate for the ProteinMPNN mutation scorer.

mutation_scan.py already builds determinism IN (scores are the mean over `num_decoding_orders`
random decoding orders at a FIXED non-zero seed, and seed=0 is rejected). This harness is the
missing *check* that the guarantee actually holds end-to-end — a class of failure the mocked
unit tests can't see (GPU/torch nondeterminism, filesystem ordering, a silently-unset seed).

Three things, cheapest first:

  0. seed=0 pre-flight  — runs WITHOUT weights: confirms the scorer rejects seed=0 (which
     ProteinMPNN's `if args.seed:` would treat as unset and randomize). If this fails you have
     a determinism hole regardless of anything downstream.

  1. same-seed determinism  — the gate. Run conditional_probs twice at the SAME (seed,
     num_decoding_orders) and assert the [L,21] log_p arrays are byte-identical. PASS/FAIL,
     and the process exit code reflects it so this can gate CI.

  2. cross-seed spread  — a REPORT, not a gate. Run at several seeds and measure how much the
     per-residue log_p still moves — i.e. the residual decoding-order noise the mean-over-N
     is controlling. Lower is better; it should shrink as you raise --num-decoding-orders.
     This is the same noise the ProteinGym benchmark summarizes as spearman_std.

Run (ProPredict env, PROTEINMPNN_PATH set in .env; ProteinMPNN runs fine on CPU here):
    python -m benchmarks.check_determinism
    python -m benchmarks.check_determinism --pdb path/to/single_chain.pdb --num-decoding-orders 16

NOTE: ProteinMPNN scores a single chain. The default structure below is one chain of HIV-PR;
point --pdb at any single-chain PDB to check determinism on your own target.
"""
import argparse
import os
import sys
import tempfile

import numpy as np

from config import PROTEINMPNN_PATH
from orchestrator.mutation_scan import _run_proteinmpnn_conditional_probs

DEFAULT_PDB = os.path.join(
    os.path.dirname(__file__), "epistasis_structures", "4G3O.raw.pdb"
)


def _logp(pdb_string: str, seed: int, n_orders: int) -> np.ndarray:
    """One [L, 21] conditional-probs run at a fixed (seed, num_decoding_orders)."""
    with tempfile.TemporaryDirectory() as td:
        return _run_proteinmpnn_conditional_probs(
            pdb_string, td, PROTEINMPNN_PATH, seed=seed, num_decoding_orders=n_orders
        )


def _preflight_seed_zero() -> bool:
    """Confirm seed=0 is rejected before any subprocess. Runs without weights."""
    with tempfile.TemporaryDirectory() as td:
        try:
            _run_proteinmpnn_conditional_probs("PLACEHOLDER", td, PROTEINMPNN_PATH, seed=0)
        except ValueError:
            print("[ok]   seed=0 rejected (would randomize decoding order)")
            return True
        except Exception as e:  # got past the seed guard, failed elsewhere
            print(f"[FAIL] seed=0 did NOT raise ValueError first: {e!r}")
            return False
    print("[FAIL] seed=0 was accepted — determinism guarantee is broken")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="ProteinMPNN scorer determinism gate")
    ap.add_argument("--pdb", default=DEFAULT_PDB, help="single-chain PDB to score")
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--num-decoding-orders", type=int, default=8)
    ap.add_argument("--spread-seeds", default="37,101,271",
                    help="comma-separated seeds for the cross-seed spread report")
    args = ap.parse_args()

    ok_guard = _preflight_seed_zero()

    if not PROTEINMPNN_PATH:
        sys.exit("PROTEINMPNN_PATH not set — set it in .env to run the real determinism gate.")
    if not os.path.exists(args.pdb):
        sys.exit(f"PDB not found: {args.pdb}")

    pdb = open(args.pdb).read()

    # Gate 1 — same-seed byte determinism.
    a = _logp(pdb, args.seed, args.num_decoding_orders)
    b = _logp(pdb, args.seed, args.num_decoding_orders)
    identical = a.shape == b.shape and np.array_equal(a, b)
    max_abs = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")
    print(f"\n=== same-seed determinism (seed={args.seed}, "
          f"N={args.num_decoding_orders}) ===")
    print(f"  log_p shapes: {a.shape} vs {b.shape}   max|Δ| = {max_abs:.3e}")
    print("  --> " + ("PASS (byte-identical)"
                       if identical else "FAIL (non-deterministic at fixed seed)"))

    # Report 2 — cross-seed spread (residual decoding-order noise; not a gate).
    seeds = [int(s) for s in args.spread_seeds.split(",")]
    stack = np.stack([_logp(pdb, s, args.num_decoding_orders) for s in seeds])  # [S, L, 21]
    per_res_std = stack.std(axis=0).mean(axis=-1)  # mean over aa of cross-seed std, per residue
    print(f"\n=== cross-seed spread over seeds {seeds} ===")
    print(f"  mean per-residue std = {per_res_std.mean():.3e}   "
          f"max = {per_res_std.max():.3e}")
    print("  (lower is better; raise --num-decoding-orders to shrink it)")

    sys.exit(0 if (identical and ok_guard) else 1)


if __name__ == "__main__":
    main()