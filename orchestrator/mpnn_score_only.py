"""In-process ProteinMPNN `score_only` with SHARED decoding orders — scoring protocol v2.

Why v2 exists
-------------
The upstream CLI (`protein_mpnn_run.py`) seeds torch ONCE per process (`:29`) and then draws
the decoding order `randn_1 = torch.randn(chain_M.shape)` sequentially inside its loop over
fasta sequences (`:256`). With `--batch_size 1`, fasta sequence k therefore consumes global
stream draws `[K*k .. K*k+K-1]`. Consequences, both measured in
`benchmarks/diagnose_score_only_batching.py`:

  * identical fasta content AND order -> bit-identical scores (exact reproducibility holds);
  * the same sequence at a different index -> different decoding orders -> different score.

That second property is poison for the WT/single/double epistasis contrast
`eps = f(AB) - f(A) - f(B) + f(WT)`: its four terms sit at four different stream offsets and
so carry four INDEPENDENT order perturbations, while the eps signal is ~3x smaller than the
between-sequence score spread. Measured on 4G3O (47 aa, 30 doubles, K=8): reordering the
batch left raw-score Spearman at 0.978 but dropped eps Spearman to 0.273.

What v2 changes
---------------
One model load; the backbone featurized once; K decoding-order tensors drawn ONCE from a
dedicated `torch.Generator` and REUSED for every sequence in the call. Two consequences:

  * **Batch-composition invariance, by construction.** A sequence's score depends only on
    (backbone, sequence, model, K, seed) — not on its index, its neighbours, or how the
    batch was split. Reordering, inserting unrelated sequences, chunking and
    one-call-per-sequence all return identical numbers.
  * **Common random numbers across the cycle.** Because WT, singles and double are scored
    against the SAME orders, the order-dependent component is common to all four terms and
    largely cancels in the contrast. Measured on the same panel at K=8, cross-seed eps
    reliability rose 0.533 -> 0.977 and eps noise SD fell 0.0159 -> 0.0024.

The estimand is UNCHANGED. Each score is still a mean over K decoding orders drawn from the
same distribution, so its expectation over draws is what v1 estimated; only the correlation
structure of the errors differs. The diagnostic confirms this empirically: the implied
signal SD (sqrt(observed^2 - noise^2)) agrees across protocols, 0.0170 (v1) vs 0.0154 (v2) —
v2 removes noise rather than manufacturing signal.

What v2 does NOT change: this is a numerical-consistency fix. It makes candidate comparisons
trustworthy and stops the search selecting noisy winners. It says nothing about whether
ProteinMPNN's compatibility score tracks stability, function or fitness — see
`Process/rowB-step2-stability-arm.md`, where the answer is "weakly at best".

Shared draws are NOT interchangeable with per-sequence seeding. Deriving a stable seed per
sequence would also give batch invariance, but would leave the four errors in the contrast
independent — it buys v1's reliability (0.533), not v2's (0.977).

Fidelity to upstream
--------------------
Model construction, featurization and the scoring reduction mirror `protein_mpnn_run.py`
exactly (hidden_dim=128, num_layers=3, num_letters=21, `augment_eps=backbone_noise` which
defaults to 0.0, `k_neighbors=checkpoint["num_edges"]`, `_scores(S, log_probs, mask)` for
`global_score`). Because this reaches into upstream internals rather than the
CLI, it is pinned to the same clone commit as everything else (8907e66).

v2 is NOT expected to equal v1 numerically on the same inputs, and does not: v1 burns K
draws on its native-PDB pass before the first fasta sequence, so even a one-sequence batch
gets different orders (measured max |v1-v2| = 0.030 at K=4, the scale of order noise). The
fidelity check is therefore conditional on the draws: feed `_score_with_orders` the exact
order tensors upstream would have used, and it must reproduce v1's `global_score` to float
precision. `tests/test_mpnn_score_only.py` does that against the real model, which is what
catches drift in model construction, featurization or the scoring reduction.

Decoding order is the only stochastic input: `augment_eps=0` means no backbone jitter, and
scoring runs one structure at a time so there is no batch-dim reduction nondeterminism.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCORE_PROTOCOL_V1 = "score_only/v1"  # upstream CLI; one global RNG stream per process
SCORE_PROTOCOL_V2 = "score_only/v2"  # in-process, K decoding orders shared across the batch

# ProteinMPNN's own alphabet ordering (protein_mpnn_utils.py:50) — index into S / log_p.
_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# One loaded model per (clone, weights, device) within a process. Loading costs ~1-2 s on
# CPU, which would otherwise be paid once per call and dominate a small batch.
_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _import_upstream(proteinmpnn_dir: str):
    """Import ProteinMPNN's modules from the clone at `proteinmpnn_dir`.

    The clone is not an installable package, so its directory goes on sys.path. Imported
    lazily (never at module import) so this module stays importable without torch, per the
    optional-dependency convention."""
    import sys

    if not os.path.isdir(proteinmpnn_dir):
        raise RuntimeError(
            f"PROTEINMPNN_PATH {proteinmpnn_dir!r} is not a directory. Point it at a clone of "
            "https://github.com/dauparas/ProteinMPNN."
        )
    run_script = os.path.join(proteinmpnn_dir, "protein_mpnn_run.py")
    if not os.path.isfile(run_script):
        raise RuntimeError(
            f"protein_mpnn_run.py not found at {run_script}. Check PROTEINMPNN_PATH points at a "
            "clone of https://github.com/dauparas/ProteinMPNN."
        )
    real = os.path.realpath(proteinmpnn_dir)
    if real not in sys.path:
        sys.path.insert(0, real)
    from protein_mpnn_utils import (  # noqa: PLC0415 — deliberate lazy third-party import
        ProteinMPNN,
        StructureDatasetPDB,
        _scores,
        parse_PDB,
        tied_featurize,
    )

    return {
        "ProteinMPNN": ProteinMPNN,
        "StructureDatasetPDB": StructureDatasetPDB,
        "_scores": _scores,
        "parse_PDB": parse_PDB,
        "tied_featurize": tied_featurize,
    }


def _load_model(proteinmpnn_dir: str, model_name: str, device: str, up: Dict[str, Any]):
    """Load (and cache) the ProteinMPNN checkpoint, constructed exactly as
    `protein_mpnn_run.py:180` does."""
    key = (os.path.realpath(proteinmpnn_dir), model_name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch

    ckpt_path = os.path.join(proteinmpnn_dir, "vanilla_model_weights", f"{model_name}.pt")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"ProteinMPNN weights not found at {ckpt_path} (model_name={model_name!r}).")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_dim, num_layers = 128, 3  # protein_mpnn_run.py:34-35
    model = up["ProteinMPNN"](
        ca_only=False,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=0.0,  # == the CLI's --backbone_noise default: no coordinate jitter
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _MODEL_CACHE[key] = model
    logger.debug("loaded ProteinMPNN %s (num_edges=%s) on %s", model_name, checkpoint["num_edges"], device)
    return model


def prepare(
    pdb_string: str,
    proteinmpnn_dir: str,
    model_name: str = "v_48_020",
    device: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str]:
    """Load the model and featurize the backbone once.

    Returns `(model, feats, upstream, device)`, where `feats` holds the tensors the scoring
    loop needs. Split out from `score_only_shared_orders` so a fidelity test can drive the
    real scoring path with externally supplied decoding orders."""
    import copy as _copy

    import torch

    device = device or "cpu"
    up = _import_upstream(proteinmpnn_dir)
    model = _load_model(proteinmpnn_dir, model_name, device, up)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "structure.pdb")
        with open(pdb_path, "w") as fh:
            fh.write(pdb_string)
        pdb_dict_list = up["parse_PDB"](pdb_path, ca_only=False)
        dataset = up["StructureDatasetPDB"](pdb_dict_list, truncate=None, max_length=200000)

    if not len(dataset):
        raise RuntimeError("ProteinMPNN parsed no usable chain from the supplied PDB string.")

    all_chains = [k[-1:] for k in pdb_dict_list[0] if k[:9] == "seq_chain"]
    chain_id_dict = {pdb_dict_list[0]["name"]: (all_chains, [])}
    batch_clones = [_copy.deepcopy(dataset[0])]

    with torch.no_grad():
        f = up["tied_featurize"](batch_clones, device, chain_id_dict, None, None, None, None, None, ca_only=False)
    feats = {
        "X": f[0],
        "S": f[1],
        "mask": f[2],
        "chain_M": f[4],
        "chain_encoding_all": f[5],
        "chain_M_pos": f[10],
        "residue_idx": f[12],
    }
    return model, feats, up, device


def _score_with_orders(
    model: Any,
    feats: Dict[str, Any],
    sequences: List[str],
    orders: List[Any],
    up: Dict[str, Any],
    device: str,
) -> List[float]:
    """Score each sequence against EVERY tensor in `orders`, returning the per-sequence mean
    `global_score`. The caller owns the draws, which is what makes the shared-draw property
    explicit and the fidelity test possible."""
    import torch

    X, S, mask = feats["X"], feats["S"], feats["mask"]
    chain_M, chain_M_pos = feats["chain_M"], feats["chain_M_pos"]
    residue_idx, chain_encoding_all = feats["residue_idx"], feats["chain_encoding_all"]

    structure_len = int(S.shape[1])
    out: List[float] = []
    with torch.no_grad():
        for seq in sequences:
            if len(seq) != structure_len:
                # Upstream threads into S[:, :len(seq)] and leaves the tail alone, which
                # would silently score a chimera. Refuse instead.
                raise ValueError(
                    f"sequence length {len(seq)} != structure length {structure_len}. score_only "
                    "threads each sequence onto the backbone, so a mismatch would score a chimera."
                )
            s_input = torch.tensor([_ALPHABET.index(aa) for aa in seq], device=device)[None, :].repeat(X.shape[0], 1)
            s_threaded = S.clone()  # clone per sequence: never carry a previous tail over
            s_threaded[:, : len(seq)] = s_input
            total = 0.0
            for randn_1 in orders:
                log_probs = model(X, s_threaded, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                total += float(up["_scores"](s_threaded, log_probs, mask).mean().item())
            out.append(total / len(orders))
    return out


def score_only_shared_orders(
    pdb_string: str,
    sequences: List[str],
    proteinmpnn_dir: str,
    model_name: str = "v_48_020",
    seed: int = 37,
    num_decoding_orders: int = 8,
    device: Optional[str] = None,
) -> List[float]:
    """Return the per-sequence mean `global_score` (mean NLL, LOWER = better fit), aligned to
    input order, using K decoding orders shared across every sequence in the call.

    Invariant: the returned value for a sequence depends only on
    `(pdb_string, sequence, model_name, seed, num_decoding_orders, device)`. It does not
    depend on the other sequences in `sequences`, on their order, or on how the work was
    split across calls. `score_only_shared_orders(pdb, [a, b])` and
    `[score_only_shared_orders(pdb, [a])[0], score_only_shared_orders(pdb, [b])[0]]` are
    equal.

    `seed` must be non-zero, matching the CLI guard: ProteinMPNN's `if args.seed:` treats 0
    as unset. Nothing here would randomize on 0 (we seed an explicit Generator), but
    accepting 0 would make v2 and v1 disagree on what the seed means.
    """
    if seed == 0:
        raise ValueError(
            "seed=0 is rejected for protocol parity: ProteinMPNN's CLI `if args.seed:` treats 0 "
            "as unset and randomizes. Use any non-zero int."
        )
    if num_decoding_orders < 1:
        raise ValueError(f"num_decoding_orders must be >= 1, got {num_decoding_orders}")
    if not sequences:
        return []
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(f"score_only_shared_orders requires equal-length sequences (got lengths {sorted(lengths)}).")
    bad = sorted({c for s in sequences for c in s} - set(_ALPHABET))
    if bad:
        raise ValueError(f"sequences contain characters outside ProteinMPNN's alphabet: {bad}")

    import torch

    model, feats, up, device = prepare(pdb_string, proteinmpnn_dir, model_name, device)

    # THE FIX: draw the K decoding orders once, from a dedicated Generator, and reuse them
    # for every sequence. A private Generator (rather than torch.manual_seed) keeps this
    # independent of any global torch RNG state the caller may rely on, and keeps the draws
    # a function of `seed` alone rather than of how much randomness ran before.
    gen = torch.Generator(device=device).manual_seed(int(seed))
    orders = [torch.randn(feats["chain_M"].shape, generator=gen, device=device) for _ in range(num_decoding_orders)]

    return _score_with_orders(model, feats, sequences, orders, up, device)
