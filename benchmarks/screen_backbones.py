"""Screen candidate backbones against Row C's eligibility rule.

`research_plan/rowC-reliability-generalization.md` pre-registers an explicit list of structures
and states that every entry was checked against the actual PDB file rather than asserted from
memory. This is that check, kept in the repo so the claim is reproducible — it is the provenance
for a pre-registered list, not a one-off.

Reports per code: experimental method, resolution, model count, protein-chain count, the chain
screened, how many gap-free polypeptide fragments that chain breaks into, its residue count,
the longest fragment's length, and any nonstandard residues. Read-only: fetches into the
existing structure cache and writes nothing else.

The `ok` column implements the **primary (X-ray) stratum** rule only: single gap-free chain of
standard residues, X-ray, resolution <= 2.7 A. A `no` for `method=SOLUTION NMR` therefore means
"not in the primary stratum", NOT "excluded from the study" — Row C declares the 32 NMR domains
as a separate secondary stratum, analysed beside the X-ray arm and never pooled with it. Only
2O2W (powder diffraction) is excluded outright.

Why the checks matter — each one rejected a real candidate:
  * fragment count: 1EMA (GFP) looks like a 225-residue barrel and is 6 fragments, longest 64.
  * resolution:     1STP 2.60 A, 1HTI 2.80 A.
  * chain id:       8TLN's protein chain is `E`, and it is 2 fragments.
  * length:         1GPB is 823 residues, far past the size strata.

Run:
    python -m benchmarks.screen_backbones --set tsuboyama
    python -m benchmarks.screen_backbones --set rowc
    python -m benchmarks.screen_backbones 1STN 1MBO 3GRS
"""

import argparse
import io
import os
import sys
import warnings
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
MAX_RESOLUTION = 2.7

# The 45 Tsuboyama-2023 assays with >=300 usable doubles: Study 2's eligibility pool.
TSUBOYAMA = """1E0L 2JVD 2WQG 1K1V 1PV0 1W4G 4G3O 2N88 2K28 2MYX 2KRU 2KWH 2KVT 2L7M 2MXD
2MI6 1LP1 2LHR 2LCL 5UBS 1I2T 1GYZ 5JRT 1TUD 2BTT 2L9R 1R69 1UZC 1ORC 1TG0 2LO1 1V1C 1YU5
2M0Y 6ACV 1F0M 2O2W 1PSE 3L1X 1WCL 1UFM 2D1U 1O6X 2AMI 3DKM""".split()

# Study 1's pre-registered 16, in stratum order.
ROWC = [
    "4G3O",
    "1LP1",
    "1TUD",
    "1I2T",
    "1R69",
    "1TG0",
    "1F0M",
    "3DKM",
    "1STN",
    "1MBO",
    "1PPN",
    "2PTN",
    "1BTL",
    "5CPA",
    "1AJ8",
    "3GRS",
]


def header_facts(raw: str):
    """Method and resolution from the header records, stopping at the first ATOM."""
    method: Optional[str] = None
    resolution: Optional[float] = None
    for line in raw.splitlines():
        if line.startswith("ATOM"):
            break
        if line.startswith("EXPDTA") and method is None:
            method = line[10:].strip()
        elif line.startswith("REMARK   2 RESOLUTION."):
            tok = line.split("RESOLUTION.")[1].replace("ANGSTROMS.", "").strip()
            try:
                resolution = float(tok)
            except ValueError:
                resolution = None
    return method, resolution


def screen(code: str) -> Dict:
    from Bio.PDB import PDBParser, PPBuilder
    from Bio.PDB.Polypeptide import is_aa

    from benchmarks.benchmark_score_only_epistasis import _fetch_pdb

    try:
        raw = _fetch_pdb(code)
    except Exception as exc:  # noqa: BLE001 - any fetch failure is just "unavailable"
        return {"code": code, "error": f"fetch failed: {type(exc).__name__}"}
    method, resolution = header_facts(raw)
    try:
        struct = PDBParser(QUIET=True).get_structure(code, io.StringIO(raw))
    except Exception as exc:  # noqa: BLE001
        return {"code": code, "error": f"parse failed: {type(exc).__name__}", "method": method}

    model = list(struct)[0]
    chains = [c for c in model if sum(1 for r in c if is_aa(r, standard=False)) >= 20]
    if not chains:
        return {"code": code, "error": "no protein chain >= 20 aa", "method": method}
    chain = chains[0]
    peptides = PPBuilder().build_peptides(chain)
    if not peptides:
        return {"code": code, "error": "no polypeptide built", "method": method}
    longest = max(peptides, key=len)
    seq = str(longest.get_sequence())

    rec = {
        "code": code,
        "method": method or "?",
        "resolution": resolution,
        "n_models": len(list(struct)),
        "n_chains": len(chains),
        "chain_id": chain.id,
        "n_fragments": len(peptides),
        "chain_aa": sum(1 for r in chain if is_aa(r, standard=False)),
        "longest_frag": len(longest),
        "nonstandard": sorted(set(seq) - STANDARD),
        # The longest fragment's one-letter sequence. For an eligible backbone n_fragments == 1,
        # so this IS the whole chain -- `benchmark_score_reliability.py` builds its mutant panel
        # from it rather than re-parsing the file.
        "sequence": seq,
    }
    reasons: List[str] = []
    if "X-RAY" not in rec["method"].upper():
        reasons.append(f"method={rec['method']}")
    elif resolution is None:
        reasons.append("no resolution in header")
    elif resolution > MAX_RESOLUTION:
        reasons.append(f"resolution {resolution:.2f} > {MAX_RESOLUTION}")
    if rec["n_fragments"] != 1:
        reasons.append(f"{rec['n_fragments']} fragments (longest {rec['longest_frag']}/{rec['chain_aa']})")
    if rec["nonstandard"]:
        reasons.append(f"nonstandard {','.join(rec['nonstandard'])}")
    rec["eligible"] = not reasons
    rec["reasons"] = reasons
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Screen backbones against Row C's eligibility rule")
    ap.add_argument("codes", nargs="*", help="PDB codes to screen")
    ap.add_argument("--set", choices=["tsuboyama", "rowc"], help="a named candidate set")
    args = ap.parse_args(argv)

    codes = list(args.codes)
    if args.set == "tsuboyama":
        codes = TSUBOYAMA
    elif args.set == "rowc":
        codes = ROWC
    if not codes:
        ap.error("give PDB codes or --set")

    hdr = (
        f"{'pdb':5} {'method':22} {'res':>5} {'mdl':>3} {'chn':>3} {'id':>2} "
        f"{'frag':>4} {'chainAA':>7} {'longest':>7} {'ok':>3}  why not"
    )
    print(hdr)
    print("-" * len(hdr))
    eligible = []
    for code in codes:
        r = screen(code)
        if "error" in r:
            print(
                f"{code:5} {'ERROR':22} {'':>5} {'':>3} {'':>3} {'':>2} {'':>4} {'':>7} {'':>7} {'no':>3}  {r['error']}"
            )
            continue
        res = f"{r['resolution']:.2f}" if r["resolution"] else "n/a"
        ok = "yes" if r["eligible"] else "no"
        if r["eligible"]:
            eligible.append(code)
        print(
            f"{r['code']:5} {r['method'][:22]:22} {res:>5} {r['n_models']:>3} {r['n_chains']:>3} "
            f"{r['chain_id']:>2} {r['n_fragments']:>4} {r['chain_aa']:>7} {r['longest_frag']:>7} "
            f"{ok:>3}  {'; '.join(r['reasons'])}"
        )
    print(f"\n{len(eligible)}/{len(codes)} eligible: {' '.join(eligible)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
