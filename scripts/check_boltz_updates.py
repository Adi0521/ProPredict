#!/usr/bin/env python3
"""
Report whether the pinned Boltz-2 build has fallen behind upstream.

Boltz-2 is pinned to an exact commit in two places (modal_app.py and
requirements-gpu.txt) so the benchmark record stays reproducible — see
Process/boltz-version-pin.md for why `boltz==2.2.1` is NOT a safe substitute. The cost of
pinning is that upstream fixes are invisible until someone looks. This script is the look.

It is deliberately read-only: it never edits the pin. Bumping is a decision that should be
paired with a benchmark re-run, not something a script does behind your back.

Runs locally against the GitHub API — no Modal, no GPU, no cost. To check what is actually
installed in the built Modal image (as opposed to what the source declares), use
`modal run modal_app.py::report_boltz_version`.

Usage:
    python scripts/check_boltz_updates.py
    python scripts/check_boltz_updates.py --json

Exit codes:
    0  up to date (or ahead)
    1  behind upstream — updates available
    2  could not determine (network failure, rate limit, or the two pins disagree)
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

REPO = "jwohlwend/boltz"
API = f"https://api.github.com/repos/{REPO}"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the pin lives. Both must agree — a silent disagreement between them means the Modal
# image and a local GPU install would run different Boltz builds.
PIN_SITES = {
    "modal_app.py": os.path.join(_ROOT, "modal_app.py"),
    "requirements-gpu.txt": os.path.join(_ROOT, "requirements-gpu.txt"),
}

# Matches the commit in either "...boltz.git@<sha>" or "...boltz@<sha>" form. Only a real
# 40-hex SHA counts: a branch name or tag here would mean the pin is not actually pinned.
_SHA_RE = re.compile(r"jwohlwend/boltz(?:\.git)?@([0-9a-f]{40})\b")

# Commit subjects worth surfacing loudly — these are the ones that can move numbers, and so
# the ones that make a bump worth a benchmark re-run rather than a quiet upgrade.
_NUMERICS_HINTS = ("precision", "float32", "float16", "autocast", "nan", "inf",
                   "dtype", "device", "seed", "determin", "scale", "sampling")


def _get(url: str):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "propredict-boltz-version-check",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def read_pins() -> dict:
    """Read the pinned SHA from every pin site. Missing file or no match -> None."""
    pins = {}
    for label, path in PIN_SITES.items():
        try:
            with open(path) as fh:
                text = fh.read()
        except OSError:
            pins[label] = None
            continue
        # A file may mention the SHA in prose comments as well as the real directive;
        # every occurrence must agree, so collapsing to a set is the check we want.
        found = set(_SHA_RE.findall(text))
        pins[label] = found.pop() if len(found) == 1 else (None if not found else "CONFLICT")
    return pins


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    pins = read_pins()
    distinct = {v for v in pins.values() if v and v != "CONFLICT"}

    if "CONFLICT" in pins.values() or not distinct:
        print("ERROR: could not read a single unambiguous pinned SHA.", file=sys.stderr)
        for label, sha in pins.items():
            print(f"  {label}: {sha or 'not found'}", file=sys.stderr)
        return 2

    if len(distinct) > 1:
        # The failure this check exists to catch early: Modal and local installs drifting.
        print("ERROR: pin sites disagree — Modal and local installs would differ!", file=sys.stderr)
        for label, sha in pins.items():
            print(f"  {label}: {sha}", file=sys.stderr)
        return 2

    pinned = distinct.pop()

    try:
        head = _get(f"{API}/commits/main")
        latest_sha = head["sha"]
        cmp_data = _get(f"{API}/compare/{pinned}...{latest_sha}")
        try:
            rel = _get(f"{API}/releases/latest")
            latest_release = rel.get("tag_name")
        except urllib.error.HTTPError:
            latest_release = None
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError) as e:
        print(f"ERROR: could not reach the GitHub API: {e}", file=sys.stderr)
        print("(unauthenticated requests are rate-limited to 60/hour)", file=sys.stderr)
        return 2

    behind = cmp_data.get("ahead_by", 0)   # pinned...latest: "ahead" = commits we lack
    commits = cmp_data.get("commits", [])

    def _subject(c):
        return c["commit"]["message"].splitlines()[0]

    notable = [c for c in commits
               if any(h in _subject(c).lower() for h in _NUMERICS_HINTS)]

    if args.json:
        print(json.dumps({
            "pinned": pinned,
            "latest_main": latest_sha,
            "latest_release": latest_release,
            "behind_by": behind,
            "up_to_date": behind == 0,
            "commits": [{"sha": c["sha"][:8], "subject": _subject(c)} for c in commits],
            "notable_numerics_commits": [c["sha"][:8] for c in notable],
        }, indent=2))
        return 0 if behind == 0 else 1

    pinned_date = ""
    try:
        pinned_date = _get(f"{API}/commits/{pinned}")["commit"]["committer"]["date"][:10]
    except Exception:  # noqa: BLE001 — cosmetic only
        pass

    print(f"pinned:  {pinned[:8]}" + (f" ({pinned_date})" if pinned_date else ""))
    print(f"latest:  {latest_sha[:8]} ({head['commit']['committer']['date'][:10]}) on main")
    if latest_release:
        print(f"release: {latest_release}   "
              "(NOTE: the version string can lag main — see Process/boltz-version-pin.md)")

    if behind == 0:
        print("\nUp to date — pin matches upstream main.")
        return 0

    print(f"\nBEHIND BY {behind} commit(s):\n")
    for c in commits:
        mark = " *" if c in notable else "  "
        print(f"{mark} {c['sha'][:8]}  {c['commit']['committer']['date'][:10]}  {_subject(c)[:72]}")

    if notable:
        print(f"\n* {len(notable)} commit(s) look numerics-related — a bump should be treated as")
        print("  a new baseline and re-benchmarked, not compared across the boundary.")

    print("\nTo adopt: update the SHA in BOTH modal_app.py and requirements-gpu.txt,")
    print("then re-run the CASP15 baseline and log it as a new run in benchmarks/BENCHMARKS.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
