#!/usr/bin/env python3
"""
Report whether the pinned Boltz-2 build has fallen behind upstream.

Boltz-2 is pinned to an exact commit in three places (modal_app.py, requirements-gpu.txt and
pyproject.toml) so the benchmark record stays reproducible — see
Process/boltz-version-pin.md for why `boltz==2.2.1` is NOT a safe substitute. The cost of
pinning is that upstream fixes are invisible until someone looks. This script is the look.

Two independent things are checked:

1. **Is the pin behind upstream?** All three sites must agree on one SHA, and that SHA is
   compared against upstream `main`.
2. **Do our exact dependency pins still agree with Boltz's own?** Boltz pins several shared
   packages with hard `==`, so if we pin the same package to a different exact version, the
   `[gpu]` extra and requirements-gpu.txt become *unresolvable* — `pip` fails outright. That
   is a live failure mode, not a theoretical one (Process/pyproject.md: requests 2.31.0 vs
   2.32.3, biopython 1.83 vs 1.84). A bump that moves Boltz's pins re-breaks it even with all
   three SHAs in perfect agreement, which is exactly why this is checked here.

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
    2  could not determine, or the repo is internally inconsistent: network failure, rate
       limit, the pin sites disagree, or our exact pins conflict with Boltz's own
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

# Where the pin lives. All three must agree — a silent disagreement between them means the
# Modal image, a local GPU install and an editable `.[gpu]` install would run different
# Boltz builds. pyproject.toml joined this list in Process/pyproject.md.
PIN_SITES = {
    "modal_app.py": os.path.join(_ROOT, "modal_app.py"),
    "requirements-gpu.txt": os.path.join(_ROOT, "requirements-gpu.txt"),
    "pyproject.toml": os.path.join(_ROOT, "pyproject.toml"),
}

# Our own dependency declarations, for the exact-pin cross-check against Boltz's pyproject.
OUR_PYPROJECT = os.path.join(_ROOT, "pyproject.toml")

# Boltz's own pyproject at an arbitrary commit. raw.githubusercontent needs no auth and is
# not metered against the API rate limit the rest of this script uses.
_BOLTZ_PYPROJECT_URL = f"https://raw.githubusercontent.com/{REPO}/{{sha}}/pyproject.toml"

# Matches the commit in either "...boltz.git@<sha>" or "...boltz@<sha>" form. Only a real
# 40-hex SHA counts: a branch name or tag here would mean the pin is not actually pinned.
_SHA_RE = re.compile(r"jwohlwend/boltz(?:\.git)?@([0-9a-f]{40})\b")

# Commit subjects worth surfacing loudly — these are the ones that can move numbers, and so
# the ones that make a bump worth a benchmark re-run rather than a quiet upgrade.
_NUMERICS_HINTS = (
    "precision",
    "float32",
    "float16",
    "autocast",
    "nan",
    "inf",
    "dtype",
    "device",
    "seed",
    "determin",
    "scale",
    "sampling",
)


# A PEP 508 requirement pinned with a hard `==`, e.g. `uvicorn[standard]==0.24.0`. Only these
# can collide: where either side uses a range (`torch>=2.1.0` vs Boltz's `torch>=2.2`), pip
# picks a version satisfying both and nothing breaks.
_EXACT_PIN_RE = re.compile(
    r"""^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)   # distribution name
         (?:\[[^\]]*\])?                        # optional extras, e.g. [standard]
         \s*==\s*
         (?P<ver>[A-Za-z0-9][A-Za-z0-9.*+!_-]*)$""",
    re.VERBOSE,
)


def _normalize(name: str) -> str:
    """PEP 503 name normalisation, so Bio_Python and biopython compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_requirements(toml_text: str) -> list:
    """
    Quoted requirement strings from a pyproject's [project] tables.

    Deliberately regex, not tomllib: tomllib is 3.11+ and this script must run on 3.10
    (`requires-python = ">=3.10"`), and the CI job installs nothing. Everything from the
    first `[tool.` table onward is dropped so linter config (`select = ["E4", ...]`) and
    package lists cannot be mistaken for dependencies; the remaining non-requirement strings
    (name, description, readme) simply fail to match _EXACT_PIN_RE.
    """
    head = re.split(r"^\[tool\.", toml_text, maxsplit=1, flags=re.MULTILINE)[0]
    return re.findall(r'"([^"\n]+)"', head)


def _exact_pins(requirements) -> dict:
    """{normalized name: version} for every requirement pinned with a hard `==`."""
    pins = {}
    for raw in requirements:
        req = raw.split(";")[0].strip()  # drop environment markers
        m = _EXACT_PIN_RE.match(req)
        if m:
            pins[_normalize(m.group("name"))] = m.group("ver")
    return pins


def conflicting_exact_pins(ours: dict, theirs: dict) -> dict:
    """
    Packages both sides pin exactly, to different versions → guaranteed ResolutionImpossible.

    Only exact-vs-exact is reported. Deciding whether a *range* of ours excludes their pinned
    version would mean reimplementing PEP 440 specifier logic without `packaging`, which isn't
    stdlib; ranges also have never been the failure mode here.
    """
    return {name: (ours[name], theirs[name]) for name in ours.keys() & theirs.keys() if ours[name] != theirs[name]}


def _get(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "propredict-boltz-version-check",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "propredict-boltz-version-check"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def our_exact_pins() -> dict:
    with open(OUR_PYPROJECT) as fh:
        return _exact_pins(_declared_requirements(fh.read()))


def boltz_exact_pins(sha: str) -> dict:
    return _exact_pins(_declared_requirements(_get_text(_BOLTZ_PYPROJECT_URL.format(sha=sha))))


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    behind = cmp_data.get("ahead_by", 0)  # pinned...latest: "ahead" = commits we lack
    commits = cmp_data.get("commits", [])

    def _subject(c):
        return c["commit"]["message"].splitlines()[0]

    notable = [c for c in commits if any(h in _subject(c).lower() for h in _NUMERICS_HINTS)]

    # Exact-pin cross-check. `dep_check` is "unavailable" rather than fatal if Boltz's
    # pyproject can't be fetched: a conflict is a property of files in this repo, so it only
    # appears when someone edits them, and the next run will catch it. Going red on a
    # transient raw.githubusercontent hiccup would train people to ignore this job.
    ours = our_exact_pins()
    conflicts_now, conflicts_if_bumped, dep_check = {}, {}, "ok"
    try:
        conflicts_now = conflicting_exact_pins(ours, boltz_exact_pins(pinned))
        if behind:
            conflicts_if_bumped = conflicting_exact_pins(ours, boltz_exact_pins(latest_sha))
    except (urllib.error.URLError, urllib.error.HTTPError, UnicodeDecodeError, TimeoutError) as e:
        dep_check = "unavailable"
        print(f"WARNING: could not fetch Boltz's pyproject.toml for the pin cross-check: {e}", file=sys.stderr)
    else:
        dep_check = "conflict" if conflicts_now else "ok"

    if args.json:
        print(
            json.dumps(
                {
                    "pinned": pinned,
                    "latest_main": latest_sha,
                    "latest_release": latest_release,
                    "behind_by": behind,
                    "up_to_date": behind == 0,
                    "commits": [{"sha": c["sha"][:8], "subject": _subject(c)} for c in commits],
                    "notable_numerics_commits": [c["sha"][:8] for c in notable],
                    "dep_check": dep_check,
                    "dep_conflicts_at_pinned": {k: {"ours": o, "boltz": b} for k, (o, b) in conflicts_now.items()},
                    "dep_conflicts_if_bumped": {
                        k: {"ours": o, "boltz": b} for k, (o, b) in conflicts_if_bumped.items()
                    },
                },
                indent=2,
            )
        )
        # A dep conflict outranks being behind: installs are broken *now*, which is worse
        # news than an available upgrade.
        if conflicts_now:
            return 2
        return 0 if behind == 0 else 1

    pinned_date = ""
    try:
        pinned_date = _get(f"{API}/commits/{pinned}")["commit"]["committer"]["date"][:10]
    except Exception:  # noqa: BLE001 — cosmetic only
        pass

    print(f"pinned:  {pinned[:8]}" + (f" ({pinned_date})" if pinned_date else ""))
    print(f"latest:  {latest_sha[:8]} ({head['commit']['committer']['date'][:10]}) on main")
    if latest_release:
        print(f"release: {latest_release}   (NOTE: the version string can lag main — see Process/boltz-version-pin.md)")

    if conflicts_now:
        print("\nERROR: our exact pins disagree with Boltz's own at the PINNED commit.")
        print('       `pip install ".[gpu]"` and `pip install -r requirements-gpu.txt` cannot resolve:')
        for name, (mine, theirs) in sorted(conflicts_now.items()):
            print(f"         {name:12} ours =={mine}   boltz =={theirs}")
        print("       Fix: move ours to Boltz's version in pyproject.toml AND requirements.txt.")
    elif dep_check == "ok":
        print(f"deps:    {len(ours)} exact pin(s) checked, none conflict with Boltz's own")

    if behind == 0:
        print("\nUp to date — pin matches upstream main.")
        return 2 if conflicts_now else 0

    print(f"\nBEHIND BY {behind} commit(s):\n")
    for c in commits:
        mark = " *" if c in notable else "  "
        print(f"{mark} {c['sha'][:8]}  {c['commit']['committer']['date'][:10]}  {_subject(c)[:72]}")

    if notable:
        print(f"\n* {len(notable)} commit(s) look numerics-related — a bump should be treated as")
        print("  a new baseline and re-benchmarked, not compared across the boundary.")

    if conflicts_if_bumped:
        print("\nAdopting this bump would ALSO require moving these pins (Boltz changed them):")
        for name, (mine, theirs) in sorted(conflicts_if_bumped.items()):
            print(f"  {name:12} ours =={mine}   boltz(latest) =={theirs}")
        print("  Leaving them as-is makes `[gpu]` and requirements-gpu.txt unresolvable.")

    print("\nTo adopt: update the SHA in ALL THREE of modal_app.py, requirements-gpu.txt and")
    print("pyproject.toml, then re-run the CASP15 baseline and log it as a new run in")
    print("benchmarks/BENCHMARKS.md.")
    return 2 if conflicts_now else 1


if __name__ == "__main__":
    sys.exit(main())
