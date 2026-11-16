"""
Print pyproject.toml's dependencies (core + the named extras) as a pip requirements file.

The Docker images install from this so dependencies are declared once, in pyproject.toml,
while the heavy pip layer still caches on pyproject.toml alone (the source is copied in
afterwards). Usage, inside the image build:

    python pyproject_deps.py pyproject.toml ml modal tracking dev > /tmp/requirements.txt
    pip install -r /tmp/requirements.txt

Only LEAF extras are accepted. An extra that references this project itself (e.g.
`gpu = ["propredict[ml]", ...]`, `all = ["propredict[...]"]`) is rejected rather than
emitted: pip would treat `propredict[ml]` as a package to fetch from PyPI, which at best
fails and at worst installs an unrelated package that happens to own the name.

Stdlib-only on purpose (tomllib, 3.11+; tomli fallback for the 3.10 dev env), so it runs in
the build stage before any dependency is installed. See Process/pyproject-adoption.md.
"""

import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 — local dev env only; images run 3.11
    import tomli as tomllib

_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize(name: str) -> str:
    """PEP 503 name normalisation."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements(pyproject_path: str, extras: list) -> list:
    """Core dependencies followed by each named extra, de-duplicated, in declaration order."""
    with open(pyproject_path, "rb") as fh:
        project = tomllib.load(fh)["project"]

    own_name = _normalize(project["name"])
    optional = project.get("optional-dependencies", {})

    unknown = [e for e in extras if e not in optional]
    if unknown:
        raise ValueError(f"unknown extra(s) {unknown}; defined: {sorted(optional)}")

    out = []
    for source, reqs in [("dependencies", project.get("dependencies", []))] + [(f"[{e}]", optional[e]) for e in extras]:
        for req in reqs:
            match = _NAME.match(req)
            if not match:
                raise ValueError(f"unparseable requirement {req!r} in {source}")
            if _normalize(match.group(1)) == own_name:
                raise ValueError(
                    f"{source} references the project itself ({req!r}); pass the leaf extras it "
                    "expands to instead — pip would otherwise look this name up on PyPI"
                )
            if req not in out:
                out.append(req)
    return out


def main(argv: list) -> int:
    if len(argv) < 2:
        print(f"usage: {argv[0]} PYPROJECT [EXTRA ...]", file=sys.stderr)
        return 2
    try:
        reqs = requirements(argv[1], argv[2:])
    except (OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("\n".join(reqs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
