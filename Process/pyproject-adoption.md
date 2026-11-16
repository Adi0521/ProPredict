# Adopting pyproject.toml as the install source

**Status:** Step 1 of 4 complete (Docker). Follows Process/pyproject.md, which created
`pyproject.toml` but left every build installing from `requirements*.txt`.

## Plan

| Step | Consumer | Change | Status |
|---|---|---|---|
| 1 | `Dockerfile.api`, `Dockerfile.celery` | install from pyproject via `scripts/pyproject_deps.py` | **done** |
| 2 | `modal_app.py` main image | `pip_install_from_pyproject(..., ["ml","modal","tracking","dev"])`; keep cu126 torch + Boltz SHA steps in order | pending |
| 3 | `requirements-gpu.txt` | delete; `pip install -e ".[gpu]"` replaces it; `check_boltz_updates.py` + workflow go from 3 pin sites to 2 | pending |
| 4 | `requirements.txt` | delete; update README, CLAUDE.md, pyproject comments | pending |

Later, deliberately separate (each changes what gets installed, so none can be verified as
"freeze unchanged"): drop `[dev]` from runtime images, move `[dev]` to PEP 735
`[dependency-groups]`, generate a transitive lock file (`torch>=2.1.0` and `meeko` float today).

## Two constraints that shaped the design

1. **Self-referencing extras.** `[gpu]` contains `"propredict[ml]"` and `[all]` contains
   `"propredict[...]"`. Anything that pastes an extra's strings into `pip install` makes pip
   look up a package named `propredict` on PyPI: at best a failed build, at worst an unrelated
   package that owns the name. Modal's `pip_install_from_pyproject` does exactly that (read
   from the installed modal 1.4.2 source: it `extend`s the raw strings). So only **leaf**
   extras (`ml`, `modal`, `tracking`, `dev`, `bench`) may be passed, and the extractor
   **rejects** a self-reference rather than emitting it.
2. **Layer caching.** `pip install .` needs the source copied first, so every code edit would
   reinstall torch. Instead only `pyproject.toml` + the extractor are copied before the heavy
   layer; the source arrives in the existing `COPY . .` afterwards.

## Step 1 — Docker

- **`scripts/pyproject_deps.py`** (new): stdlib-only (`tomllib`; `tomli` fallback for the 3.10
  dev env). Prints core deps + named extras as a requirements file, de-duplicated in declaration
  order. Rejects unknown extras and self-references (PEP 503-normalised name match). Exit 1 on
  error, so `> file && pip install` aborts the build.
- **Both Dockerfiles**: `COPY requirements.txt` → `COPY pyproject.toml scripts/pyproject_deps.py
  /tmp/deps/`, then extract `ml modal tracking dev` and `pip install -r` the result. The
  project itself is still **not** pip-installed (it never was): commands run from `/app`, which
  compose bind-mounts.
- **`tests/test_pyproject_deps.py`** (new, 6 tests): ordering/dedup, self-reference rejection
  with name normalisation, unknown extra, exit codes, `[tool.*]` tables not leaking in, and that
  the real pyproject's image extras carry the runtime packages and exclude `[bench]`/Boltz.
- Comments updated in `pyproject.toml` and CLAUDE.md (Docker no longer reads requirements.txt).

### Decisions

- **`[dev]` kept in the images** so this step is behaviour-neutral (requirements.txt shipped
  pytest/httpx too). Dropping it is a separate, later change.
- **Extractor script, not a uv lock file**: no new tool, and the resolved set stays identical.
  A lock file would change the installed versions, so it gets its own step.
- **Only extra package: `ruff`.** `[dev]` includes it and requirements.txt didn't. It's a
  standalone binary nothing imports; accepted rather than splitting `[dev]`.

### Verification

- Extracted set vs `requirements.txt` (comments stripped, sorted): identical except `ruff`.
- **Image freeze diff.** Built all four images with `--no-cache`, concurrently, so both sides
  resolved at the same moment (`before` = `git archive HEAD` context). `pip freeze --all`:
  - api: 199 → 200 packages, the only diff `+ ruff==0.16.10`
  - celery: 199 → 200 packages, the only diff `+ ruff==0.16.10`
- **Smoke**: both new images `import api.main, orchestrator.tasks, torch, transformers`
  (torch 2.14.1); `celery -A orchestrator.tasks` loads in the worker image.
- **Caching**: after a source-only change, the pip layer is `CACHED` and only `COPY . .` reruns.
- **Tests**: `ruff check .` / `ruff format --check .` clean; full local suite (excl. test_api) 215 passed, 3 skipped — 530 s of it is the real ESMFold integration test, which runs because the weights are cached here;
  `tests/test_api.py` 5 passed. That was run against a throwaway `propredict_test` database
  (from a scratch copy of the repo whose `.env` had `DB_NAME` changed), because `config.py`'s
  `load_dotenv(override=True)` ignores an env-var override and the test deletes every `jobs`
  row. The real `propredict` DB kept its 2 rows; the throwaway DB was dropped afterwards.

## Sources

- Modal `Image.pip_install_from_pyproject`: https://modal.com/docs/reference/modal.Image (and the
  installed modal 1.4.2 source)
- `tomllib`: https://docs.python.org/3/library/tomllib.html
- PEP 503 name normalisation: https://peps.python.org/pep-0503/#normalized-names
- Docker build cache: https://docs.docker.com/build/cache/
