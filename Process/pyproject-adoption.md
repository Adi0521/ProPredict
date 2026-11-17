# Adopting pyproject.toml as the install source

**Status:** Steps 1–2 of 4 complete (Docker, Modal). Follows Process/pyproject.md, which created
`pyproject.toml` but left every build installing from `requirements*.txt`.

## Plan

| Step | Consumer | Change | Status |
|---|---|---|---|
| 1 | `Dockerfile.api`, `Dockerfile.celery` | install from pyproject via `scripts/pyproject_deps.py` | **done** |
| 2 | `modal_app.py` main image | `pip_install_from_pyproject(..., ["ml","modal","tracking","dev"])`; keep cu126 torch + Boltz SHA steps in order; pin cuEquivariance (step 2a) | **done** |
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

## Step 2 — Modal main image

- **`modal_app.py`**: `pip_install_from_requirements("requirements.txt")` →
  `pip_install_from_pyproject("pyproject.toml", optional_dependencies=MODAL_IMAGE_EXTRAS)`, with
  `MODAL_IMAGE_EXTRAS = ["ml", "modal", "tracking", "dev"]` — the same set as Docker. The cu126
  torch step and the Boltz git-SHA step are unchanged and still follow it: Boltz is not taken from
  `[gpu]` because it must install after the CUDA torch.
- **Modal's helper skips unknown extras silently** (`if dep_group_name in optionals`) and pastes
  self-references through, so a typo such as `"trackng"` would just drop wandb from the image.
  `tests/test_pyproject_deps.py` now reads `MODAL_IMAGE_EXTRAS` (via `ast`, no `modal` import)
  and both Dockerfiles' extras, asserts all three equal, and runs them through the extractor,
  which raises on unknown/self-referencing names. Mutation-checked: a typo and an added `gpu`
  each fail the test.
- Modal sorts all deps into one `pip install` line; reproducing its logic locally gives exactly
  the requirements.txt set + `ruff`, no duplicates.

### Step 2a — the Modal image could not be built fresh (pre-existing, fixed)

The first verification builds — of the **old** `requirements.txt` image and the new one, run
side by side — both failed identically at the Boltz weight-bake step:
`RuntimeError: operator torchvision::nms does not exist`.

- **Cause:** `.pip_install("cuequivariance-ops-torch-cu12")` was unpinned. It now resolves to
  0.13.0, which requires `torch>=2.11`, so pip uninstalled `torch==2.6.0+cu126` and installed
  torch 2.15.0 + a CUDA 13 stack, leaving `torchvision 0.21.0+cu126` broken. Every fresh build
  from `main` was broken; the benchmark-record image only existed because Modal had it cached,
  and the 2026-11-15 numpy change to requirements.txt had already invalidated that cache.
- **Version recovery:** an attempt to re-hydrate the benchmark-era image from `cce39f0`
  (requirements.txt before the numpy change) missed Modal's cache and was aborted at the first
  `Building image` line. No version was ever recorded (`Process/proteinmpnn-version-pin.md`
  lists cuEquivariance as unpinned). From PyPI metadata: `cuequivariance-ops-torch-cu12` ≤ 0.10.0
  lists torch only under the `test` extra; 0.11.1+ require `torch>=2.11` (0.11.0 was the
  transition). Run 011 — the rebuild behind the current record — was **2026-07-21**, when 0.10.0
  was the newest release of all three packages (0.11.x: 07-26 for `cuequivariance`/`-torch`,
  08-04 for the ops package). So that image almost certainly had 0.10.0. Inferred from release
  dates, not observed.
- **Fix:** pin `cuequivariance-ops-torch-cu12==0.10.0`, `cuequivariance-torch==0.10.0`,
  `cuequivariance==0.10.0`, each call also carrying `torch==2.6.0` so pip can never swap torch
  again (`==2.6.0` matches the installed `2.6.0+cu126` under PEP 440, so it's a no-op).

### Verification (new image, workspace `kalshi-hft-real`)

| Check | Result |
|---|---|
| image build | green |
| `report_boltz_version` | `b1ebfc46…` (matches pin), weights baked, torch `2.6.0+cu126`, numpy `1.26.4`, scipy `1.13.1` — identical to the Run 011 snapshot in BENCHMARKS.md |
| `report_proteinmpnn_version` | `8907e66`, `PASS: true` |
| `test_ligands_modal` | all ok (`gnina_in_image: False` is expected — GNINA has its own image) |
| `test_membrane_modal` | **fails, pre-existing** — see below |

Full `pip freeze --all` saved as `Process/modal-image-freeze-2026-11-16.txt` (230 packages) — the
first recorded freeze of this image. No old-vs-new freeze diff: the old image wasn't in Modal's
cache, so a "before" would have been a second fresh build testing only pip determinism on an
identical input set, which step 1 already showed for Docker.

**Pre-existing, not fixed here:**
- `test_membrane_modal` returns `charmm36_ff_error: Could not locate file "charmm36/lipids.xml"`
  (through an early `return` that skips its `print`, so `modal run` shows nothing). OpenMM has no
  such file — CHARMM36 lipids are inside `charmm36.xml`. The micromamba layer (OpenMM 8.5.2) was
  served from cache, so this rebuild didn't change it. The same path is in
  `orchestrator/membrane.py:227` and `orchestrator/simulation.py:816`, i.e. the real OpenMM
  membrane path too. Process/mutation-task-5b-membrane-tests.md records only a parse check.
- Boltz's exact pins conflict with what's installed (`protobuf 5.29.6` vs googleapis-common-protos'
  `>=6.33.5`; `click 8.1.7` vs huggingface-hub's `>=8.4.2`). Identical in the old
  requirements.txt build, so unchanged by this step.

Local: `ruff check` / `ruff format --check` clean; `tests/test_pyproject_deps.py` +
`tests/test_boltz.py` 33 passed, 2 skipped; `check_boltz_updates.py` 3 sites agree, no conflict.

Cost note: three failed CPU builds (old + new before the fix, plus one aborted cache probe) and
one successful build.

### Sources (step 2)

- Modal `Image.pip_install_from_pyproject` — installed modal 1.4.2 source.
- PyPI JSON API, release dates + `requires_dist`: https://pypi.org/pypi/cuequivariance-ops-torch-cu12/json
- PEP 440 local version matching: https://peps.python.org/pep-0440/#version-matching
- OpenMM bundled force fields: http://docs.openmm.org/latest/userguide/application/02_running_sims.html#force-fields
