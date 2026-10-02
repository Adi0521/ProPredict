# pyproject.toml + ruff

**Date:** 2026-10-02
**Status:** Complete

## What changed

- **`pyproject.toml`** (new) — project metadata, dependencies, setuptools package list, pytest
  config, and ruff config.
- **`pytest.ini`** (deleted) — its single `addopts = --continue-on-collection-errors` moved to
  `[tool.pytest.ini_options]`. Behaviour unchanged.
- `requirements.txt` / `requirements-gpu.txt` / `environment-conda.yml` — **untouched**.

## Dependency layout

| Group | Contents | Why separate |
|---|---|---|
| core | FastAPI, Celery, SQLAlchemy, Pydantic, BioPython, PropKa, anthropic, … | needed by API + worker |
| `[ml]` | torch, transformers, meeko | heavy; ESMFold-local / ligand prep only |
| `[modal]` | modal | only for GPU-cloud mode (`MODAL_ENABLED`) |
| `[tracking]` | wandb | benchmarks only |
| `[gpu]` | `[ml]` + Boltz pinned to `b1ebfc46` | same exact-commit pin as `requirements-gpu.txt` / `modal_app.py` (Process/boltz-version-pin.md) |
| `[dev]` | pytest, pytest-asyncio, httpx, ruff | test/lint only — previously mixed into runtime `requirements.txt` |
| `[all]` | ml + modal + tracking + dev | equivalent of today's `requirements.txt` (plus ruff) |

Typical installs: `pip install -e ".[all]"` (local dev), `pip install -e ".[gpu,dev]"` (GPU box).

## Decisions

- **requirements*.txt kept as-is for now.** Both Dockerfiles and `modal_app.py`
  (`pip_install_from_requirements`) still install from them, so this step changes no image
  builds. Cost: dependencies are listed twice until those migrate — a comment at the top of
  `dependencies` flags this. Migrating Docker (`pip install .`) and Modal
  (`Image.pip_install_from_pyproject`) is the natural follow-up.
- **Conda-only tools stay in `environment-conda.yml`.** OpenMM, RDKit, OpenFF, Vina, PyRosetta,
  MDAnalysis lack reliable pip wheels on ARM64 (see Process/4.1-requirements-split.md); they're
  feature-flag gated anyway.
- **`requires-python = ">=3.10"`** — local conda env is 3.10.19; Docker/Modal use 3.11.
- **Explicit setuptools package list** (`api`, `models`, `orchestrator`, `orchestrator.backends`,
  `benchmarks` + `config` module) instead of auto-discovery, which would trip over the many
  top-level dirs (`results/`, `wandb/`, `scripts/`, …). Verified by building a wheel.
- **Boltz git URL in an extra** is fine for local/editable installs; PyPI would reject it, but
  this project isn't published.

## Ruff

- Rules: `select = ["E4", "E7", "E9", "F", "I", "B"]` — classic core set + isort + bugbear.
- **Uses `select`, not `extend-select`.** Ruff 0.16 widened its built-in default rule set
  (with `extend-select` the first run reported 597 findings across ~40 rule families, incl.
  `S`, `SIM`, `RUF`, `PLR`). An explicit list keeps the rule set stable across ruff versions;
  ruff is additionally pinned `>=0.16,<0.17` in `[dev]`.
- `line-length = 120` (only 10 tracked lines exceed it; E501 isn't selected anyway, it just
  drives `ruff format`). `target-version = "py310"`.
- Excludes tracked data/output dirs (`results`, `research_plan`, `mutation-plans`,
  `benchmarks/proteingym_cache`); gitignored paths are skipped automatically.

**Baseline (not fixed in this step):** `ruff check .` → 159 findings, 76 auto-fixable — mostly
I001 unsorted imports (52), B904 raise-without-from (21), F401 unused imports (18), E402 (17),
B905 zip-without-strict (15). `ruff format --check .` → 46 of 94 files would be reformatted.
Cleaning these up is a separate step.

## Verification

- `validate-pyproject pyproject.toml` → valid
- `pip wheel . --no-deps` → builds; wheel contains all five packages + `config.py`
- `pytest tests/test_boltz.py` → 24 passed, 2 skipped (pytest config picked up from pyproject)

## Sources

- Ruff configuration: https://docs.astral.sh/ruff/configuration/
- Ruff rule selection (`select` vs `extend-select`): https://docs.astral.sh/ruff/linter/#rule-selection
- pyproject metadata spec: https://packaging.python.org/en/latest/specifications/pyproject-toml/
- setuptools pyproject config: https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
- Modal `pip_install_from_pyproject`: https://modal.com/docs/reference/modal.Image

---

# Follow-up: `ruff check --fix` (safe fixes only)

**Date:** 2026-10-02

## What changed

- Ran `ruff check --fix .` — **safe fixes only** (no `--unsafe-fixes`). 56 fixes across 29 files:
  import sorting (I001), unused imports removed (F401), placeholder-less f-strings (F541),
  duplicate imports (F811), one-import-per-line (E401). No logic changes.
- Added `tests/Colab-Prototypes` to ruff's `extend-exclude`: it holds the upstream ColabFold
  `AlphaFold2.ipynb` kept as a reference prototype, and ruff would otherwise rewrite the
  notebook JSON for no benefit.

## Checks before applying

- **Removed imports aren't re-export / patch targets.** `git grep` for `orchestrator.tasks.count_clashes`,
  `orchestrator.tasks.run_gromacs_em`, `api.main.SessionLocal`, etc. found no `mock.patch` targets
  or `from X import` consumers relying on them.
- **Optional-dep availability probes untouched.** Ruff deliberately does not auto-remove unused
  imports inside `try/except ImportError` (`rdkit.Chem.AllChem` in ligands.py, `openmm.app.Modeller`
  in membrane.py); those remain as reported F401s.
- **Import reordering is semantics-safe here.** isort only reorders within a contiguous import
  block, so `load_dotenv()` / `sys.path` statements still run in the same place. A module import
  smoke test (config, models, all orchestrator modules) passed — notably `orchestrator.agent` now
  imports before the backends in `tasks.py` with no circular-import issue.

## Remaining (60, need manual judgement — not auto-fixable)

B904 raise-without-from (21), B905 zip-without-strict (14), E741 ambiguous names (6),
F841 unused variables (6), B008 call-in-default-arg (5, mostly FastAPI `Depends()` — idiomatic,
likely to be ignored per-file), E402 (4), F401 availability probes (3), B023 (1).

---

# Follow-up: `ruff format`

**Date:** 2026-10-02

- Ran `ruff format .` (config from `[tool.ruff]`: line-length 120, py310). All 94 linted files now
  pass `ruff format --check`.
- Ruff's formatter verifies the AST is unchanged after formatting, so this is layout-only.
  Spot-checked anyway: module import smoke test OK, `tests/test_boltz.py` + `tests/test_orchestrator.py`
  → 67 passed, 2 skipped.
- Lint findings unchanged at 60 (formatting doesn't affect them).
- Once committed, add the formatting commit hash to `.git-blame-ignore-revs` so `git blame`
  skips it (`git config blame.ignoreRevsFile .git-blame-ignore-revs`).

---

# Follow-up: manual lint fixes — group A (mechanical)

**Date:** 2026-10-02

Fixed 47 of the 60 remaining findings. The other 13 are deferred: 10 suppressions (group B:
B008 FastAPI `Depends`, F401 availability probes, deliberate E402 in benchmark_pipeline_modal.py,
B023 in the ablation lambda) and 3 real issues (group C: unchecked `gmx energy` result in
simulation.py, dead `rng`/`seed` in build_hiv_pr_dataset.py, a test in test_boltz.py that only
exercises its own mock).

## What changed

- **B904 ×21 — exception chaining.** `raise X` inside `except` → `raise X from err`, naming the
  handler (`except ImportError as err:`) where it was anonymous. For the optional-dependency
  guards this means the underlying ImportError (e.g. a broken native lib vs. not installed) now
  shows in the traceback instead of being hidden behind "X is not installed". One exception:
  `benchmark_pipeline_modal.py` raises "No chains found" from a `KeyError` fallback path where
  the KeyError is expected, not the cause — that one is `from None`.
- **B905 ×14 — `zip(..., strict=True)`.** Every site pairs sequences that must be equal length
  (oracle scores ↔ sequences, aligned CA atom lists, x/y for correlation, WT ↔ mutant residues,
  DMS dataframe columns). A length mismatch now raises `ValueError` instead of silently
  truncating — a mismatch at any of these sites would be a bug. Ruff's own fix inserts
  `strict=False` (behaviour-preserving); we deliberately chose `strict=True`. Requires Python
  ≥3.10, matching `requires-python`.
- **E741 ×6** — `l` renamed to `line` / `pkg` / `lig`.
- **E402 ×3** — `import redis` moved to the top of orchestrator/tasks.py; the
  `orchestrator.progress` import in api/main.py moved above the `if MODAL_ENABLED` block (it
  doesn't depend on the branch); a mid-file schema import in tests/test_agent.py merged into the
  top import.
- **F841 ×3** — removed dead `results_dir` (backends/boltz.py), `length` (mutation_search.py),
  `out` (tests/test_ligands.py).

## Gotcha (for anyone scripting similar fixes)

The B904 edits were applied by an AST script. Python's `ast` `col_offset` is a **UTF-8 byte**
offset, not a character index, so on two lines containing an em-dash in api/main.py the
`from err` landed at the start of the following line. Caught in review and fixed by hand.

---

# Follow-up: manual lint fixes — group B (fix vs. suppress)

**Date:** 2026-10-02

Originally planned as 10 suppressions. On review, most of the flagged code had a real fix, so
only the genuine availability probes are suppressed. Rule of thumb applied: suppress only when the
flagged code is correct *and* the linter's alternative would be worse.

## Fixed (7)

- **B008 ×5 — FastAPI `Depends()` in defaults (api/main.py).** Switched to FastAPI's recommended
  `Annotated` style via a `SessionDep = Annotated[Session, Depends(get_db)]` alias, rather than
  whitelisting `Depends` in ruff config. Verified by introspecting `app.routes`: all 5 endpoints
  still declare `get_db` as a dependency. (test_api.py needs Postgres, so it wasn't run.)
  Source: https://fastapi.tiangolo.com/tutorial/dependencies/ (Annotated, supported since 0.95).
- **E402 ×1 — benchmark_pipeline_modal.py.** The late `from benchmark_modal import ...` was
  assumed deliberate, but import position doesn't matter: locally it's a sibling module, and
  remotely `add_local_file` bakes it into the image at build time. `benchmark_modal` itself
  imports `modal_app`, so the module load order is unchanged. Moved to the top; the comment
  above `add_local_file` now says "imported above".
- **B023 ×1 — ablation lambda.** `oracle` captured the loop variable `landscape`. It was only
  used within the same iteration (false positive today), but binding it as a default argument
  (`landscape=landscape`) makes it correct even if the lambda is ever stored. Smoke-ran
  `run_ablation` to confirm.

## Suppressed (3) — `# noqa: F401` with a reason

- `from rdkit.Chem import AllChem` (ligands.py, Vina path) and `from openmm.app import Modeller`
  (membrane.py): these imports are the dependency check. Ruff's suggested
  `importlib.util.find_spec` only checks the package exists on disk and misses broken installs
  (e.g. a conda OpenMM with a missing native lib), which is exactly what should fail early. Same
  pattern as the existing `noqa` probes in backends/stubs.py.
- `from IPython.display import display` (scripts/API-Testing/ColabFold-Testing/colabfold_cpu.py):
  prototype script; deleting the import was an option, but kept and suppressed per user's call.

## Verification

- `ruff check .` → 3 remaining, all group C (F841 ×3, deferred).
- test_ligands, test_membrane, test_mutation_search, test_boltz, test_orchestrator →
  155 passed, 2 skipped.
- **Update:** test_api.py run against Docker Postgres + Redis (`docker compose up -d postgres redis`)
  → 5/5 passed, confirming the `SessionDep` / `Annotated` change end-to-end. Redis is required
  too: `/predict` enqueues a Celery task (no worker needed), and without a broker it returns 500.

---

# Follow-up: group C1 — check `gmx energy`'s exit status

**Date:** 2026-10-02

## Problem

Every GROMACS step in `orchestrator/simulation.py` runs through the local `_gmx()` helper
(`check=True`) — except `gmx energy`, which called `subprocess.run` directly and ignored the
result. Ruff flagged the one in `run_gromacs_em` (F841, `energy_proc` unused); the identical call
in `run_gromacs_md` wasn't flagged only because it wasn't assigned. On failure the real error was
hidden and surfaced one line later as a `FileNotFoundError` on the `.xvg`.

## Risk check before changing (real binary)

Switching to `check=True` would break runs that currently succeed if `gmx energy` returned
non-zero on success. Tested against real GROMACS 2025.2 in a throwaway container from the
`propredict-celery_worker` image (`docker run --rm`, repo mounted read-only), wrapping
`subprocess.run` to record exit codes:

| Case | `gmx energy` rc | `.xvg` written |
|---|---|---|
| Successful EM, `myprotein.pdb` (467 frames, PE −377,809.72 kJ/mol) | 0 | yes |
| Missing `em.edr` | 1 | no |
| Corrupt `em.edr` | 1 | no |

Risk not real; the exit code is a reliable success signal. (GROMACS isn't installed on the host —
`CLAUDE.local.md`'s `/opt/homebrew/bin/gmx` is stale.)

## Change

Both sites → `_gmx("energy", "-f", "em.edr", "-o", "<file>.xvg", stdin_input="Potential\n")`.
Failures now raise `CalledProcessError` naming `gmx energy`, like every other step.
Re-ran the probe on the modified code: `energy` now runs with `check=True`, rc 0, identical PE.
Mocked suites (boltz, orchestrator, membrane, ligands): 114 passed, 2 skipped.

## Found along the way → ISSUES.md I-22

To reach `gmx energy`, the probe had to inject `grompp -maxwarn 1` (probe only, no repo change):
`run_gromacs_em` never runs `genion`, so on any net-charged protein GROMACS 2025's `grompp`
aborts on the Ewald net-charge warning. It also has no callers. Logged rather than fixed here.
Also noted: the worker image is ~6 months old (Python 3.10 vs `Dockerfile.celery`'s 3.11).
