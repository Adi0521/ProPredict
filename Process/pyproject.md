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
