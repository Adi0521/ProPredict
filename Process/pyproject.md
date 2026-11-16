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

---

# Follow-up: group C2 — dead `seed` in `build_hiv_pr_dataset.py`

**Date:** 2026-10-04

## Problem

- `main(path=None, n_per_drug=40, seed=0)` built `rng = random.Random(seed)` and never used it
  (ruff F841). Selection is deterministic — candidates are sorted by log10 fold-change and
  `n_per_drug` evenly spaced indices are taken — so `seed` had no effect.
- Found while checking: the entry point is `main(*sys.argv[1:])`, so CLI args arrive as
  strings. Passing `n_per_drug` (`... PI_DataSet.txt 40`) crashed with
  `TypeError: '>' not supported between instances of 'int' and 'str'` — only `path` ever
  worked from the command line.

## Change

- Removed `rng`, the `seed` parameter, and the now-unused `import random`; comment notes the
  selection is deterministic.
- `n_per_drug = int(n_per_drug)` so the second CLI arg works.
- More than two CLI args now exits with a usage message (`SystemExit`, exit 1) rather than
  being silently accepted — user's call (raise rather than ignore-with-warning).

## Verification

Ran the old and new scripts from scratchpad copies (`main()` always writes
`hiv_pr_resistance_dataset.json` next to the script, so running in place would overwrite the
committed artifact — the ISSUES.md I-3 pattern) against one freshly downloaded HIVDB
`PI_DataSet.txt` (654 KB):

- old vs new, default args → **byte-identical** output (sha256 `962432681ae33556…`).
- new vs the **committed** `benchmarks/hiv_pr_resistance_dataset.json` → **byte-identical**: the
  published Row A input regenerates exactly from current HIVDB data.
- `n_per_drug=40` via CLI: old crashes (TypeError), new = identical to default; `n_per_drug=10`
  → 40 isolate runs (10 × 4 drugs).
- extra arg → `usage: ... [PI_DataSet.txt path] [n_per_drug] (got 3 args)`, exit 1.

## Noted, not changed

`ensure_dataset()`'s docstring says third-party data goes into "a gitignored cache", but
`benchmarks/hivdb_cache/` is **not** in `.gitignore` — a first run would leave a 654 KB
untracked file that could be committed by accident.

---

# Follow-up: group C3 — Boltz wiring test that only tested its mock

**Date:** 2026-10-04

## Problem

`tests/test_boltz.py::test_boltz_appended_to_predictions_when_enabled` patched
`orchestrator.tasks.call_boltz`, then imported and called that same patched name — so it only
asserted that a `MagicMock` returns its `return_value` (ruff F841: `mock_boltz` unused). Its
stated purpose, "main task wires Boltz into predictions list", was never exercised.

## Change

Rewrote it to run the real `_run_prediction_core`, mocking only the backends and Redis
(same pattern as `TestRunPredictionCoreProgress` in tests/test_orchestrator.py):
`BOLTZ_ENABLED=True`, `ENSEMBLE_NUM_SEEDS=1`, agent off, ESMFold → pLDDT 95 (so the decision is
"accept" and the refinement loop — which can also call Boltz — never runs), `call_boltz` → a
`boltz2` prediction, context `{"pH": 6.5}` (non-empty but no MD trigger). Asserts:
`call_boltz` called exactly once with the request's sequence and context and an int seed, and
`result["predictions"]` is `[esmfold, boltz2]` with Boltz's pLDDT intact. The inter-model
alignment step is left unmocked — `tasks.py` already guards it with try/except.

## Verification (mutation check)

- New test passes; `ruff check .` → **0 findings** (the last of the 60).
- Temporarily replaced `predictions.append(boltz_pred)` in `tasks.py` with `pass`: the test
  **fails** (`assert ['esmfold'] == ['esmfold', 'boltz2']`). The old test would have passed.
- Restored; `tasks.py` sha256 identical before/after (`cfead748…`). test_boltz + test_orchestrator
  → 67 passed, 2 skipped.

---

# Follow-up: fix the `[gpu]` / `requirements-gpu.txt` resolution conflict

**Date:** 2026-10-05

## Problem

`pip install ".[gpu]"` failed with `ResolutionImpossible`. Boltz at the pinned commit
`b1ebfc46` uses hard `==` pins that collide with ours:

| Package | ProPredict pinned | Boltz `b1ebfc46` pins |
|---|---|---|
| `requests` | `2.31.0` | `2.32.3` |
| `biopython` | `1.83` | `1.84` |

**`requirements-gpu.txt` was broken the same way, and that is not latent — it is live.** That
file is `-r requirements.txt` plus the Boltz line, i.e. a *single* pip resolution, so
`pip install -r requirements-gpu.txt` already failed before this change. Verified directly.

The Modal image escapes it only by accident of layering: `modal_app.py` calls
`.pip_install_from_requirements("requirements.txt")` and then a *separate*
`.pip_install("git+...boltz@b1ebfc46")`. Two pip invocations, so the second silently upgrades
`requests` → 2.32.3 and `biopython` → 1.84 with no conflict detection. Confirmed by installing
`requests==2.31.0` + `biopython==1.83` into a venv and dry-running the Boltz install against it:
pip reports both being replaced.

So the GPU image has been **running** 2.32.3 / 1.84 all along, while the manifests claimed
2.31.0 / 1.83.

## Change

Bumped both pins to match Boltz exactly, in `pyproject.toml` and `requirements.txt` (the only
two sites; `requirements-gpu.txt` inherits via `-r`, and nothing else pins them). Comments at
both sites say the values are dictated by Boltz's hard pins and must be re-checked when the
Boltz commit moves.

Chose exact-match over loosening to `>=`: the resolver would then pick freely, which is the
opposite of what the Boltz-pin work (Process/boltz-version-pin.md) exists to prevent.

## Verification

All four install paths dry-run resolve (none did for the two GPU ones before):

| Path | Before | After |
|---|---|---|
| `pip install -r requirements-gpu.txt` | **ResolutionImpossible** | resolves (boltz 2.2.1) |
| `pip install ".[gpu,dev]"` | **ResolutionImpossible** | resolves (boltz 2.2.1) |
| `pip install ".[all]"` | resolves | resolves |
| `pip install -e ".[dev]"` | resolves | resolves |

**BioPython 1.83 → 1.84 is the only behavioural risk** (`requests` is a patch bump, and the GPU
image already ran it). The mocked suite does **not** mock `Bio`, so it exercises the real
library: 208 passed, 4 skipped on 1.84, unchanged.

Beyond the suite, ran the same digest script under both versions, with
`BiopythonDeprecationWarning` promoted to an error:

- `Superimposer` doing real work — a rigid z-rotation plus per-atom noise applied to a second
  copy of `benchmarks/epistasis_structures/4G3O.raw.pdb`, so the alignment must undo the
  rotation but not the noise: `mean_disagreement_nm` 0.0584, `per_res_max` 0.0931,
  `per_res_sum` 3.097, 53 common residues.
- `NeighborSearch` returning a genuine non-zero count — synthetic CA chain at 1.5 Å spacing so
  non-adjacent residues fall inside the 3.8 Å cutoff: **6 clashes**.
  (First attempt used 2.0 Å spacing and scored 0, which would have tested nothing.)
- `protein_letters_3to1` / `is_aa` sequence extraction (the `benchmark_modal` path), and the
  `PDBIO` / `MMCIFParser` imports in `backends/boltz.py`.

**Digests were identical on 1.83 and 1.84** apart from the version field, and no deprecation
warning fired. `ruff check .` → all checks passed; `ruff format --check .` → 95 files formatted.

## Noted, not changed

- Under `[gpu]`, Boltz's `numpy>=1.26,<2.0` pulls **numpy 1.26.4**, while `[all]` resolves
  **2.2.6** — GPU and non-GPU environments run different numpy majors. Pre-existing, inherited
  from Boltz, not introduced here.
- Boltz declares `requires-python = ">=3.10,<3.13"`; our `requires-python = ">=3.10"` does not
  reflect the upper bound, so a 3.13 env fails only at Boltz install time.
- `pyproject.toml` is now a **third** Boltz-coupled site, and `scripts/check_boltz_updates.py`
  only knows about `modal_app.py` and `requirements-gpu.txt`. A Boltz bump that changes its
  `requests`/`biopython` pins will silently re-break `[gpu]`. Next step.

---

# Follow-up: teach `check_boltz_updates.py` about the third pin site + Boltz's own pins

**Date:** 2026-10-05

## Why

Two gaps left by the `[gpu]` fix above:

1. `pyproject.toml` became a **third** Boltz pin site, but `scripts/check_boltz_updates.py`
   and the weekly workflow only knew about `modal_app.py` and `requirements-gpu.txt`. The
   generated issue told whoever read it to update *two* files — the instruction was itself a
   drift mechanism.
2. SHA agreement is not sufficient. What actually broke `[gpu]` was Boltz's own
   `requests`/`biopython` `==` pins diverging from ours. A bump can re-break installs with all
   three SHAs in perfect agreement.

## What changed

**`scripts/check_boltz_updates.py`**

- `pyproject.toml` added to `PIN_SITES`. No regex change needed — `_SHA_RE` already matched its
  `boltz @ git+...@<sha>` form, with no false positive from the prose `b1ebfc46` nearby (that's
  8 chars; the regex demands 40). `read_pins()`'s CONFLICT logic works unchanged at three sites.
- New exact-pin cross-check: fetch Boltz's own `pyproject.toml` at a given SHA and report any
  package **both** sides pin with a hard `==` to **different** versions — the precise condition
  for `ResolutionImpossible`. Run against the pinned SHA (are we broken now?) and, when behind,
  against upstream `main` (would adopting the bump break us?).
- Exit code **2** for a conflict, ranking above "behind" (1): broken installs now are worse news
  than an available upgrade. Fits exit 2's existing "repo is internally inconsistent" meaning.
- A failed fetch is `dep_check: "unavailable"` with a stderr warning, **not** fatal. A conflict
  is a property of files in this repo, so it only appears when someone edits them and the next
  run catches it; going red on a transient raw.githubusercontent hiccup trains people to ignore
  the job.
- New JSON fields: `dep_check`, `dep_conflicts_at_pinned`, `dep_conflicts_if_bumped`.

Deliberate constraints kept:

- **Still stdlib-only, still 3.10-safe.** Boltz's pyproject is parsed by regex, not `tomllib`
  — `tomllib` is 3.11+ and `requires-python` is `>=3.10` (the local conda env is 3.10.19, where
  importing it fails). The CI job still installs nothing.
- Fetched from `raw.githubusercontent.com`, which needs no auth and isn't metered against the
  API rate limit the rest of the script uses.
- **Only exact-vs-exact is compared.** Judging whether a *range* of ours excludes their pinned
  version means reimplementing PEP 440 specifiers without `packaging` (not stdlib), and ranges
  have never been the failure mode: `pyyaml>=6.0` vs their `==6.0.2` resolves, as does
  `torch>=2.1.0` vs `>=2.2`.

**`.github/workflows/boltz-version-check.yml`**

- Three-site wording in the header comment, the exit-2 cause list, and the adopt instructions.
- Exit 2 can now arrive *with* valid JSON (a dep conflict is a finding, not a failure to
  measure), so a new branch renders the conflicting pins as a table instead of discarding the
  detail as "unknown".
- A bump that also moves Boltz's pins renders them under "⚠️ This bump also moves dependency
  pins", and the adopt steps gained "move any dependency pins listed above".
- Issue de-dupe generalised from `latest_main` to a `stateKey`: the latest SHA for a bump, the
  conflict signature for a conflict. One issue per *state* still, but a *different* conflict now
  reports instead of being swallowed. A null key (couldn't measure) never matches, so those keep
  reporting, as before.

## Verification

Live run on the real repo: 3 sites agree, 15 exact pins checked, no conflict, up to date,
exit 0. That path alone proves little, so each other branch was forced.

**Script, by mutation** — against a throwaway copy of the four files it reads
(`$TMP/fakerepo`), so the real repo was never edited; `git diff --stat` confirms only the two
intended files changed:

| Mutation | Expected | Got |
|---|---|---|
| all three pinned to `ac1c8582` (v2.2.0) | behind, exit 1 | behind by 48, exit 1 |
| + `requests` back to `2.31.0` | conflict outranks behind, exit 2 | exit 2, reported `requests ours ==2.31.0 / boltz ==2.32.3` |
| `pyproject.toml` SHA alone drifts | exit 2, all three listed | exit 2, all three listed |

The second row reproduces the exact bug fixed in the previous step; the third is the case that
was **invisible before this change**.

**Helpers, by assertion** — `[tool.*]` tables don't leak in as dependencies (a planted
`poison==9.9.9` under `[tool.other]` and ruff's `select = ["E4"]` are both excluded); extras
parse (`uvicorn[standard]==0.24.0` → `uvicorn`); ranges and bare names are skipped; environment
markers stripped; a bare `requests` in Boltz's `test = ["pytest", "requests"]` does not erase
the exact pin; PEP 503 normalisation; and `conflicting_exact_pins` ignores shared-but-equal and
one-sided packages.

**Workflow JS, executed** — extracted from the YAML and run under node with stubbed
`github`/`core`/`context`/`fs` and the `${{ }}` expression substituted, since a syntax or logic
error there would otherwise surface only on a Monday cron. YAML parses; all 7 branches behave:
dep-conflict issue, behind issue, could-not-measure issue, same-conflict → quiet,
different-conflict → update + conflict-specific comment, same-SHA → quiet, and the
"bump also moves pins" table.

`dep_conflicts_if_bumped` could not be exercised from real history — every Boltz commit back
through v2.1.0 pins `requests==2.32.3` / `biopython==1.84`, so there is no real bump that moves
them. Covered by assertion on the comparison function plus a hand-built payload for the JS
rendering; flagged here because that one path has no real-data coverage.

`ruff check .` → all checks passed; `ruff format --check .` → 95 files formatted;
`pytest tests/test_boltz.py` → 24 passed, 2 skipped.

## Not done

No unit-test file for the script (user's call — the helpers are pure and would test cleanly
with tmp files and a canned Boltz pyproject string, so `tests/test_boltz_version_check.py`
stays available as a cheap follow-up). The verification above was run ad hoc instead, so it is
not locked against regression.

---

# Follow-up: declare the dependencies we actually import (`numpy`, new `[bench]` extra)

**Date:** 2026-11-15

## Problem

`pyproject.toml` under-declared four packages that the code imports at module level. All four
happened to be present in the author's conda env, so nothing failed locally — but a fresh
install from the manifests could not run the ProteinGym benchmarks at all.

| Package | Imported by | Was declared? | Actually arrived via |
|---|---|---|---|
| `numpy` | `orchestrator/mutation_scan.py`, `mutation_search.py` — **core** | no | `biopython`'s own `numpy` dep |
| `pandas` | `benchmarks/benchmark_proteinmpnn_checkpoints.py:45` | no | only `boltz`, i.e. `[gpu]` |
| `scipy` | same `:46`, and `benchmark_pipeline_modal.py:292` | no | only `boltz` |
| `pyarrow` | needed by `pd.read_parquet` (`..._checkpoints.py:120`, `..._epistasis.py:94`) | no | **nothing** |

`wandb`, `torch` and `transformers` declare none of the three (checked via their metadata), and
pandas ships `pyarrow` only as an *extra* (`pandas[parquet]`), so `read_parquet` had no engine
on any fresh install — including `[gpu]`.

## Change

- **`numpy>=1.26` promoted to a core dependency** in `pyproject.toml` and `requirements.txt`.
  Core modules import it directly; relying on biopython to supply it was accidental.
- **New `[bench]` extra**: `pandas>=2.2.2`, `scipy>=1.11`, `pyarrow>=14`.
- `[all]` now includes `bench`, so a local dev install can run the benchmarks.
- **`[bench]` deliberately NOT added to `requirements.txt`.** That file is what the API and
  worker images install, and they don't run benchmarks — adding ~100 MB of pandas/scipy/pyarrow
  to every image for nothing. A comment at the bottom of `requirements.txt` and on `[all]`
  records the intentional divergence, since `[all]` is no longer byte-equivalent to it.

**Every new entry is a RANGE, never `==`** — the one real design constraint here. Boltz pins
`scipy==1.13.1` exactly and caps `numpy>=1.26,<2.0`, so an exact pin of ours on either would
have made `[gpu]` unresolvable: precisely the bug fixed two steps ago. The ranges are chosen to
contain Boltz's values.

## Verification

Fresh venvs built **only** from the manifests (never the conda env, which masks the whole bug):

- Before: `pip install -e ".[dev]"` → `import pandas` fails. After: `numpy 2.2.6` present and
  `config` + both `orchestrator.mutation_*` modules import.
- `pip install -e ".[bench]"` → `to_parquet` and both harnesses' exact `read_parquet(columns=…)`
  calls succeed, `scipy.stats.spearmanr`/`wilcoxon` resolve, and
  `benchmarks.benchmark_proteinmpnn_checkpoints` imports. pyarrow 25.0.1.
- **`pyarrow` proved load-bearing:** a venv with `pandas>=2.2.2` + `scipy>=1.11` and nothing
  else does **not** receive pyarrow, and parquet fails with *"Missing optional dependency
  'pyarrow'… Missing optional dependency 'fastparquet'"*. So nothing in the graph covers it.

Resolution, all dry-run:

| Target | numpy | pandas | scipy | pyarrow | boltz |
|---|---|---|---|---|---|
| `.[bench]` | 2.2.6 | 2.3.3 | 1.15.3 | 25.0.1 | — |
| `.[gpu,bench]` | **1.26.4** | 2.3.3 | **1.13.1** | 25.0.1 | 2.2.1 |
| `.[all]` | 2.2.6 | 2.3.3 | 1.15.3 | 25.0.1 | — |
| `-r requirements.txt` | 2.2.6 | — | — | — | — |
| `-r requirements-gpu.txt` | 1.26.4 | 2.3.3 | 1.13.1 | — | 2.2.1 |

`[gpu,bench]` backing numpy down to 1.26.4 and scipy to Boltz's exact 1.13.1 is the ranges
doing their job; an exact pin on either would have failed here.

Manifest parity: the only entries in `pyproject.toml` and not `requirements.txt` are the three
`[bench]` packages plus `ruff` — i.e. nothing diverged by accident.

`scripts/check_boltz_updates.py` → 15 exact pins checked, no conflict, up to date, exit 0
(numpy/scipy are ranges, so correctly not flagged). `ruff check .` → all checks passed;
`ruff format --check .` → 95 files formatted; mocked suite → 208 passed, 4 skipped, unchanged.

## Noted along the way

The `project.license` TOML-table deprecation (still open) now prints a `BuildBackendWarning`
with a `!!` banner on **every** editable install, pointing at 2027-Feb-18. Cosmetic, but it is
noise on a command run often — worth folding into the next cleanup.
