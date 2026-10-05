# Celery worker image rebuild

**Date:** 2026-10-02
**Status:** Complete (findings logged; no code changes)

## Why

The `propredict-celery_worker` image was ~6 months old and all real-binary GROMACS checks in the
ruff/C1 pass (`Process/pyproject.md`) ran on it. Rebuilt from the current `Dockerfile.celery`
(`docker compose build celery_worker`) to confirm the current code works on what the Dockerfile
actually produces. Old image kept as `propredict-celery_worker:pre-rebuild` for comparison.

## What changed in the image

| | Old (6 mo) | Rebuilt |
|---|---|---|
| Base | Debian 13 (trixie) | Ubuntu 24.04 (`mambaorg/micromamba:1.5.10-noble`) |
| Python | 3.10.20 | 3.11.16 |
| GROMACS (apt) | 2025.2 | **2023.3** — a *downgrade*; Ubuntu noble's apt ships an older build |
| OpenMM / RDKit / OpenFF | present / present / — | 8.6.1 / 2026.03.1 / 0.18.0 |
| Vina | **missing** | present |
| torch / transformers / meeko / modal / wandb | **all missing** | 2.10.0 / 5.18.0 / 0.8.0 / 1.6.0 / 0.30.0 |
| anthropic | 0.89.0 | **1.5.0** (major) |
| pydantic | 2.5.0 | 2.13.4 |
| Size | 1.41 GB | 4.66 GB |

The old image could not have run ESMFold-local, the ProteinMPNN mutation scorer, or Vina docking.

**Major-version jumps** (`requirements.txt` pins these only with `>=`): checked the API surface the
code actually uses — `transformers.EsmForProteinFolding` imports under 5.18, and
`anthropic.Anthropic().messages.create` still accepts `model/max_tokens/system/tools/messages`.

## Verification on the rebuilt image

- **GROMACS probe** (same as C1): `gmx energy` exits 0 under `check=True`; PE −377,809.56 kJ/mol
  vs −377,809.72 on 2025.2 (~4e-7 relative, version float drift). `run_gromacs_em` still fails on
  the +4 protein without `genion` (ISSUES.md I-22, updated to cover both versions).
- **Test suite in-container** (`--ignore=tests/test_api.py`, repo mounted read-only):
  208 passed, 3 skipped, 1 failed. Skips: Boltz CLI not in this image (expected — Boltz is
  `requirements-gpu.txt`/Modal), ProteinMPNN integration needs env vars.
  The failure is `test_esmfold_local.py::test_call_esmfold_local_integration`: it downloads the
  real `facebook/esmfold_v1` weights and the Docker VM disk is full (59 GB, 0 free). Environment,
  not code — it passes on the host where the weights are cached.

## Findings not acted on

- **Docker VM disk is 100% full.** `docker system df`: 36.7 GB reclaimable images, 8.5 GB build cache.
- **No `.dockerignore`** → `COPY . .` bakes `.env` (incl. `ANTHROPIC_API_KEY`) into the image at
  `/app/.env`. Local-only today, but leaks the key if the image is ever pushed. Compose bind-mounts
  `.:/app` anyway, so the baked copy isn't needed.
- **ESMFold weights are ~8.4 GB, not ~2 GB** as stated in CLAUDE.md and the integration test's
  docstring. The worker image doesn't bake them, so each fresh container downloads 8.4 GB on first
  ESMFold-local call. The integration test also isn't gated on weights being present, despite its
  section comment saying so.

---

# Follow-up: `.dockerignore`, ESMFold size/gating, CLAUDE.md

**Date:** 2026-10-02

## Docker disk

After the user's cleanup: build cache 8.5 GB → 0; VM disk 100% → 63% used (21 GB free), enough for
the ~8 GB ESMFold download. ~25 GB of unused images remains reclaimable, mostly other projects'.

## `.dockerignore` (new)

Excludes secrets (`.env`, `.env.*`, keeping `.env.example`), VCS/local tooling (`.git`, `.github`,
`.claude`, `CLAUDE.local.md`, `.modal/`), Python/lint caches, and large regenerable outputs
(`wandb/`, ProteinGym cache, ColabFold results, GROMACS run files). Applies to both Dockerfiles.
Verified by rebuilding the worker: `/app/.env`, `/app/.git`, `/app/wandb` are absent.
**Not yet cleaned:** `propredict-celery_worker:pre-rebuild` and the 6-month-old
`propredict-api:latest` still contain a baked `.env`.

## ESMFold weights: size + test gating

- Measured: `facebook/esmfold_v1` weight file is 7.9 GiB on disk (HF reports 8442 MB) — not
  "~2 GB". Corrected in CLAUDE.md, README.md, and the test docstrings. (Host HF cache holds two
  snapshots — `pytorch_model.bin` and `model.safetensors` — totalling 16 GB.)
- `tests/test_esmfold_local.py::test_call_esmfold_local_integration` was documented as "skipped
  unless weights are available" but actually started the 8 GB download. Now skips unless a weight
  file (`model.safetensors` or `pytorch_model.bin`) is already in the HF cache, checked offline via
  `huggingface_hub.try_to_load_from_cache`. The docstring gives the one-liner to populate the cache.
  Also fixed the module docstring, which claimed a GPU/MPS requirement — it runs on CPU.
- Verified both branches: worker container (no cache) → 6 passed, 1 skipped in 3 s;
  host (cached) → 7 passed (integration test ran end-to-end, 7 min on M3 CPU).
- README now notes the worker's HF cache isn't on a volume (survives restart, not
  `docker compose down`/rebuild). Not changed: whether to add a cache volume.

## CLAUDE.md corrections

- Commands: `pip install -e ".[all]"`, `ruff check/format`, and `docker compose up -d postgres redis`
  for test_api.py.
- test_api.py: needs Postgres **and** Redis (`/predict` enqueues to Celery); `init_db()` runs in a
  session fixture, not on import; the fixture **deletes every `jobs` row**.
- ESMFold ~8 GB; dependencies declared in both pyproject.toml and requirements*.txt (Boltz pin in
  three places); don't remove `.env` from `.dockerignore`.

## Follow-up: which ESMFold weight file loads (2026-10-03)

- The `facebook/esmfold_v1` **main** branch ships only `pytorch_model.bin`. The cached
  `model.safetensors` came from **refs/pr/6**, an unmerged safetensors-conversion PR.
- Checked by intercepting `transformers.modeling_utils._get_resolved_checkpoint_files` (stops
  before the 8 GB load): on the host (transformers 5.7.0) the model resolves to
  `pytorch_model.bin` both online and with `HF_HUB_OFFLINE=1`. The PR #6 safetensors was unused.
- The user removed the unused revision via `huggingface_hub.scan_cache_dir().delete_revisions(
  "ba837a39…").execute()` (blob + snapshot + `refs/pr/6`; dry-run reviewed first). Host cache
  16 GB → 7.9 GB; offline resolution still `pytorch_model.bin`.
- **Not verified:** what transformers 5.18 (worker image) resolves to from an empty cache. Two
  in-container attempts were abandoned — the download ran at ~1.2 → 0.7 → 0.14 MB/s (with and
  without `HF_TOKEN`; token auth confirmed working), i.e. the bottleneck is the network path, not
  HF anonymous rate limits. `HF_TOKEN` was added to the user's `.env`; pass it into containers
  with `--env-file .env` (`.dockerignore` keeps it out of the image).

## Follow-up: persistent Hugging Face cache volume (2026-10-04)

- Added a named volume `hf_cache` → `/root/.cache/huggingface` on `celery_worker` in
  `docker-compose.yml` (compose name `propredict_hf_cache`). The ~8 GB ESMFold weights now
  download once per machine instead of once per container; they survive `docker compose down`
  and image rebuilds, and are only removed by `docker compose down -v`.
- Chosen over bind-mounting the host's `~/.cache/huggingface` read-only: the named volume is
  portable across machines; the cost is one initial download (~8 GB, slow on this network).
- Verified with two throwaway `docker compose run --rm --no-deps celery_worker` containers:
  a marker written by the first was read by the second, then removed.
- `HF_TOKEN` reaches the worker without extra wiring: `.env` is bind-mounted via `.:/app` and
  `config.py` runs `load_dotenv(override=True)` before any model load.
- README updated (it previously said the cache wasn't on a volume).

## Follow-up: `.dockerignore` gaps vs `.gitignore` (2026-10-04)

Added: local run artifacts by name (`result.pdb`, `myprotein.pdb`, `input.pka`,
`benchmark_results.json`, `*.gro`), `benchmarks/hivdb_cache/`, virtualenvs, editor/OS files,
notebook checkpoints, coverage/tox output. Deliberately **not** `*.pdb` — one PDB in
`benchmarks/epistasis_structures/` is tracked. Note `.dockerignore` patterns are root-anchored
(unlike `.gitignore`), so files that appear in subdirectories use `**/` (`**/.DS_Store`,
`**/*.swp`, `**/.ipynb_checkpoints/`, `**/*.py[cod]`).
Verified on rebuild: the four artifacts and `.env` are absent from `/app`, the tracked PDB is
present, no `.DS_Store`, core modules import.
