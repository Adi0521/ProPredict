# Baking the Boltz-2 weights into the Modal image

**Date:** 2026-07-27
**Source:** Follow-up flagged in `Process/boltz-version-pin.md` ("bake the weights — safe once
pinned"). Now that both git-cloned deps are pinned, a rebuild is deterministic, so baking is
sound.
**Status: done in source; verify on next rebuild** (see below).

## Goal

Every cold Boltz-2 container re-downloaded a few GB of weights on its first `predict`, inside
the 1800s function timeout. Baking them into the image moves that to a one-time build step.

## What the pinned source actually says (verified, not assumed)

I could not run boltz locally (GPU-cloud only), so I read `src/boltz/main.py` at the pinned
commit `b1ebfc46` directly rather than guess. Findings that shaped the implementation:

1. **There is no `boltz download` command.** The CLI is a `click.group()` with exactly one
   command, `predict`. The previously-commented `#.run_commands("boltz download", …)` would
   have **failed the build** — a phantom command. Good that it was never enabled.
2. **Weights download lazily inside `predict`** via `download_boltz2(cache)` (main.py:1141),
   which pulls `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, and the `mols/` CCD data. The affinity
   checkpoint is included — needed by `test_boltz_affinity_gpu`.
3. **`download_boltz2` does not create the cache dir**; only `predict` does
   (`cache.mkdir(...)`, main.py:1112). A direct call must `mkdir` first.
4. **Cache resolution**: `get_cache_path()` returns `$BOLTZ_CACHE` (must be absolute) else
   `~/.boltz`. `call_boltz` passes neither `--cache` nor `BOLTZ_CACHE`, so runtime falls
   through to `~/.boltz` today.
5. **boltz2 uses `mols/`, not `ccd.pkl`** — `ccd.pkl` is a boltz1 artifact. (An earlier draft
   of the verifier checked for `ccd.pkl` and would have false-failed.)

## Implementation

Two changes to the Modal image in `modal_app.py`:

**Absolute cache path via env var.** Baking to build-time `~/.boltz` and hoping it matches
runtime `$HOME` is the same "probably works" gamble that caused the version-drift messes.
Instead:

```python
.env({"PROTEINMPNN_PATH": "/opt/ProteinMPNN", "BOLTZ_CACHE": "/opt/boltz-cache"})
```

`BOLTZ_CACHE` is absolute, so build and runtime resolve to the identical directory
independent of `$HOME`. boltz reads it via `get_cache_path()`, and it's the only thing
steering `call_boltz` (which passes no `--cache`).

**Download at build time, CPU-only:**

```python
.run_commands(
    "mkdir -p /opt/boltz-cache",
    "python -c 'from pathlib import Path; from boltz.main import download_boltz2; "
    "download_boltz2(Path(\"/opt/boltz-cache\"))'",
)
```

Calls `download_boltz2` directly rather than running a dummy `predict` (which would need a
GPU build step for a forward pass). Pure download, no GPU. Shell escaping was checked with
`shlex.split` — the `-c` payload parses as valid Python.

**No `timeout=` kwarg.** The old commented line carried `timeout=1200`, but
`Image.run_commands` (modal 1.4.2) accepts only `env / secrets / volumes / gpu / force_build`
— passing `timeout` raises `TypeError` at image-definition time. That was never caught before
because the line was commented out. Modal uses an overall build timeout; the few-GB download
fits inside it.

**Layer ordering is load-bearing:** this step sits *after* the boltz `pip_install`, so
bumping the boltz pin invalidates the install layer → invalidates the download layer →
re-bakes weights matching the new build. Weights can't go stale relative to the code.

**Failure mode is safe:** if a future boltz moves `download_boltz2` out of `boltz.main`, the
build fails loudly. Better than a silent broken image.

## Verification (`report_boltz_version` extended)

`report_boltz_version` now also inspects the cache — CPU-only, no GPU run:

```
modal run modal_app.py::report_boltz_version
```

Expect on the rebuilt image:
- `boltz_cache_dir: "/opt/boltz-cache"`
- `boltz_cache_files: {"boltz2_conf.ckpt": true, "boltz2_aff.ckpt": true}`
- `boltz_cache_has_mols: true`
- `weights_baked: true`

If `weights_baked` is false, cold containers are still re-downloading and the bake did not
take.

**Not yet verified** — the change to the image definition will trigger a rebuild on the next
`modal run` (a long one: the download runs during build). After that:
1. `report_boltz_version` → `weights_baked: true`
2. `test_boltz_affinity_gpu` → still `PASS: true`, and its first predict should start without
   a "Downloading…" pause. Numbers must be unchanged (same weights, just pre-fetched):
   `affinity_score ≈ 1.216`, `affinity_probability ≈ 0.166`.

## Note for the Docker/Celery image

`Dockerfile.celery` was **not** changed. It targets local ESMFold, not GPU Boltz-2, so it
never downloads Boltz weights. If Boltz is ever enabled there, mirror this bake (and the
`BOLTZ_CACHE` env var) into that image too.

## Local tests

```
pytest tests/ --ignore=tests/test_api.py -k "not integration"   # 174 passed
```
Unchanged — this is image configuration; no local test exercises it.
