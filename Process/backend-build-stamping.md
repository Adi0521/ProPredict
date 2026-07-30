# Stamping the backend build onto predictions and benchmark rows

**Date:** 2026-07-21
**Source:** Follow-on from `Process/boltz-version-pin.md`. Pinning fixes *future* drift;
stamping makes drift **visible in the data**, so an unidentifiable build can never again be
discovered months later by archaeology.
**Status: done.** 174 unit tests pass (8 new).

## The gap

`benchmarks/results.jsonl` recorded git commit, config, and environment for every run — but
not which Boltz build produced the numbers. `_environment_info()` *looked* like it covered
this:

```python
try:
    import boltz
    env["boltz_version"] = getattr(boltz, "__version__", "installed (unknown version)")
except ImportError:
    pass
```

**It never once fired.** `log_run()` executes on the dispatching laptop, never in the GPU
container, so `import boltz` always raised and the key was silently skipped —
`grep -c boltz_version benchmarks/results.jsonl` → **0**. Same failure shape as the affinity
bug: a `try/except` that quietly records nothing, with no test asserting presence.

And even had it worked it would have written `2.2.1`, which does not identify a build (see
the pin doc). Two independent reasons the provenance was worthless.

## Design — report from where the truth is

The build is only knowable inside the container that ran it. So the worker reports it and
the local entrypoint lifts it to the run level, rather than the local side guessing.

```
benchmark_one (Modal, boltz installed)   -> result["_backend_version"]
run_benchmark (local entrypoint)         -> collapses to one value -> log_run(backend_build=...)
log_run                                  -> entry["backend_build"] (+ W&B config)
```

| Piece | What |
|---|---|
| `backends/boltz.py::get_boltz_build_info()` | version + resolved commit from `direct_url.json`; `label` = `"2.2.1@b1ebfc46ecf5"` |
| `schemas.py::StructurePrediction.backend_version` | stamped on every prediction |
| `backends/esmfold.py` | stamps `ESMFOLD_MODEL_NAME` (local) / the endpoint URL (remote) |
| `benchmark_modal.py::benchmark_one` | returns `_backend_version` per target |
| `benchmark_modal.py::run_benchmark` | collapses to one; `MIXED:` prefix + warning if targets disagree |
| `benchmarks/log_benchmark.py::log_run` | new `backend_build` arg → top-level `backend_build` key |

**Decisions:**

- **Commit, not version.** `label` carries both (`2.2.1@b1ebfc46ecf5`) because the version
  alone is ambiguous and the commit alone is unreadable.
- **`MIXED:` rather than picking one.** If targets in a run report different builds, the run
  spanned two images and the record must say so. Silently taking the first would manufacture
  exactly the false confidence this work exists to remove.
- **`declared:` fallback.** When no worker reports (all targets failed, or re-logging an old
  file), `log_run` falls back to the commit declared in the repo, prefixed `declared:` — it
  is weaker evidence (what the source pins, not what ran) and is labelled as such.
- **ESMFold stamps the checkpoint**, since that is what determines its output; the remote API
  stamps the endpoint URL, because the hosted service does not report its checkpoint and
  inventing a version would be worse than naming the endpoint.
- **Removed the dead `import boltz` probe** rather than leaving it as decoration.

## Bug found and fixed on the way: provenance was being dropped mid-pipeline

`_run_prediction_core` replaces `best_prediction` in two places (agent update, Rosetta
relax). Both rebuilt it field-by-field:

```python
best_prediction = StructurePrediction(
    structure_pdb=relaxed_pdb,
    plddt_scores=..., mean_plddt=..., seed=..., model_name=...,
)   # affinity_score, affinity_probability, backend_version -> silently GONE
```

So after any relax, the stored `ensemble_result` lost its Boltz affinity. **This was a live
bug, not just a stamping concern** — and it was invisible until now precisely because
`affinity_score` was always `None` (see `Process/boltz-affinity-key-fix.md`). Fixing the
affinity key turned a dormant bug into a real one; adding `backend_version` would have
walked into it a third time.

Fixed with `model_copy(update={...})`, which carries every unlisted field forward — so a
future field added to `StructurePrediction` cannot regress this again. That property is
pinned by `test_model_copy_carries_every_unlisted_field`.

## Tests (8 new)

`tests/test_boltz.py`
- `TestBoltzBuildInfo` — VCS install (version + commit + label), wheel install (no commit),
  boltz absent (all `None`, must not raise), malformed `direct_url.json` (must not crash a
  prediction over a provenance detail)
- `test_call_boltz_stamps_backend_version` — the stamp reaches `StructurePrediction`

`tests/test_esmfold_local.py` — asserts the stamp equals `ESMFOLD_MODEL_NAME` read from
config, not a hardcoded string, so overriding the checkpoint cannot mis-stamp silently.

`tests/test_orchestrator.py`
- `test_backend_version_survives_to_the_result` — propagation into `predictions[]` and
  `ensemble_result`
- `TestPredictionFieldPreservation` — the relax path preserves affinity + build; plus a
  direct guard on `model_copy`

**Testing note worth remembering:** the first version of the build-info test patched
`sys.modules["importlib.metadata"]` and passed vacuously. `import importlib.metadata as md`
binds via `getattr` on the already-imported `importlib` package, so the swap was ignored and
the mock never applied. Patch `importlib.metadata.version` / `.distribution` directly. An
earlier draft of the propagation test had the same flavour of flaw — it asserted a stamp
while mocking out the code that applies it.

## Not done

- **Historical rows are not backfilled.** Runs 001–011 have no `backend_build`, and for Run
  001 the build is genuinely unrecoverable (its image layer is gone). Writing a plausible
  value into old rows would be fabrication; `BENCHMARKS.md` carries the caveat instead.
- **`api/main.py` does not surface `backend_version`** in the response model beyond what
  `StructurePrediction` already carries — it is stored in `result_json`, so it is queryable,
  but there is no dedicated endpoint field.

## Verification

```
pytest tests/ --ignore=tests/test_api.py -k "not integration"   # 174 passed
python -c "from benchmarks.log_benchmark import _declared_boltz_pin; print(_declared_boltz_pin())"
#   -> declared:b1ebfc46ecf5
```

The Modal-side path (`benchmark_one` reporting a real build) is not exercised by the local
suite — it needs a GPU benchmark run. Expect `backend_build: "2.2.1@b1ebfc46ecf5"` on the
next logged run; if it shows `declared:` or `unknown`, the worker-report path is broken.
