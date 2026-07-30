# Pinning the Boltz-2 version

**Date:** 2026-07-21
**Source:** Surfaced while verifying the affinity fix (`Process/boltz-affinity-key-fix.md`) —
not from a plan doc. Noticed that `modal_app.py` installed Boltz-2 from unpinned git HEAD
while `benchmarks/BENCHMARKS.md` recorded runs against "Boltz-2" with no version.
**Status: done.** Pinned to commit `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`.

## The problem

Two install sites, both unpinned:

```python
modal_app.py:42       .pip_install("git+https://github.com/jwohlwend/boltz.git")
requirements-gpu.txt  boltz @ git+https://github.com/jwohlwend/boltz
```

Modal caches image layers, so the boltz build was frozen at whatever git HEAD was when the
image was first built — with no record of which commit that was. Two consequences:

1. **The benchmark record was unreproducible.** Every row in `BENCHMARKS.md` (TM-score,
   RMSD, pLDDT across 88 CASP15 targets) was produced by an unknown Boltz build.
2. **A silent-drift hazard.** Any cache bust would pull a different Boltz with no signal
   that anything changed — new numbers would look like a pipeline change.

## Recovering what was actually installed

Added `modal_app.py::report_boltz_version` (CPU-only, seconds). pip records the resolved VCS
commit in the distribution's `direct_url.json`, which makes an exact answer possible rather
than a guess:

```json
{
  "boltz_version": "2.2.1",
  "resolved_commit": "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc",
  "torch_version": "2.6.0+cu126", "numpy_version": "1.26.4", "scipy_version": "1.13.1"
}
```

## The trap: version string 2.2.1 is NOT the v2.2.1 release

This is the whole reason the obvious pin would have been wrong. Compared the installed
commit against the tag via the GitHub compare API:

| | |
|---|---|
| installed | `b1ebfc46` (2026-05-29), reports version `2.2.1` |
| `v2.2.1` tag | `cb04aecc` (2025-09-08) |
| relationship | **ahead by 6, behind by 0** |

The 6 commits past the tag, three of which touch numerics:

```
98bd07f9  Allocate tensors directly on target device
83bb04c4  fix: disable autocast using active device type in boltz2   <- numerics
296ec7e9  Clarify default value for --step_scale parameter
c46ac60c  Merge pull request #683
63000a7c  fix/cpu-float32-precision                                   <- numerics
b1ebfc46  Merge pull request #654
```

Boltz's `setup.py` version was never bumped after the tag, so the version string is
ambiguous: `2.2.1` names two different builds. **`pip install boltz==2.2.1` from PyPI would
have silently dropped an autocast fix and a float32-precision fix** — changing numerics
while appearing to be a tightening, and breaking comparability with every recorded
benchmark. Exactly the failure the pin was meant to prevent.

**Decision: pin to the exact SHA**, not to `boltz==2.2.1`. Reproduces the benchmarked build
byte for byte. The cost — no upstream fixes until someone bumps it — is the point of a pin;
bump deliberately and re-benchmark.

One risk checked and cleared: `orchestrator/backends/boltz.py:116` branches on
`int(version.split(".")[0])` to decide pLDDT scaling. `"2.2.1"` → `2` either way, so the
pLDDT path is unaffected by this choice.

## What changed

| File | Change |
|---|---|
| `modal_app.py` | pin `@b1ebfc46...`; comment explaining why not `==2.2.1`; removed the stale "not yet on PyPI" note (boltz *is* on PyPI, latest 2.2.1) |
| `requirements-gpu.txt` | same pin, kept in lockstep |
| `modal_app.py` | new `report_boltz_version()` — how to recover this again after a bump |
| `benchmarks/BENCHMARKS.md` | provenance header + Run 001 caveat + Run 006 amendment (below) |

## Side finding — Run 001 ran on a different Boltz

Commit `b1ebfc46` is dated **2026-05-29**; **Run 001 ran 2026-05-19**. So the image was
rebuilt mid-series and Run 001 used an older, now-unrecoverable Boltz build (its image layer
has been replaced). Runs 002–011 are consistent with the pin.

This recasts Run 006's headline claim — "results are within noise of Run 001, confirming
reproducibility." That agreement (0.5025 vs 0.5028 TM) spanned **both** a harness change and
a Boltz change. As a stability result it is *stronger* than advertised; as a measurement of
diffusion seed noise it is confounded, and the "stochasticity is minimal at 200 steps"
takeaway should not be read as a clean seed-noise number. Both entries annotated in place;
no recorded figures were altered.

## Follow-ups this unblocks

- **Bake the weights** — `#.run_commands("boltz download", timeout=1200)` is still commented
  out at `modal_app.py:51`, so every cold container re-downloads Boltz weights at runtime.
  This was unsafe to enable while the version floated; with the pin it is straightforward,
  and the layer sits after the install so a future bump invalidates it correctly.
- **Next version bump** should be a deliberate change: bump the SHA in both files, re-run
  the CASP15 baseline, and log it as a new run rather than comparing across the boundary.

## Verification

```
pytest tests/ --ignore=tests/test_api.py -k "not integration"   # 166 passed
modal run modal_app.py::report_boltz_version                    # recovers the pinned build
```

The pin itself is not exercised until the next image rebuild — Modal will reuse the cached
layer while the definition string is unchanged, and the pinned SHA is what that cache already
holds, so no rebuild is triggered by this change.
