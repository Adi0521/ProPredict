# Pinning the ProteinMPNN clone

**Date:** 2026-07-22
**Source:** Same class of bug as `Process/boltz-version-pin.md`, spotted while auditing what
else the Modal image installs unpinned. Not from a plan doc.
**Status: done.** Pinned to `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` (2023-06-27).

## Why this one matters more than a normal dependency pin

ProteinMPNN is consumed as a **git clone**, and the model weights ship *inside the repo*
(`vanilla_model_weights/`, ~26 MB). So the commit does not merely determine the scorer's
code — it determines **the weights, and therefore the scores**. An unpinned clone could
silently change every mutation score the pipeline produces.

Both container images cloned it with `--depth 1` and no revision:

```
modal_app.py:63       .run_commands("git clone --depth 1 …/ProteinMPNN.git /opt/ProteinMPNN")
Dockerfile.celery:19  RUN git clone --depth 1 …/ProteinMPNN.git /opt/ProteinMPNN
```

`--depth 1` with no revision means "whatever `main` points at when this layer is built."
Layer caching then freezes that at an unrecorded value — the identical failure mode that
left Boltz-2 two months stale with no record of which build produced the benchmarks.

This was live risk, not hypothetical: **the Modal image was rebuilt earlier today** (the
boltz pin invalidated that layer and everything after it), which re-ran this clone against
whatever `main` was at that moment.

## What was actually installed

Unlike the Boltz case, everything already agreed:

| Source | Commit | Date |
|---|---|---|
| Local clone (`PROTEINMPNN_PATH=/Users/adi-kewalram/ProteinMPNN`) | `8907e66` | 2023-06-27 |
| Upstream `main` HEAD | `8907e66` | identical (`compare` API: `status: identical`) |

**ProteinMPNN has not changed in three years.** So today's rebuild re-cloned the same commit
the ProteinGym validation gate (`Process/mutation-task-2-validation-gate.md`) and the
checkpoint benchmark (`Process/mutation-task-7-checkpoint-benchmark.md`) were run against.
Nothing drifted; this pin is purely protective.

That also means the Spearman figures in those write-ups remain valid — they are attributable
to `8907e66`, which is now recorded rather than merely presumed.

## What changed

| File | Change |
|---|---|
| `modal_app.py` | full clone + `git checkout 8907e66…`; new `report_proteinmpnn_version()` |
| `Dockerfile.celery` | same pin, kept in lockstep |
| `config.py` | comment recording the pin + how to verify a local clone |
| `.env.example` | clone instructions now include the `checkout`, with the reason |

**Decision — full clone + checkout, not `--depth 1`.** A shallow clone cannot check out an
arbitrary commit, and `--depth 1` alone tracks HEAD, which is the bug. Fetching a single
commit shallowly (`git init` + `fetch --depth 1 <sha>`) would work on GitHub but is more
moving parts for a 26 MB repo. Simplicity won.

**Local clones are not pinned by anything.** The images are now pinned, but a developer's
clone at `PROTEINMPNN_PATH` is whatever they checked out. Since the ProteinGym numbers came
from a *local* run, `config.py` and `.env.example` both document the verification:

```bash
git -C "$PROTEINMPNN_PATH" rev-parse HEAD   # expect 8907e667…
```

## Verification

```
git -C /Users/adi-kewalram/ProteinMPNN rev-parse HEAD   # 8907e667… ✓
grep -rn 8907e667 modal_app.py Dockerfile.celery config.py .env.example   # 5 sites agree ✓
pytest tests/ --ignore=tests/test_api.py -k "not integration"             # 174 passed ✓
```

**Not yet verified in-image.** The pin only takes effect on the next image rebuild, and the
change to the `run_commands` layer will trigger one. Confirm afterwards with:

```bash
modal run modal_app.py::report_proteinmpnn_version
```

Expect `PASS: true` — commit matches, `vanilla_model_weights/` present, `protein_mpnn_run.py`
present. The weights check is deliberate: the whole reason for pinning is that they travel
with the commit, so their absence would be a silent scorer failure.

## Follow-up this leaves open

`scripts/check_boltz_updates.py` (and the weekly GitHub Action that runs it) only watches
Boltz-2. ProteinMPNN now has a pin with **no update visibility** — exactly the gap the boltz
checker was built to close. Generalising that script to check both pinned dependencies is
the natural next step; upstream being dormant for three years makes it low-urgency but not
pointless.

Still unpinned in the Modal image after this: `cuequivariance-ops-torch-cu12`,
`cuequivariance-torch`, and the entire micromamba layer (openmm, pdbfixer, rdkit, openff-*,
vina, ambertools, openbabel, acpype).
