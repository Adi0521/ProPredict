# Boltz-2 affinity fix: wrong JSON key + wrong units

**Date:** 2026-07-21
**Source:** `research_plan/rowA-boltz-affinity-invariance.md` § 0 "Two blocking bugs" (Bug 1
and Bug 3). Bug 2 (homodimer support) is deliberately **not** in this change — see below.
**Status: fixed.** 166 unit tests pass.

## The bug

`orchestrator/backends/boltz.py` read the affinity summary as:

```python
affinity_score = aff_data.get("affinity")
```

Boltz-2 does not write a key called `affinity`. Per the research plan's verification against
`boltz==2.2.1` (`src/boltz/data/write/writer.py:308-326`), it writes:

```python
affinity_summary = {
    "affinity_pred_value": pred_affinity_value.item(),
    "affinity_probability_binary": pred_affinity_probability.item(),
}
```

So `.get("affinity")` returned `None` on **every run that has ever executed**. Consequences:
`StructurePrediction.affinity_score` was always `None`, the success log's affinity branch never
fired, and `apply_mutation`'s `if pred.affinity_score is not None:` never fired. The entire
affinity capability was silently dead and had never produced a number.

**Why it survived.** `tests/test_boltz.py` mocked the affinity file as
`{"affinity": -8.42, ...}` — the fixture encoded the same wrong key as the code, so the test
agreed with the bug and stayed green. The assertion was `== pytest.approx(-8.42)`; even a bare
`is not None` would have caught it, but only against a fixture using the real keys.

**Second defect (Bug 3): the units were wrong everywhere.** The value was labelled `kcal/mol`.
`affinity_pred_value` is actually **log10(IC50) with IC50 in µM** — lower means tighter
binding. Nothing performed arithmetic on it, so no math was wrong, but the label had leaked
into the agent's prompt and tool results, and the agent *reasons* over that number. "−8.4
kcal/mol" and "−8.4 log10(IC50 µM)" are very different claims about a molecule.

## What changed

| File | Change |
|---|---|
| `backends/boltz.py` | read `affinity_pred_value` + `affinity_probability_binary`; exclude `pae` files from the glob; log real units |
| `models/schemas.py` | new `affinity_probability` field; corrected `affinity_score` comment |
| `orchestrator/tasks.py` | log line units |
| `orchestrator/agent.py` | tool-result key `affinity_kcal_mol` → `affinity_log10_ic50_um`; new `affinity_binder_probability`; prompt + system-prompt wording |
| `modal_app.py` | surface `affinity_probability` in the GPU smoke-test summary |
| `tests/test_boltz.py` | fixture uses the real keys; `is not None` regression guard; new pae-collision test |
| `tests/test_agent.py` | `_fake_pred` carries `affinity_probability`; assert the new keys and that the old one is gone |

### Decision — surface `affinity_probability_binary` as well

The plan floated it as optional; taken. It is a **different quantity from a different head**
trained on different data: binder-vs-decoy detection, the right output for hit discovery,
whereas `affinity_pred_value` is the one for SAR / Δ work. Both the schema comment and the
agent system prompt say explicitly that the two must not be combined into one number.

### Decision — the glob, resolved by the GPU run (2026-07-21)

**Superseded by verification.** The section below records why the glob was initially left
broad; the A10G run settled it and the glob is now anchored to `affinity_*.json`.

Ground truth from `modal_app.py::test_boltz_affinity_gpu`:

```json
"all_json_basenames": ["affinity_input.json", "confidence_input_model_0.json",
                       "manifest.json", "input.json"],
"affinity_json_keys": {"affinity_input.json": [
    "affinity_pred_value",  "affinity_pred_value1",  "affinity_pred_value2",
    "affinity_probability_binary", "affinity_probability_binary1", "affinity_probability_binary2"]},
"keys_match_our_parser": true, "affinity_score": 1.2158, "affinity_probability": 0.1659,
"PASS": true
```

Boltz writes exactly one affinity file, `affinity_<record_id>.json`. So the plan's original
suggestion was right, and the caution below was defending against a filename
(`input_affinity_0.json`) that **only ever existed in the fabricated mock** — a third instance
of the same root cause: the fixture was written from imagination rather than observation. The
fixture now uses the real filename, the real key set (including the `*1`/`*2` ensemble
members), and values captured from the run.

**Lesson to carry forward: pin fixtures to observed output, never to what the code expects.**

### Original reasoning (pre-verification) — the glob was NOT anchored to `affinity_*.json`

The plan suggested tightening `*affinity*.json` to `affinity_*.json`. **Not done**, and this is
the one place the plan should not be followed literally. Boltz's affinity filename is
record-id dependent, and the existing test fixture writes `input_affinity_0.json` — which an
anchored `affinity_*.json` glob would not match at all. Since boltz is not installed locally
(this project only runs it on Modal), the real filename could not be verified here, and
anchoring on an unverified guess risked replacing a silent-`None` bug with a silent-miss bug.

Instead the glob stays broad and **excludes basenames containing `pae`**, which is the actual
collision the plan was worried about (`pae_affinity_*.json` sorting first), plus `sorted()` for
a deterministic pick. `tests/test_boltz.py::test_call_boltz_ignores_pae_affinity_file` covers it.

### Not in this change — Bug 2 (homodimer support)

`call_boltz` still builds exactly one protein chain, so obligate homodimers (HIV-1 protease,
whose active site forms at the dimer interface) cannot be modelled. That needs a
chain-multiplicity argument and ligand chain IDs shifted off `B` to avoid collision — a real
API change deserving its own review and write-up, per `CLAUDE.md`'s step-by-step rule. It
remains a blocker for the Row A experiment.

## Verification status — CONFIRMED on real output (2026-07-21)

**`modal run modal_app.py::test_boltz_affinity_gpu` → `PASS: true` on an A10G.** The full
output is in the glob section above and logged as Run 011 in `benchmarks/BENCHMARKS.md`.
Confirmed: the keys are `affinity_pred_value` / `affinity_probability_binary`, the file is
`affinity_<record_id>.json`, and `call_boltz` now returns both values populated
(`affinity_score=1.216`, `affinity_probability=0.166` for ethanol vs a 33-mer — weak and
improbable, which is the correct answer for that pair).

This mattered because local coverage is **mock-only**, and mocks are exactly what hid this bug
for months. The mocked tests assert the corrected keys, but a mock cannot prove those keys
match what Boltz-2 writes — before this run, that rested entirely on the research plan's
reading of the 2.2.1 source. It also could not have been caught by `modal_app.py::test_boltz_gpu`,
which folds a bare sequence with no ligand, so Boltz never runs the affinity head and both
fields stay `None` there regardless of whether the parser is right.

The verification function deliberately **does not trust our own parser**. Part 1 runs the
`boltz predict` CLI
directly (not through `call_boltz`) on a protein + ligand with an `affinity` property, then
reports the real output filenames and the top-level JSON keys of every `*affinity*` file —
ground truth that nothing in our code can influence. Part 2 runs `call_boltz` on the same
input and checks that both `affinity_score` and `affinity_probability` come back populated.
`PASS` is true only if the raw keys contain `affinity_pred_value` **and** both fields parsed.

Two GPU runs; the ground-truth one drops to 50 sampling steps since it only needs the file
layout, not a good structure.

Keep this function around: it is the only check in the tree that can catch a Boltz-side
rename, and it costs one A10G run.

## Still open after this change

- **Homodimer support (Bug 2)** — see above. Blocks the Row A / HIV-PR affinity experiment,
  which is the only planned use with real measured binding data to validate against.
- **boltz is installed from unpinned git HEAD** (`modal_app.py:42`,
  `pip_install("git+https://github.com/jwohlwend/boltz.git")`). Modal caches the layer, so the
  version behind every benchmark to date is whatever HEAD was at first image build, and it is
  not recorded anywhere. Pin it before the next quality run; also a prerequisite for baking
  weights into the image via `boltz download` (currently commented out at `modal_app.py:51`).

## Tests

```
pytest tests/ --ignore=tests/test_api.py -k "not integration"    # 166 passed, 4 deselected
modal run modal_app.py::test_boltz_affinity_gpu                  # PASS (A10G, 2026-07-21)
```

Note: a bare `pytest tests/` hangs — `tests/test_esmfold_local.py`'s integration test loads
the real ESMFold model. Pre-existing, unrelated to this change, but it means `-k "not
integration"` is the usable local loop.
