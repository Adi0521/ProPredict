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

### Decision — the glob was NOT anchored to `affinity_*.json`

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

## Verification status — read this before trusting the fix

Local coverage is **mock-only**, and mocks are exactly what hid this bug for months. The
mocked tests now assert the corrected keys, but a mock cannot prove those keys match what
Boltz-2 actually writes; that rests on the research plan's reading of the 2.2.1 source.

**The fix is not confirmed against real output yet.** `modal_app.py::test_boltz_gpu` folds a
bare sequence with no ligand, so Boltz never runs the affinity head and both fields stay
`None` there. Confirming this end-to-end needs a ligand-bearing run on Modal —
`tests/test_boltz.py::test_call_boltz_affinity_integration` is written for it but skips
without the boltz CLI. That run is the outstanding follow-up.

## Tests

```
pytest tests/ --ignore=tests/test_api.py     # 166 passed, 4 skipped
```
