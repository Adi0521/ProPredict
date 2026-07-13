# Mutation Master Plan — Task 4: Fix stale `num_clashes` after refinement

**Date:** 2026-07-12
**Source:** `mutation-plans/Process-plan-master.md` Task 4 (from hardening doc)
**Depends on:** Task 3 (same file, `orchestrator/agent.py`).

## The bug
`num_clashes` was computed once before the agent loop (`agent.py`, pre-loop
`count_clashes(prediction.structure_pdb)`) and never updated, even though tool branches
reassign `state["current_pdb"]`. Result: the final `score` and the reported
`PostProcessingResult.num_clashes` reflected the *original* structure, so the agent
could accept a refined structure carrying new clashes without anyone seeing the real
count.

## Deviation from the plan text (intentional)
The plan said to recompute in the **`run_rosetta_relax`** and **`run_simulation`**
branches. Verified against the actual code, that's not quite right:

- **`run_simulation` does not reassign `state["current_pdb"]`.** It stores results in
  `state["sim_result"]`; the only structure it returns (`simulation_pdb`,
  `simulation.py:925`) is the *fully solvated system* (water + ions + lipids). Recomputing
  clashes there would be a no-op at best, and meaningless (CA–CA clash counting on a
  solvated box) if wired to `simulation_pdb`. **No change made.**
- **`run_boltz_prediction` *does* reassign `state["current_pdb"]`** (plus `plddt_scores`
  / `mean_plddt`) but did not update `num_clashes` — a real instance of the same bug the
  plan omitted. **Fixed here.**

So the recompute was added to the two branches that actually change the structure:
`run_rosetta_relax` and `run_boltz_prediction`. (`apply_mutation`, added in Task 3,
already recomputes — so all three mutating paths are now consistent.)

## Edits — `orchestrator/agent.py`
1. **`run_rosetta_relax` branch:** after `state["current_pdb"] = relaxed_pdb`, added
   `state["num_clashes"] = count_clashes(state["current_pdb"])` and surfaced
   `num_clashes` in the returned JSON.
2. **`run_boltz_prediction` branch:** after `state["mean_plddt"] = best.mean_plddt`,
   added `state["num_clashes"] = count_clashes(best.structure_pdb)` and added
   `num_clashes` to the result dict.
3. **Final reporting:** `PostProcessingResult(num_clashes=num_clashes, ...)` →
   `num_clashes=state["num_clashes"]`. The `score` line already read
   `state["num_clashes"]`, so the reported field and the score now agree. (The pre-loop
   local `num_clashes` is still used for the *initial* prompt to the agent, which is
   correct — that's the structure's starting state.)

## Tests — `tests/test_agent.py` (3 added)
- `test_rosetta_relax_recomputes_clashes`: mock relax + `count_clashes`→7; assert
  `state["num_clashes"]==7`, present in JSON, `count_clashes` called on the relaxed PDB.
- `test_boltz_prediction_recomputes_clashes`: mock `call_boltz` + `count_clashes`→5;
  assert state + JSON reflect 5.
- `test_run_agent_refinement_reports_updated_clashes`: end-to-end. Injects a fake
  `anthropic` module into `sys.modules` (the SDK isn't installed in the `ProPredict`
  env; the client is fully mocked, so no real anthropic types are used), scripts the
  agent to run relax then accept, with `count_clashes.side_effect=[2, 9]`. Asserts the
  returned `PostProcessingResult.num_clashes == 9` (post-relax) and `score == 80 - 9*5`.
  This covers the exact reported-field gap.

## Verification
`python -m pytest tests/test_agent.py tests/test_boltz.py -q` in the `ProPredict` conda
env: **25 passed, 2 skipped** (skips are `test_boltz` GPU integration tests).

## Benchmark note
Per the plan: only affects `AGENT_ENABLED=True` runs that go through Rosetta relax or
Boltz re-prediction. Current `benchmarks/results.jsonl` entries are direct `boltz-2`
backend calls, not agent-mediated — unaffected.

## Next
Task 5 — tests for `orchestrator/ligands.py` and `orchestrator/membrane.py` (independent;
zero coverage today).
