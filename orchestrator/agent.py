import json
import logging
import random
from typing import Optional, Dict, Any, List, Tuple

from config import (
    ROSETTA_ENABLED,
    GROMACS_ENABLED,
    OPENMM_ENABLED,
    BOLTZ_ENABLED,
    MD_PRODUCTION_NS,
    AGENT_API_KEY,
    AGENT_BASE_URL,
    AGENT_MODEL,
    AGENT_MAX_ITERATIONS,
    AGENT_MAX_MUTATIONS,
    PROTEINMPNN_PATH,
    PROTEINMPNN_MODEL_NAME,
    PROTEINMPNN_SEED,
    PROTEINMPNN_NUM_DECODING_ORDERS,
)
from models.schemas import StructurePrediction, PostProcessingResult
from orchestrator.backends.boltz import call_boltz
from orchestrator.backends.esmfold import call_esmfold_api
from orchestrator.simulation import run_rosetta_relax, run_openmm_simulation, run_gromacs_md
from orchestrator.scoring import count_clashes, compute_post_processing
from orchestrator.mutation_scan import score_candidate_mutations

logger = logging.getLogger(__name__)

# Standard amino acids for mutation validation. Deliberately duplicated from
# PredictionRequest.validate_sequence in models/schemas.py rather than imported —
# keeps agent.py independent of the request schema's validators.
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


_AGENT_TOOLS = [
    {
        "name": "analyze_structure",
        "description": (
            "Identify low-confidence residue regions from the per-residue pLDDT profile. "
            "Call this first to understand where the structure needs attention."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "pLDDT threshold; residues below this are flagged (default 70)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "run_rosetta_relax",
        "description": (
            "Run PyRosetta FastRelax to improve sidechain geometry. "
            "Use when pLDDT is borderline (60-75) or when clashes are detected. "
            "Requires ROSETTA_ENABLED=True."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_simulation",
        "description": (
            "Run a full MD simulation (OpenMM preferred, GROMACS fallback). "
            "Use when membrane/ligand context requires dynamics or thermodynamic validation. "
            "Requires OPENMM_ENABLED=True or GROMACS_ENABLED=True."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "production_ns": {
                    "type": "number",
                    "description": "Production run length in nanoseconds (default: MD_PRODUCTION_NS config value)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "run_boltz_prediction",
        "description": (
            "Re-predict the structure using Boltz-2 with multiple random seeds (3-5), which gives "
            "AlphaFold3-class accuracy and supports ligand co-folding with binding affinity estimation. "
            "Runs 3-5 seeds with random values, picks the best by pLDDT, and reports the spread. "
            "Use when ESMFold pLDDT is low, inter-model disagreement is high, or ligands are present. "
            "Requires BOLTZ_ENABLED=True."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "num_seeds": {
                    "type": "integer",
                    "description": "Number of random seeds to try (default 3, max 5)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "scan_mutations",
        "description": (
            "Rank candidate single-point substitutions for the current structure using "
            "the ProteinMPNN structural log-odds scorer (validated against ProteinGym). "
            "Returns top candidates as {position, from_aa, to_aa, score}; a POSITIVE "
            "score means the substitution is more structurally compatible than the "
            "wild-type residue there. This is a STRUCTURAL-COMPATIBILITY ranking only — "
            "NOT a proxy for function, stability, or fitness — so treat it as a shortlist "
            "to feed into apply_mutation, not ground truth. Read-only: does not change "
            "the sequence. Requires PROTEINMPNN_PATH; if unset, the tool reports it is "
            "unavailable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "1-indexed positions to restrict the scan to (e.g. a "
                        "low-confidence region from analyze_structure). Omit to scan "
                        "every position."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max candidates to return, best-first (default 10).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "apply_mutation",
        "description": (
            "Mutate the sequence at a given position and re-predict the structure. "
            "Use to test whether a point mutation resolves a low-confidence region or "
            "matches a requested mutation in context.mutations. Re-runs the active "
            "prediction backend (Boltz-2 if enabled, else ESMFold) on the mutated sequence. "
            "Limited to AGENT_MAX_MUTATIONS calls per session — use them deliberately."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "integer",
                    "description": "1-indexed residue position to mutate.",
                },
                "from_aa": {
                    "type": "string",
                    "description": (
                        "Expected current amino acid (1-letter code) at `position`, for "
                        "verification. Optional but recommended."
                    ),
                },
                "to_aa": {
                    "type": "string",
                    "description": "Target amino acid (1-letter code) to mutate to.",
                },
            },
            "required": ["position", "to_aa"],
        },
    },
    {
        "name": "accept_structure",
        "description": "Accept the current structure. Final decision — call when quality is sufficient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why this structure is accepted"}
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "escalate_structure",
        "description": (
            "Flag the structure for human review. Final decision — call when quality is too low "
            "or when required backends are unavailable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why human review is needed"}
            },
            "required": ["reasoning"],
        },
    },
]

_AGENT_SYSTEM = """\
You are a protein structure quality-control agent. You assess a predicted protein structure, \
optionally refine it with available tools, then make a final decision.

Available prediction backends:
- ESMFold (fast, CPU-friendly, deterministic) — always available
- Boltz-2 (AlphaFold3-class accuracy, GPU, supports ligand co-folding and binding affinity) — when BOLTZ_ENABLED=True
If Boltz-2 produced the current prediction, prefer its structure for downstream refinement.
If ESMFold produced the current prediction and quality is poor, consider run_boltz_prediction before escalating.
When ligands are present and Boltz-2 predicted an affinity score, include it in your reasoning.
Boltz-2's affinity value is log10(IC50) with IC50 in micromolar — NOT kcal/mol, and LOWER means
tighter predicted binding. Its binder probability is a separate binder-vs-decoy head trained on
different data; treat the two as distinct signals and never combine them into one number.

Guidelines:
- mean_pLDDT >= 75 AND <= 2 clashes -> accept unless context requires simulation
- mean_pLDDT 60-74 -> run Rosetta relax if enabled, then reassess
- mean_pLDDT < 60 -> try run_boltz_prediction if BOLTZ_ENABLED, else escalate
- Membrane or ligand context present -> run simulation before accepting
- If a required backend is disabled -> escalate and explain

Two mutation tools work together:
- scan_mutations ranks candidate substitutions by ProteinMPNN structural log-odds
  (positive = more structurally compatible than the wild-type residue). It is a
  read-only shortlist, NOT a proxy for function, stability, or fitness. Requires
  PROTEINMPNN_PATH; if unavailable the tool says so — fall back to analyze_structure
  and context.mutations.
- apply_mutation mutates the sequence at a position and re-predicts. Limited to
  AGENT_MAX_MUTATIONS calls per session — use them deliberately.
Typical flow: scan_mutations to shortlist substitutions in a low-confidence region,
then apply_mutation on the most promising candidate to confirm it improves the
structure. Only mutate when context.mutations requests it or analysis plus a scan give
a concrete reason — never mutate speculatively without stating why in your reasoning.

Be concise. Make a terminal decision as soon as you have enough information."""


def _execute_agent_tool(
    tool_name: str,
    tool_input: Dict[str, Any],
    state: Dict[str, Any],
) -> str:
    """Execute one agent tool call and return a JSON string result."""

    if tool_name == "analyze_structure":
        threshold = float(tool_input.get("threshold", 70.0))
        plddt = state["plddt_scores"]

        low = [(i + 1, s) for i, s in enumerate(plddt) if s < threshold]
        regions: List[Dict[str, Any]] = []
        if low:
            start, prev = low[0][0], low[0][0]
            for res_num, _ in low[1:]:
                if res_num > prev + 3:
                    regions.append({"start": start, "end": prev})
                    start = res_num
                prev = res_num
            regions.append({"start": start, "end": prev})

        return json.dumps({
            "total_residues": len(plddt),
            "low_confidence_count": len(low),
            "low_confidence_fraction": round(len(low) / len(plddt), 3) if plddt else 0,
            "regions_below_threshold": regions,
            "worst_residue": int(plddt.index(min(plddt))) + 1 if plddt else None,
            "worst_score": round(min(plddt), 1) if plddt else None,
        })

    if tool_name == "run_rosetta_relax":
        if not ROSETTA_ENABLED:
            return json.dumps({"error": "ROSETTA_ENABLED=False — cannot run relax"})
        try:
            relaxed_pdb, score = run_rosetta_relax(state["current_pdb"])
            state["current_pdb"] = relaxed_pdb
            state["rosetta_energy"] = score
            state["num_clashes"] = count_clashes(state["current_pdb"])
            return json.dumps({
                "status": "completed",
                "rosetta_energy": round(score, 3),
                "num_clashes": state["num_clashes"],
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    if tool_name == "run_simulation":
        if not (OPENMM_ENABLED or GROMACS_ENABLED):
            return json.dumps({"error": "No simulation backend enabled (OPENMM_ENABLED / GROMACS_ENABLED)"})
        production_ns = float(tool_input.get("production_ns", MD_PRODUCTION_NS))
        pH = float(state["context"].get("pH", 7.4))
        temperature_c = float(state["context"].get("temperature_c", 25.0))
        membrane_ctx = state["context"].get("membrane")
        ligand_ctx = state["context"].get("ligands") or []
        try:
            if OPENMM_ENABLED:
                sim = run_openmm_simulation(
                    state["current_pdb"], pH=pH,
                    temperature_c=temperature_c, production_ns=production_ns,
                    membrane_context=membrane_ctx,
                    ligand_contexts=ligand_ctx if ligand_ctx else None,
                )
            else:
                sim = run_gromacs_md(
                    state["current_pdb"], pH=pH,
                    temperature_c=temperature_c, production_ns=production_ns,
                    membrane_context=membrane_ctx,
                    ligand_contexts=ligand_ctx if ligand_ctx else None,
                )
            state["sim_result"] = sim
            summary: Dict[str, Any] = {
                k: v for k, v in sim.items()
                if k not in ("rmsd_nm", "rg_nm", "protonation")
            }
            if sim.get("rmsd_nm"):
                summary["rmsd_final_nm"] = round(sim["rmsd_nm"][-1], 4)
                summary["rmsd_mean_nm"] = round(sum(sim["rmsd_nm"]) / len(sim["rmsd_nm"]), 4)
            if sim.get("rg_nm"):
                summary["rg_mean_nm"] = round(sum(sim["rg_nm"]) / len(sim["rg_nm"]), 4)
            return json.dumps({"status": "completed", **summary})
        except Exception as e:
            return json.dumps({"error": str(e)})

    if tool_name == "run_boltz_prediction":
        if not BOLTZ_ENABLED:
            return json.dumps({"error": "BOLTZ_ENABLED=False — cannot run Boltz-2"})
        num_seeds = min(int(tool_input.get("num_seeds", 3)), 5)
        num_seeds = max(num_seeds, 3)
        seeds = [random.randint(0, 2**31 - 1) for _ in range(num_seeds)]
        try:
            preds: List[StructurePrediction] = []
            for s in seeds:
                try:
                    pred = call_boltz(state["sequence"], context=state["context"], seed=s)
                    preds.append(pred)
                    logger.info(f"[agent/boltz2] seed={s}: mean pLDDT={pred.mean_plddt:.2f}")
                except (RuntimeError, FileNotFoundError, ValueError) as e:
                    logger.warning(f"[agent/boltz2] seed={s} failed: {e}")
            if not preds:
                return json.dumps({"error": "All Boltz-2 seeds failed"})
            best = max(preds, key=lambda p: p.mean_plddt)
            state["current_pdb"] = best.structure_pdb
            state["plddt_scores"] = best.plddt_scores
            state["mean_plddt"] = best.mean_plddt
            state["num_clashes"] = count_clashes(best.structure_pdb)
            all_plddts = [round(p.mean_plddt, 2) for p in preds]
            result: Dict[str, Any] = {
                "status": "completed",
                "model_name": "boltz2",
                "seeds_tried": num_seeds,
                "seeds_succeeded": len(preds),
                "best_mean_plddt": round(best.mean_plddt, 2),
                "num_clashes": state["num_clashes"],
                "all_mean_plddts": all_plddts,
                "plddt_spread": round(max(all_plddts) - min(all_plddts), 2),
                "best_seed": best.seed,
            }
            if best.affinity_score is not None:
                result["affinity_log10_ic50_um"] = round(best.affinity_score, 3)
            if best.affinity_probability is not None:
                result["affinity_binder_probability"] = round(best.affinity_probability, 3)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    if tool_name == "scan_mutations":
        if not PROTEINMPNN_PATH:
            return json.dumps({
                "error": "PROTEINMPNN_PATH not configured — structural mutation scorer unavailable"
            })
        positions = tool_input.get("positions")
        if positions is not None:
            try:
                positions = [int(p) for p in positions]
            except (TypeError, ValueError):
                return json.dumps({"error": "positions must be a list of integers"})
        try:
            top_k = int(tool_input.get("top_k", 10))
        except (TypeError, ValueError):
            return json.dumps({"error": "top_k must be an integer"})
        try:
            candidates = score_candidate_mutations(
                state["current_pdb"],
                state["sequence"],
                positions=positions,
                top_k=top_k,
                proteinmpnn_dir=PROTEINMPNN_PATH,
                model_name=PROTEINMPNN_MODEL_NAME,
                seed=PROTEINMPNN_SEED,
                num_decoding_orders=PROTEINMPNN_NUM_DECODING_ORDERS,
            )
        except Exception as e:
            return json.dumps({"error": f"mutation scan failed: {e}"})
        return json.dumps({
            "status": "completed",
            "note": (
                "structural-compatibility log-odds; positive = more compatible than "
                "wild-type. Not a function/stability/fitness proxy."
            ),
            "candidates": candidates,
        })

    if tool_name == "apply_mutation":
        applied_so_far = len(state.get("mutations_applied", []))
        if applied_so_far >= AGENT_MAX_MUTATIONS:
            return json.dumps({
                "error": (
                    f"mutation limit reached ({applied_so_far}/{AGENT_MAX_MUTATIONS} "
                    "AGENT_MAX_MUTATIONS) — no further mutations this session"
                )
            })

        try:
            position = int(tool_input["position"])
            to_aa = str(tool_input["to_aa"]).upper()
        except (KeyError, ValueError, TypeError):
            return json.dumps({"error": "position and to_aa are required and must be valid"})

        from_aa = tool_input.get("from_aa")
        seq = state["sequence"]

        if position < 1 or position > len(seq):
            return json.dumps({
                "error": f"position {position} out of range (sequence length {len(seq)})"
            })
        if to_aa not in _VALID_AA:
            return json.dumps({"error": f"'{to_aa}' is not a standard amino acid code"})

        idx = position - 1
        actual_from = seq[idx]
        if from_aa and str(from_aa).upper() != actual_from:
            return json.dumps({
                "error": (
                    f"from_aa mismatch: sequence has '{actual_from}' at position "
                    f"{position}, not '{from_aa}'"
                )
            })

        mutated_seq = seq[:idx] + to_aa + seq[idx + 1:]

        try:
            if BOLTZ_ENABLED:
                pred = call_boltz(mutated_seq, context=state["context"], seed=0)
            else:
                pred = call_esmfold_api(mutated_seq, seed=0)
        except Exception as e:
            return json.dumps({"error": f"re-prediction failed, mutation not applied: {e}"})

        state["sequence"] = mutated_seq
        state["current_pdb"] = pred.structure_pdb
        state["plddt_scores"] = pred.plddt_scores
        state["mean_plddt"] = pred.mean_plddt
        state["num_clashes"] = count_clashes(pred.structure_pdb)
        state.setdefault("mutations_applied", []).append(f"{actual_from}{position}{to_aa}")

        result = {
            "status": "completed",
            "mutation": f"{actual_from}{position}{to_aa}",
            "model_name": pred.model_name,
            "mean_plddt": round(pred.mean_plddt, 2),
            "num_clashes": state["num_clashes"],
        }
        if pred.affinity_score is not None:
            result["affinity_log10_ic50_um"] = round(pred.affinity_score, 3)
        if pred.affinity_probability is not None:
            result["affinity_binder_probability"] = round(pred.affinity_probability, 3)
        return json.dumps(result)

    if tool_name in ("accept_structure", "escalate_structure"):
        state["terminal_tool"] = tool_name
        state["agent_reasoning"] = tool_input.get("reasoning", "")
        return json.dumps({"status": "decision_recorded"})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_agent_refinement(
    prediction: StructurePrediction,
    context: Dict[str, Any],
    sequence: str,
    inter_model_data: Optional[Dict[str, Any]] = None,
) -> Tuple[PostProcessingResult, Optional[str]]:
    """
    Use Claude as an adaptive agent to assess and refine a protein structure.

    The agent iterates: analyze -> (optionally) refine/simulate -> decide.
    Falls back to threshold logic if the anthropic package is not installed
    or AGENT_API_KEY is not set.

    inter_model_data: optional output of align_and_compare_structures(); when
    provided the agent receives disagreement regions in its initial prompt.

    Returns:
        (PostProcessingResult, updated_pdb_or_None)
        updated_pdb is non-None when the agent ran Rosetta relax.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic not installed — falling back to threshold logic (pip install anthropic)")
        return compute_post_processing(prediction), None

    if not AGENT_API_KEY:
        logger.warning("AGENT_API_KEY not set — falling back to threshold logic")
        return compute_post_processing(prediction), None

    num_clashes = count_clashes(prediction.structure_pdb)

    state: Dict[str, Any] = {
        "current_pdb":    prediction.structure_pdb,
        "plddt_scores":   prediction.plddt_scores,
        "mean_plddt":     prediction.mean_plddt,
        "num_clashes":    num_clashes,
        "context":        context,
        "sequence":       sequence,
        "rosetta_energy": None,
        "sim_result":     None,
        "terminal_tool":  None,
        "agent_reasoning": "",
        "mutations_applied": [],
    }

    disagreement_lines = ""
    if inter_model_data and inter_model_data.get("mean_disagreement_nm") is not None:
        disagreement_lines = (
            f"\nInter-model disagreement ({inter_model_data.get('n_models_compared', '?')} models): "
            f"mean {inter_model_data['mean_disagreement_nm']:.3f} nm CA RMSD"
        )
        if inter_model_data.get("disagreement_regions"):
            disagreement_lines += f"\n  High-disagreement regions: {inter_model_data['disagreement_regions']}"

    affinity_line = ""
    if prediction.affinity_score is not None:
        affinity_line = (
            f"\nBinding affinity (Boltz-2): {prediction.affinity_score:.3f} "
            "log10(IC50 in uM) — lower means tighter predicted binding"
        )
    if prediction.affinity_probability is not None:
        affinity_line += (
            f"\nBinder probability (Boltz-2): {prediction.affinity_probability:.3f} "
            "(binder-vs-decoy; separate head from the affinity value)"
        )

    user_msg = (
        f"Assess this predicted protein structure:\n\n"
        f"Sequence length: {len(sequence)} residues\n"
        f"Prediction model: {prediction.model_name}\n"
        f"Mean pLDDT: {prediction.mean_plddt:.1f}/100\n"
        f"Steric clashes: {num_clashes}"
        f"{affinity_line}\n"
        f"Per-residue pLDDT (first 20): "
        f"{[round(s, 1) for s in prediction.plddt_scores[:20]]}"
        f"{'...' if len(prediction.plddt_scores) > 20 else ''}"
        f"{disagreement_lines}\n\n"
        f"Context:\n"
        f"  pH: {context.get('pH', 7.4)}\n"
        f"  Temperature: {context.get('temperature_c', 25.0)} C\n"
        f"  Membrane: {context.get('membrane')}\n"
        f"  Ligands: {[l.get('name') for l in context.get('ligands', [])] or None}\n"
        f"  Mutations requested: {context.get('mutations')}\n\n"
        f"Available backends: "
        f"Rosetta={'enabled' if ROSETTA_ENABLED else 'disabled'}, "
        f"OpenMM={'enabled' if OPENMM_ENABLED else 'disabled'}, "
        f"Boltz-2={'enabled' if BOLTZ_ENABLED else 'disabled'}, "
        f"GROMACS={'enabled' if GROMACS_ENABLED else 'disabled'}"
    )

    client_kwargs = {"api_key": AGENT_API_KEY}
    if AGENT_BASE_URL:
        client_kwargs["base_url"] = AGENT_BASE_URL
    client = anthropic.Anthropic(**client_kwargs)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_msg}]

    for iteration in range(AGENT_MAX_ITERATIONS):
        logger.info(f"Agent iteration {iteration + 1}/{AGENT_MAX_ITERATIONS}")

        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=2048,
            system=_AGENT_SYSTEM,
            tools=_AGENT_TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            logger.warning("Agent ended without terminal tool — defaulting to escalate")
            state["terminal_tool"] = "escalate_structure"
            state["agent_reasoning"] = "Agent completed without an explicit terminal decision"
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_str = _execute_agent_tool(block.name, block.input, state)
                logger.info(f"Tool '{block.name}' -> {result_str[:120]}...")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })
            messages.append({"role": "user", "content": tool_results})

            if state["terminal_tool"] is not None:
                logger.info(f"Agent terminal decision: {state['terminal_tool']}")
                break
    else:
        logger.warning(f"Agent hit max iterations ({AGENT_MAX_ITERATIONS}) — escalating")
        state["terminal_tool"] = "escalate_structure"
        state["agent_reasoning"] = f"Max iterations ({AGENT_MAX_ITERATIONS}) reached without decision"

    score = state["mean_plddt"] - (state["num_clashes"] * 5.0)
    decision = "accept" if state["terminal_tool"] == "accept_structure" else "escalate"

    post_proc = PostProcessingResult(
        num_clashes=state["num_clashes"],
        rosetta_energy=state["rosetta_energy"],
        score=score,
        decision=decision,
        agent_reasoning=state["agent_reasoning"] or None,
    )

    if state["sim_result"]:
        post_proc.gromacs_potential_energy = state["sim_result"].get("potential_energy")
        post_proc.simulation_metrics = state["sim_result"]

    post_proc.mutations_applied = state["mutations_applied"] or None

    updated_pdb = state["current_pdb"] if state["current_pdb"] != prediction.structure_pdb else None
    return post_proc, updated_pdb
