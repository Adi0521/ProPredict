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
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    AGENT_MODEL,
    AGENT_MAX_ITERATIONS,
)
from models.schemas import StructurePrediction, PostProcessingResult
from orchestrator.backends.boltz import call_boltz
from orchestrator.simulation import run_rosetta_relax, run_openmm_simulation, run_gromacs_md
from orchestrator.scoring import count_clashes, compute_post_processing

logger = logging.getLogger(__name__)


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

Guidelines:
- mean_pLDDT >= 75 AND <= 2 clashes -> accept unless context requires simulation
- mean_pLDDT 60-74 -> run Rosetta relax if enabled, then reassess
- mean_pLDDT < 60 -> try run_boltz_prediction if BOLTZ_ENABLED, else escalate
- Membrane or ligand context present -> run simulation before accepting
- If a required backend is disabled -> escalate and explain

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
            return json.dumps({"status": "completed", "rosetta_energy": round(score, 3)})
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
            all_plddts = [round(p.mean_plddt, 2) for p in preds]
            result: Dict[str, Any] = {
                "status": "completed",
                "model_name": "boltz2",
                "seeds_tried": num_seeds,
                "seeds_succeeded": len(preds),
                "best_mean_plddt": round(best.mean_plddt, 2),
                "all_mean_plddts": all_plddts,
                "plddt_spread": round(max(all_plddts) - min(all_plddts), 2),
                "best_seed": best.seed,
            }
            if best.affinity_score is not None:
                result["affinity_kcal_mol"] = round(best.affinity_score, 3)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

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
    or ANTHROPIC_API_KEY is not set.

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

    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — falling back to threshold logic")
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
    }

    disagreement_lines = ""
    if inter_model_data and inter_model_data.get("mean_disagreement_nm") is not None:
        disagreement_lines = (
            f"\nInter-model disagreement ({inter_model_data.get('n_models_compared', '?')} models): "
            f"mean {inter_model_data['mean_disagreement_nm']:.3f} nm CA RMSD"
        )
        if inter_model_data.get("disagreement_regions"):
            disagreement_lines += f"\n  High-disagreement regions: {inter_model_data['disagreement_regions']}"

    affinity_line = (
        f"\nBinding affinity (Boltz-2): {prediction.affinity_score:.3f} kcal/mol"
        if prediction.affinity_score is not None else ""
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

    client_kwargs = {"api_key": ANTHROPIC_API_KEY}
    if ANTHROPIC_BASE_URL:
        client_kwargs["base_url"] = ANTHROPIC_BASE_URL
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
        num_clashes=num_clashes,
        rosetta_energy=state["rosetta_energy"],
        score=score,
        decision=decision,
        agent_reasoning=state["agent_reasoning"] or None,
    )

    if state["sim_result"]:
        post_proc.gromacs_potential_energy = state["sim_result"].get("potential_energy")
        post_proc.simulation_metrics = state["sim_result"]

    updated_pdb = state["current_pdb"] if state["current_pdb"] != prediction.structure_pdb else None
    return post_proc, updated_pdb
