from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime


class IonContext(BaseModel):
    """Ion concentration context."""
    name: str
    concentration_mm: float


class MembraneContext(BaseModel):
    """Membrane context for transmembrane proteins."""
    type: Optional[str] = None  # e.g., "POPC", "DMPC"
    span: Optional[List[int]] = None  # [start_residue, end_residue]


class LigandContext(BaseModel):
    """Ligand binding context."""
    name: str
    smiles: Optional[str] = None
    binding_site: Optional[List[int]] = None


class Context(BaseModel):
    """Environmental and experimental context."""
    pH: float = Field(default=7.4, ge=0.0, le=14.0)
    temperature_c: float = Field(default=25.0, ge=-273.15)
    ions: Optional[Dict[str, float]] = None  # e.g., {"Na+": 150, "Cl-": 150}
    membrane: Optional[MembraneContext] = None
    ligands: Optional[List[LigandContext]] = None
    mutations: Optional[List[Dict[str, Any]]] = None  # e.g., [{"pos": 12, "from": "A", "to": "V"}]
    constraints: Optional[Dict[str, Any]] = None  # e.g., crosslinks, distance constraints

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "pH": 7.4,
            "temperature_c": 25,
            "ions": {"Na+": 150, "Cl-": 150},
            "membrane": {"type": "POPC", "span": [20, 45]},
            "ligands": [{"name": "ATP", "binding_site": [45, 46]}],
            "mutations": [{"pos": 12, "from": "A", "to": "V"}]
        }
    })


class PredictionRequest(BaseModel):
    """Request schema for protein structure prediction."""
    sequence: str = Field(..., min_length=1, max_length=2000)
    context: Context = Field(default_factory=Context)
    priority: str = Field(default="fast", pattern="^(fast|accurate|constraint_driven)$")
    job_timeout_seconds: int = Field(default=600, ge=60, le=3600)
    run_id: Optional[str] = None
    webhook_url: Optional[str] = None

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v):
        """Validate that sequence contains only standard amino acid codes."""
        allowed = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c.upper() in allowed for c in v):
            raise ValueError("Invalid amino acid codes in sequence")
        return v.upper()

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWQYVQG",
            "context": {
                "pH": 7.4,
                "temperature_c": 25,
                "ions": {"Na+": 150, "Cl-": 150}
            },
            "priority": "fast",
            "job_timeout_seconds": 600,
            "run_id": "run-123",
            "webhook_url": "https://example.com/callback"
        }
    })


class StructurePrediction(BaseModel):
    """Structure prediction result (from ESMFold or similar)."""
    structure_pdb: str
    plddt_scores: List[float]
    mean_plddt: float
    pae_scores: Optional[List[List[float]]] = None  # Predicted aligned error (not returned by ESMFold)
    seed: int
    model_name: str = "esmfold"  # which backend produced this prediction


class PostProcessingResult(BaseModel):
    """Post-processing and scoring result."""
    num_clashes: int
    rosetta_energy: Optional[float] = None
    gromacs_potential_energy: Optional[float] = None
    simulation_metrics: Optional[Dict[str, Any]] = None  # RMSD, Rg, n_frames, pH, backend, etc.
    agent_reasoning: Optional[str] = None               # Claude agent explanation (Stage D)
    score: float
    decision: str  # "accept", "refine", "escalate"


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    run_id: str
    sequence: str
    status: str  # "pending", "completed", "failed"
    predictions: Optional[List[StructurePrediction]] = None
    ensemble_result: Optional[StructurePrediction] = None
    post_processing: Optional[PostProcessingResult] = None
    context: Optional[Context] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    # Multi-model ensemble fields (Stage E)
    n_models_used: Optional[int] = None
    inter_model_disagreement: Optional[List[float]] = None  # per-residue CA RMSD across models (nm)
    disagreement_regions: Optional[List[Dict[str, Any]]] = None  # high-disagreement stretches

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "run_id": "run-123",
            "sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWQYVQG",
            "status": "completed",
            "ensemble_result": {
                "structure_pdb": "ATOM  1  N   ALA A   1...",
                "plddt_scores": [75.2, 76.1, 74.9],
                "mean_plddt": 75.4,
                "seed": 0
            },
            "post_processing": {
                "num_clashes": 0,
                "score": 85.2,
                "decision": "accept"
            },
            "created_at": "2025-11-27T10:00:00Z",
            "completed_at": "2025-11-27T10:05:00Z"
        }
    })


class JobStatus(BaseModel):
    """Job status for polling."""
    run_id: str
    status: str
    progress_percent: int
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
