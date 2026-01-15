from typing import List, Dict, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, field_validator

class ExplicitFormula(BaseModel):
    formula: str
    variables: Dict[str, str] = Field(default_factory=dict, description="Map config alias to formula variables")

class FunctionalForm(BaseModel):
    type: Literal["quadratic", "cubic", "spline", "explicit"]
    formula: Optional[str] = None
    
    @field_validator('formula')
    def validate_formula(cls, v, values):
        if values.data.get('type') == 'explicit' and not v:
            raise ValueError('Formula required for explicit type')
        return v

class Constraint(BaseModel):
    type: Literal["concavity", "convexity", "monotonicity", "bounds", "formula"]
    parameter: Optional[str] = None
    expression: Optional[str] = None  # For symbolic constraints e.g. "beta <= 0"

class DataTransformation(BaseModel):
    variable: str
    method: Literal["scale", "log", "offset"]
    value: Optional[float] = None

class DataAggregation(BaseModel):
    dims: List[str]
    method: Literal["mean", "sum"] = "mean"
    weights: Optional[str] = None

class DataConfig(BaseModel):
    dataset_dir: str
    source_format: Literal["zarr", "parquet", "csv", "duckdb"] = "zarr"
    db_name: Optional[str] = None
    table_name: Optional[str] = None
    columns: Dict[str, str] = Field(
        default_factory=lambda: {
            "y": "log_yield_impact",
            "x": "temperature_anomaly",
            "w": "pop"
        }
    )
    group_by: List[str] = ["region"]
    transformations: Optional[List[DataTransformation]] = None
    aggregation: Optional[DataAggregation] = None

class RegionalEstimationConfig(BaseModel):
    aggregation_level: Literal["impact_region", "adm1", "adm2", "country"]
    backend: Literal["auto", "pandas", "duckdb", "polars"] = "auto"
    min_observations: int = 5
    
class GlobalEstimationConfig(BaseModel):
    method: Literal["fixed_effects", "ols"] = "fixed_effects"
    temperature_bins: float = 0.5
    
class EstimationConfig(BaseModel):
    functional_form: FunctionalForm
    global_est: GlobalEstimationConfig = Field(alias="global")
    regional: RegionalEstimationConfig
    constraints: List[Constraint] = []

class ExecutionConfig(BaseModel):
    mode: Literal["local", "slurm"] = "local"
    n_workers: int = 1
    memory_limit_gb: float = 16.0
    
    # Slurm Config
    slurm_account: Optional[str] = None
    slurm_partition: Optional[str] = "normal"
    slurm_time: str = "01:00:00"
    slurm_mem: str = "16G"
    slurm_cpus_per_task: int = 1
    slurm_extra_args: Dict[str, str] = {}
    
    test_mode: bool = False
    test_sample_size: Optional[int] = 1000
    test_seed: int = 42

class SectorConfig(BaseModel):
    name: str
    subsector: Optional[str] = None
    
class RunMetaConfig(BaseModel):
    name: str
    description: Optional[str] = None
    output_dir: str

class RunConfig(BaseModel):
    run: RunMetaConfig
    sector: SectorConfig
    data: DataConfig
    estimation: EstimationConfig
    execution: ExecutionConfig
