# Configuration Guide

FlexDamage uses YAML files for configuration. The system is typed and validated using Pydantic.

## Full Configuration Structure

### Run
Basic metadata for the execution.
```yaml
run:
  name: "my_analysis_name"
  description: "Optional description"  # Optional
  output_dir: "./results/my_analysis"
```

### Sector
Identifies the analysis domain.
```yaml
sector:
  name: "mortality"           # or "ag", "labor", etc.
  subsector: "heat"           # Optional subsector
```

### Data
Defines input data location, format, and column mappings.
```yaml
data:
  dataset_dir: "path/to/data"
  source_format: "csv"        # Options: "csv", "parquet", "zarr", "duckdb"
  db_name: "mydb"             # Required for duckdb format
  table_name: "mytable"       # Required for duckdb format

  columns:
    y: "mortality_impact"     # Dependent variable
    x: "lgdp_delta"           # For global estimation (income elasticity)
    x1: "temperature_anomaly" # For regional estimation (temperature response)
    w: "pop"                  # Weights (e.g., population)
    region: "iso3"            # Region identifier column

  group_by: ["region"]        # Default grouping columns

  # Optional transformations applied before estimation
  transformations:
    - variable: "temperature_anomaly"
      method: "scale"         # Options: "scale", "log", "offset"
      value: 1.0

  # Optional aggregation before estimation
  aggregation:
    dims: ["iso3", "year"]    # Dimensions to aggregate by
    method: "mean"            # Options: "mean", "sum"
    weights: "pop"            # Weight column for weighted aggregation
```

### Estimation
Controls the statistical models.

**Functional Form**
```yaml
estimation:
  functional_form:
    type: "explicit"          # Options: "quadratic", "cubic", "spline", "explicit"
    formula: "alpha * x + beta * x**2"  # Required when type is "explicit"
```

**Global Estimation**
Settings for Step 1 global estimation (Gamma / Income Elasticity).
```yaml
  global:
    method: "fixed_effects"   # Options: "fixed_effects", "ols"
    temperature_bins: 5       # Number of temperature bins for FE
```

**Regional Estimation**
Settings for Step 2 regional estimation.
```yaml
  regional:
    aggregation_level: "country"  # Options: "country", "adm1", "adm2", "impact_region"
    backend: "pandas"             # Options: "auto", "pandas", "duckdb", "polars"
    min_observations: 10          # Minimum observations per region
```

**Constraints**
Optional constraints on estimated parameters.
```yaml
  constraints:
    - type: "convexity"       # Enforces beta >= 0
      parameter: "beta"
    - type: "bounds"          # Custom bounds
      parameter: "alpha"
      expression: "alpha >= -1"
```

Constraint types:
- `convexity`: Enforces parameter >= 0 (U-shaped curves)
- `concavity`: Enforces parameter <= 0 (inverse U-shaped curves)
- `monotonicity`: Enforces monotonic response
- `bounds`: Custom bounds via expression
- `formula`: Custom symbolic constraint

### Execution
Runtime parameters.
```yaml
execution:
  mode: "local"               # Options: "local", "slurm"
  n_workers: 1                # Number of parallel workers
  memory_limit_gb: 16.0       # Memory limit per worker

  # Test mode settings
  test_mode: false            # Set true to run on a small sample
  test_sample_size: 1000      # Sample size when test_mode is true
  test_seed: 42               # Random seed for reproducibility

  # Slurm settings (when mode: "slurm")
  slurm_account: "myaccount"
  slurm_partition: "normal"
  slurm_time: "01:00:00"
  slurm_mem: "16G"
  slurm_cpus_per_task: 1
  slurm_extra_args: {}        # Additional slurm arguments
```

## Complete Example

```yaml
run:
  name: "mortality_analysis_2024"
  output_dir: "./results/mortality"

sector:
  name: "mortality"

data:
  dataset_dir: "./data/mortality"
  source_format: "parquet"
  columns:
    y: "mortality_impact"
    x: "lgdp_delta"
    x1: "temperature_anomaly"
    w: "pop"
    region: "iso3"
  aggregation:
    dims: ["iso3", "year"]
    method: "mean"
    weights: "pop"

estimation:
  functional_form:
    type: "explicit"
    formula: "alpha * x + beta * x**2"
  global:
    method: "fixed_effects"
    temperature_bins: 5
  regional:
    aggregation_level: "country"
    backend: "pandas"
    min_observations: 10
  constraints:
    - type: "convexity"
      parameter: "beta"

execution:
  mode: "local"
  test_mode: false
```
