# Configuration Guide

FlexDamage uses YAML files for configuration. The system is typed and validated using Pydantic.

## Structure

### Run
Basic metadata for the execution.
```yaml
run:
  name: "my_analysis_name"
  output_dir: "./results/my_analysis"
```

### Data
Defines input data location and column mappings.
```yaml
data:
  dataset_dir: "path/to/data"
  columns:
    y: "outcome_variable"      # Dependent variable (e.g. yield, mortality)
    x1: "temperature"          # Main independent variable
    w: "population"            # Weights
```

### Estimation
Controls the statistical models.

**Functional Form**
```yaml
estimation:
  functional_form:
    type: "explicit"
    formula: "alpha * x + beta * x**2" # Custom Formula
```

**Global Estimation**
Settings for the Step 1 global estimation (Gamma).
```yaml
  global:
    method: "fixed_effects"
    temperature_bins: 5
```

**Regional Estimation**
Settings for Step 2 regional estimation.
```yaml
  regional:
    backend: "pandas"          # "pandas" or "duckdb"
    min_observations: 10
```

### Execution
Runtime parameters.
```yaml
execution:
  test_mode: false             # Set true to run on a small sample
  test_sample_size: 1000
```
