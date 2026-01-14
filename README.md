# FlexDamage

A modular library for estimating flexible damage functions across multiple sectors (agriculture, mortality, energy, etc.).


- **Multi-sector support**: Configurable for different sectors and subsectors
- **Flexible Data Backends**: 
  - **In-Memory (Pandas)**: For country/ADM1 levels or smaller datasets
  - **DuckDB**: For high-resolution impact regions and massive datasets
- **Explicit Functional Forms**: Define custom formulas (e.g., `alpha * T + beta * T^2`) in configuration
- **Configurable Aggregation**: Support for impact regions, ADM1, ADM2, and country levels
- **Comprehensive Diagnostics**: Statistical summaries and spatial visualization

## Installation

```bash
pip install -e .
# Or with spatial dependencies
pip install -e ".[spatial]"
```

# Environment

```bash
module load python
source activate /project/cil/home_dirs/scadavidsanchez/envs/mamba_base 
mamba env create -f environment.yml

uv pip install ipykernel
python -m ipykernel install --user --name flex-refactor-python --display-name "flex-refactor-python"

```

## Quick Start

1. Create a configuration file (see `configs/examples/`)
2. Run the estimation pipeline:

```bash
python scripts/run_estimation.py --config configs/sectors/agriculture.yaml --test
```

## Structure

- `flexdamage/`: Core package
  - `core/`: Estimation logic
  - `data/`: Data loading and backends
  - `models/`: Symbolic models and constraints
  - `config/`: Configuration schema
  - `diagnostics/`: Visualization tools
- `configs/`: YAML configuration files
- `scripts/`: Execution scripts