# Repository Structure

This document outlines the organization of the FlexDamage library.

## Directory Layout

- **flexdamage/**: Main package source code.
    - **core/**: Core estimation logic.
        - `global_est.py`: Handles global parameter estimation (e.g., income elasticity/Gamma).
        - `regional.py`: Handles regional parameter estimation (e.g., response curves).
        - `pipeline.py`: Orchestrator that connects data, models, and estimators.
        - `f2_gen.py`: Generates future projections (F2 tables).
    - **data/**: Data access layer.
        - `backends.py`: `PandasBackend` and `DuckDBBackend` classes.
    - **models/**: Mathematical models.
        - `symbolic.py`: Parses explicit user formulas using SymPy.
    - **config/**: Configuration management.
        - `schema.py`: Pydantic models defining valid configuration structure.
    - **diagnostics/**: Visualization tools.
        - `plots.py`: Generates histograms and curve plots.
    - **utils/**: Helper utilities for logging and monitoring.

- **configs/**: Configuration files.
    - `examples/`: Config templates for different scenarios.

- **scripts/**: CLI entry points.
    - `run_estimation.py`: Main script to execute a workflow.

- **tests/**: Unit tests using pytest.

- **notebooks/**: Jupyter notebooks for tutorials and exploration.
