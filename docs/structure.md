# Repository Structure

This document outlines the organization of the FlexDamage library.

## Directory Layout

```
flexdamage-dev/
├── flexdamage/           # Main package source code
│   ├── core/             # Core estimation logic
│   ├── data/             # Data access layer
│   ├── models/           # Mathematical models
│   ├── config/           # Configuration management
│   ├── diagnostics/      # Visualization tools
│   ├── hpc/              # High-performance computing utilities
│   └── utils/            # Helper utilities
├── configs/              # Configuration files
├── scripts/              # CLI entry points
├── tests/                # Unit tests
├── notebooks/            # Jupyter tutorials
└── docs/                 # Documentation
```

## Module Details

### `flexdamage/core/`
Core estimation logic.

- `pipeline.py`: Main orchestrator that connects data, models, and estimators. Entry point for running full estimation workflows.
- `global_est.py`: Global parameter estimation (e.g., income elasticity/Gamma) using fixed effects or OLS.
- `regional.py`: Regional parameter estimation for response curves (alpha, beta coefficients).
- `f2_gen.py`: Generates future projection tables (F2 tables) using estimated parameters.

### `flexdamage/data/`
Data access layer.

- `backends.py`: Data loading backends including `PandasBackend` and `DuckDBBackend` for different data sources (CSV, Parquet, DuckDB).
- `preprocessing.py`: Data preprocessing utilities including aggregation, transformation, and validation.

### `flexdamage/models/`
Mathematical models.

- `symbolic.py`: Parses user-defined formulas using SymPy for flexible functional form specification.

### `flexdamage/config/`
Configuration management.

- `schema.py`: Pydantic models defining valid configuration structure with type validation.
- `loader.py`: YAML configuration file loader.

### `flexdamage/diagnostics/`
Visualization tools for analysis and quality control.

- `visualizer.py`: `DiagnosticVisualizer` class for generating publication-quality diagnostic plots:
  - Parameter distributions
  - Gamma estimation diagnostics
  - Residual analysis
  - Fit diagnostics
  - Polynomial summary curves
- `maps.py`: `MapVisualizer` class for geographic visualization:
  - Choropleth maps at country and regional levels
  - Comparison maps between scenarios
  - Animated GIFs for time series visualization
- `plots.py`: Legacy plotting functions for quick diagnostics.
- `styles.py`: Consistent styling system (`scientific`, `presentation`) with customizable colormaps.
- `advanced.py`: Advanced diagnostic plots.

### `flexdamage/hpc/`
High-performance computing utilities.

- Support for Slurm job submission and parallel processing.

### `flexdamage/utils/`
Helper utilities.

- `logging.py`: Logging setup and configuration.
- Monitoring and progress tracking utilities.

## Other Directories

### `configs/`
Configuration file templates.

- `examples/`: Example configurations for different use cases (mortality, agriculture).

### `scripts/`
CLI entry points.

- `run_estimation.py`: Main script to execute an estimation workflow.
- `toy_example_pipeline.py`: Self-contained example demonstrating the full pipeline.

### `tests/`
Unit tests using pytest.

- `test_maps.py`: Tests for MapVisualizer.
- `conftest.py`: Shared pytest fixtures.

### `notebooks/`
Jupyter notebooks for tutorials and exploration.

- `tutorial_walkthrough.ipynb`: Step-by-step guide to using FlexDamage components individually.
- `map_visualization_tutorial.ipynb`: Guide to geographic visualization capabilities.

### `docs/`
Documentation.

- `configuration.md`: Configuration guide.
- `structure.md`: This document.
- `quickstart_toy.md`: Quickstart guide using the toy example.
- `faq.md`: Frequently asked questions.
