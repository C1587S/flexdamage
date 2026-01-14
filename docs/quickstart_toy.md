# Toy Example Quickstart

This document provides a step-by-step guide to running the fully self-contained toy example included in the repository.

## Overview
The toy example script (`scripts/toy_example_pipeline.py`):
1. Generates synthetic data (simulating a convex mortality-temperature relationship).
2. Creates a configuration file on the fly.
3. Runs the full Global -> Regional -> Diagnostics -> F2 pipeline.
4. Verifies that all expected outputs (JSON, CSV, Plots) are created.

## Running the Example

Execute the script from the repository root:

```bash
python scripts/toy_example_pipeline.py
```

## Expected Output

You will see logs indicating progress:

```
[1] Generating Synthetic Data...
    Saved 2000 records to toy_run/data/mortality_dummy.csv

[2] Creating Configuration...
    Config saved to toy_run/config.yaml

[3] Running Estimation Pipeline...
    INFO:flexdamage.core.pipeline:Starting Pipeline: toy_mortality_example
    ...
    INFO:flexdamage.core.pipeline:Saved regional results to toy_run/results/regional_results.csv
    INFO:flexdamage.core.pipeline:Running Diagnostics...
    INFO:flexdamage.core.pipeline:Generating F2 Tables...

[4] Verifying Results...
    [OK] Global Results found: ...
    [OK] Regional Results found: ...
    [OK] Diagnostics generated: ['hist_beta.png', 'regional_curves.png', ...]
    [OK] F2 Tables generated: ['f2_projections.csv']

Toy Example Completed Successfully!
```

## Inspecting Results
Check the `toy_run/results` folder to see the generated artifacts.
