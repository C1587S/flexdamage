
"""
Toy Example of the FlexDamage Pipeline
======================================

This script demonstrates the end-to-end workflow of the flexdamage library using synthetic data.
It explicitly walks through:
1. Data Generation (simulating a convex mortality-like response)
2. Configuration Setup
3. Global Estimation (Gamma)
4. Regional Estimation (Alpha/Beta)
5. Diagnostics (Plots)
6. F2 Table Generation (Projections)

Usage:
    python scripts/toy_example_pipeline.py
"""

import sys
import shutil
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Ensure package is in path
sys.path.append(str(Path(__file__).parent.parent))

from flexdamage.config.loader import load_config
from flexdamage.core.pipeline import EstimationPipeline
from flexdamage.utils.logging import setup_logging

def main():
    # 1. Setup Environment
    base_dir = Path("toy_run")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()
    
    data_dir = base_dir / "data"
    data_dir.mkdir()
    
    results_dir = base_dir / "results"
    
    print("\n[1] Generating Synthetic Data...")
    # Simulate Mortality: U-shaped relationship with Temperature
    # Impact = alpha * T + beta * T^2
    # Let's say optimal temp is 20C. Impact increases as we move away.
    # Relative to a base, say T_anomaly. 
    # Let's use T_anomaly directly. 
    # Impact = 0.05 * T_anom^2 + 0.1 * T_anom (asymmetric)
    
    n = 2000
    np.random.seed(123)
    df = pd.DataFrame({
        "iso3": np.random.choice([f"ISO_{i}" for i in range(12)], n),
        "year": np.random.randint(2000, 2021, n),
        "gcm": np.random.choice(["ModelA", "ModelB"], n), # Add GCM dimension
        "temperature_anomaly": np.random.uniform(-3, 6, n),
        "pop": np.random.uniform(1e5, 5e6, n),
        "gdppc": np.exp(np.random.normal(9.5, 0.4, n)) # Log-normal GDP
    })
    
    # Coefficients
    true_alpha = 0.05
    true_beta = 0.02 # Convexity
    
    # Impact logic
    df["log_yield_impact"] = (
        true_alpha * df["temperature_anomaly"] + 
        true_beta * df["temperature_anomaly"]**2 + 
        np.random.normal(0, 0.1, n)
    )
    
    csv_path = data_dir / "mortality_dummy.csv"
    df.to_csv(csv_path, index=False)
    print(f"    Saved {n} records to {csv_path}")
    
    # 2. Create Configuration
    print("\n[2] Creating Configuration...")
    config_dict = {
        "run": {
            "name": "toy_mortality_example",
            "output_dir": str(results_dir)
        },
        "sector": {
            "name": "mortality",
            "subsector": "all_cause"
        },
        "data": {
            "dataset_dir": str(data_dir), # Directory containing CSVs
            "source_format": "csv",
            "columns": {
                "y": "log_yield_impact", # Reusing variable name for simplicity
                "x1": "temperature_anomaly",
                "x": "temperature_anomaly", # Fallback for global
                "w": "pop",
                "region": "iso3"
            },
            "aggregation": {
                "dims": ["iso3", "year"],
                "method": "mean",
                "weights": "pop"
            },
            "transformations": [
                {
                    "variable": "x1",
                    "method": "scale",
                    "value": 1.0 # No-op scale just to test logic
                }
            ]
        },
        "estimation": {
            "functional_form": {
                "type": "explicit",
                "formula": "alpha * x + beta * x**2"
            },
            "global": {
                "method": "fixed_effects",
                "temperature_bins": 5
            },
            "regional": {
                "aggregation_level": "country",
                "backend": "pandas",
                "min_observations": 10
            }
        },
        "execution": {
            "test_mode": False
        }
    }
    
    config_path = base_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
    print(f"    Config saved to {config_path}")
    
    # 3. Load & Run Pipeline
    print("\n[3] Running Estimation Pipeline...")
    setup_logging(level="INFO")
    
    # Load typed config
    cfg = load_config(config_path)
    pipeline = EstimationPipeline(cfg)
    
    pipeline.run()
    
    # 4. Verify Outputs
    print("\n[4] Verifying Results...")
    
    # Check Global
    g_res = results_dir / "global_results.json"
    if g_res.exists():
        print(f"    [OK] Global Results found: {g_res}")
    else:
        print(f"    [FAIL] Global Results missing!")
        
    # Check Regional
    r_res = results_dir / "regional_results.csv"
    if r_res.exists():
        r_df = pd.read_csv(r_res)
        print(f"    [OK] Regional Results found: {len(r_df)} regions estimated.")
        print("    Sample parameters:")
        print(r_df[["region", "alpha", "beta"]].head(3))
    else:
        print(f"    [FAIL] Regional Results missing!")
        
    # Check Diagnostics
    diag_dir = results_dir / "diagnostics"
    plots = list(diag_dir.glob("*.png"))
    if len(plots) > 0:
        print(f"    [OK] Diagnostics generated: {[p.name for p in plots]}")
    else:
        print(f"    [FAIL] No diagnostic plots found.")
        
    # Check F2
    f2_dir = results_dir / "f2"
    f2_files = list(f2_dir.glob("*.csv"))
    if len(f2_files) > 0:
        print(f"    [OK] F2 Tables generated: {[f.name for f in f2_files]}")
    else:
        print(f"    [FAIL] No F2 tables found.")
        
    print("\nToy Example Completed Successfully!")
    print(f"Check the '{results_dir}' folder for all outputs.")

if __name__ == "__main__":
    main()
