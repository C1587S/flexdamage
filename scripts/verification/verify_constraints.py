
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + "/flexdamage-dev") # Assuming running from root and package is in flexdamage-dev
# Wait, based on file listing: /Volumes/cil/home_dirs/scadavidsanchez/projects/ag-flex-damage-functions/flexdamage-dev/flexdamage
# Code is in flexdamage-dev subfolder.
sys.path.append(os.path.join(os.getcwd(), 'flexdamage-dev'))

from flexdamage.config.schema import RunConfig, EstimationConfig, FunctionalForm, RegionalEstimationConfig, GlobalEstimationConfig, Constraint, DataConfig, SectorConfig, RunMetaConfig, ExecutionConfig
from flexdamage.core.regional import RegionalEstimator

def verify_constraints():
    print("Verifying Constraints Implementation...")
    
    # Use dict to handle 'global' alias which matches the yaml structure typically
    estimation_dict = {
        "functional_form": {"type": "quadratic"},
        "global": {"method": "fixed_effects"},
        "regional": {"aggregation_level": "adm1"},
        "constraints": [
            {"type": "formula", "expression": "beta >= 0"}
        ]
    }
    
    config = RunConfig(
        run=RunMetaConfig(name="test", output_dir="tmp"),
        sector=SectorConfig(name="ag"),
        data=DataConfig(dataset_dir="tmp"),
        execution=ExecutionConfig(),
        estimation=EstimationConfig.model_validate(estimation_dict)
    )
    
    # 2. Synthesize Data (Quadratic with negative curvature)
    # y = 2*x - 0.5*x^2 + noise
    # Unconstrained beta should be ~ -0.5
    # Constrained beta should be 0.0
    
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "temp": np.linspace(0, 10, n),
        "loggdppc": np.random.normal(10, 1, n)
    })
    # y = alpha*T + beta*T^2
    # alpha=2, beta=-0.5
    df["y"] = 2 * df["temp"] - 0.5 * df["temp"]**2 + np.random.normal(0, 0.1, n)
    
    # RegionalEstimator expects 'x'->'loggdppc', 'y'->'y' in config
    # Wait, in RegionalEstimator code:
    # vars_map = {"x": config.data.columns.get("x", "temp")}
    
    # Let's override column mapping to ensure 'x' maps to 'temp' for the symbolic model
    config.data.columns = {"y": "y", "x": "temp", "w": "pop"}
    
    estimator = RegionalEstimator(config, gamma=0.0)
    
    print("\nRunning estimation...")
    res = estimator.estimate_region("region_1", df)
    
    if res is None:
        print("Error: Estimation failed.")
        return
        
    print("Results:", res)
    beta = res.get("beta", -999)
    print(f"Beta: {beta}")
    
    if abs(beta - 0.0) < 1e-6:
        print("SUCCESS: Beta was constrained to 0.")
    else:
        print(f"FAILURE: Beta {beta} is not 0 (Unconstrained estimate was likely negative).")

if __name__ == "__main__":
    verify_constraints()
