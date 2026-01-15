
import pandas as pd
import numpy as np
import sys
import os
import shutil
from pathlib import Path
import logging

# Fix sys.path
sys.path.append(os.path.join(os.getcwd(), 'flexdamage-dev'))

from flexdamage.config.schema import RunConfig, EstimationConfig, FunctionalForm, RegionalEstimationConfig, GlobalEstimationConfig, Constraint, DataConfig, SectorConfig, RunMetaConfig, ExecutionConfig
from flexdamage.core.pipeline import EstimationPipeline

def verify_parallel():
    print("Verifying Parallel Execution...")
    
    # 1. Create Config with n_workers=2
    # Use dict to handle 'global' alias
    est_dict = {
        "functional_form": {"type": "quadratic"},
        "global": {"method": "fixed_effects"},
        "regional": {"aggregation_level": "impact_region", "backend": "pandas"},
    }
    
    config = RunConfig(
        run=RunMetaConfig(name="test_parallel", output_dir="tmp_parallel"),
        sector=SectorConfig(name="ag"),
        data=DataConfig(dataset_dir="test_data_parallel.csv"),
        execution=ExecutionConfig(n_workers=2, test_mode=True),
        estimation=EstimationConfig.model_validate(est_dict)
    )
    
    # 2. Create Dummy Data with multiple regions
    n_regions = 10
    n_per_region = 20
    regions = [f"reg_{i}" for i in range(n_regions)]
    
    rows = []
    for reg in regions:
        for i in range(n_per_region):
            rows.append({
                "region": reg,
                "year": 2000 + i,
                "log_yield_impact": np.random.randn(),
                "temperature_anomaly": np.random.randn(),
                "x1": np.random.randn(), # loggdppc
                "pop": 1000
            })
    df = pd.DataFrame(rows)
    df.to_csv("test_data_parallel.csv", index=False)
    
    # 3. Run Pipeline
    pipeline = EstimationPipeline(config)
    try:
        pipeline.run()
        print("\nSUCCESS: Parallel pipeline run completed without errors.")
    except Exception as e:
        print(f"\nFAILURE: Parallel pipeline failed: {e}")
        # print traceback
        import traceback
        traceback.print_exc()
        
    # Cleanup
    if os.path.exists("test_data_parallel.csv"):
        os.remove("test_data_parallel.csv")
    if os.path.exists("tmp_parallel"):
        shutil.rmtree("tmp_parallel")

if __name__ == "__main__":
    # Setup logging to see "Parallelizing..." message
    logging.basicConfig(level=logging.INFO)
    verify_parallel()
