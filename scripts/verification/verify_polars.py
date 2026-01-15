
import pandas as pd
import numpy as np
import sys
import os
import shutil
from pathlib import Path

# Fix sys.path to include module
sys.path.append(os.path.join(os.getcwd(), 'flexdamage-dev'))

from flexdamage.data.backends import PandasBackend, PolarsBackend

def verify_polars():
    print("Verifying Polars Backend Consistency...")
    
    # 1. Create Dummy Data
    n = 100
    df = pd.DataFrame({
        "region": np.random.choice(["A", "B", "C"], n),
        "year": np.random.randint(2000, 2020, n),
        "val1": np.random.randn(n),
        "val2": np.random.randn(n)
    })
    
    csv_path = "test_data_polars.csv"
    df.to_csv(csv_path, index=False)
    
    # 2. Init Backends
    pd_backend = PandasBackend.from_csv(csv_path)
    pl_backend = PolarsBackend.from_csv(csv_path)
    
    # 3. Test load_data with filters & columns
    cols = ["region", "val1"]
    filters = {"region": "A"}
    
    res_pd = pd_backend.load_data(columns=cols, filters=filters)
    res_pl = pl_backend.load_data(columns=cols, filters=filters)
    
    # 4. Compare
    # Sort by val1 to ensure order doesn't matter (though order should be preserved usually)
    res_pd = res_pd.sort_values("val1").reset_index(drop=True)
    res_pl = res_pl.sort_values("val1").reset_index(drop=True)
    
    print("\nPandas Result Head:\n", res_pd.head(3))
    print("\nPolars Result Head:\n", res_pl.head(3))
    
    try:
        pd.testing.assert_frame_equal(res_pd, res_pl, check_dtype=False) # Dtypes might differ slightly (float64 vs float32 etc)
        print("\nSUCCESS: Polars backend matches Pandas backend results.")
    except AssertionError as e:
        print("\nFAILURE: Results differ.")
        print(e)
        
    # 5. Test Unique Values
    uniq_pd = pd_backend.get_unique_values("region")
    uniq_pl = pl_backend.get_unique_values("region")
    
    print(f"\nUnique Regions PD: {uniq_pd}")
    print(f"Unique Regions PL: {uniq_pl}")
    
    if uniq_pd == uniq_pl:
        print("SUCCESS: Unique values match.")
    else:
        print("FAILURE: Unique values differ.")

    # Cleanup
    if os.path.exists(csv_path):
        os.remove(csv_path)

if __name__ == "__main__":
    verify_polars()
