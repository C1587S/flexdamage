import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "temperature_anomaly": np.random.normal(1.5, 0.5, n),
        "log_yield_impact": np.random.normal(-0.1, 0.05, n),
        "pop": np.random.uniform(1e5, 1e7, n),
        "region": np.random.choice(["R1", "R2", "R3"], n),
        "year": np.random.randint(2000, 2010, n),
        "gdppc": np.random.uniform(1000, 50000, n)
    })

@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "test_config.yaml"
    content = """
    run:
      name: "test_run"
      output_dir: "{}"
      
    sector:
      name: "ag"
      
    data:
      dataset_dir: "{}"
      source_format: "csv"
      columns:
        y: "log_yield_impact"
        x: "temperature_anomaly"
        w: "pop"
        
    estimation:
      functional_form:
        type: "explicit"
        formula: "alpha * x + beta * x**2"
        
      global:
        method: "fixed_effects"
        
      regional:
        aggregation_level: "country"
        backend: "pandas"
        min_observations: 5
        
    execution:
      mode: "local"
      test_mode: true
    """.format(tmp_path / "results", tmp_path / "data")
    
    path.write_text(content)
    
    # Create dummy data file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return path, data_dir
