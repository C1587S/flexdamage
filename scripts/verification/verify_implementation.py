import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add repo to path
sys.path.append("./flexdamage-dev")

from flexdamage.config.loader import load_config
from flexdamage.models.symbolic import SymbolicModel
from flexdamage.data.backends import PandasBackend
from flexdamage.core.regional import RegionalEstimator
from flexdamage.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger("verification")

def create_dummy_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "temperature_anomaly": np.random.normal(1.5, 0.5, n),
        "log_yield_impact": np.random.normal(-0.1, 0.05, n),
        "pop": np.random.uniform(1e5, 1e7, n),
        "gdppc": np.random.lognormal(9, 1, n),
        "region": ["test_region"] * n
    })
    return df

def run_tests():
    logger.info("1. Loading Config...")
    config_path = "./flexdamage-dev/configs/examples/custom_formula.yaml"
    try:
        cfg = load_config(config_path)
        logger.info("Config loaded successfully")
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        return

    logger.info("2. Testing Symbolic Model...")
    try:
        # formula: alpha * x1 + beta * x1**2
        # x1 mapped to temperature_anomaly
        model = SymbolicModel(
            cfg.estimation.functional_form.formula,
            {"x1": "temperature_anomaly"}
        )
        logger.info(f"Parsed params: {model.get_parameter_names()}")
        
        df = create_dummy_data()
        X, params = model.prepare_design_matrix(df)
        logger.info(f"Design matrix shape: {X.shape}")
        if X.shape == (100, 2):
             logger.info("Design matrix correct")
        else:
             logger.error(f"Design matrix shape mismatch: {X.shape}")
    except Exception as e:
        logger.error(f"Symbolic model failed: {e}")
        return

    logger.info("3. Testing Regional Estimation...")
    try:
        est = RegionalEstimator(cfg, gamma=0.0) # Assume gamma=0 for test
        res = est.estimate_region("test_region", df)
        if res:
             logger.info(f"Estimation result: {res}")
        else:
             logger.error("Estimation returned None")
    except Exception as e:
        logger.error(f"Estimation failed: {e}")

if __name__ == "__main__":
    run_tests()
