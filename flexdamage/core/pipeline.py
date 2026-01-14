import pandas as pd
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

from ..config.schema import RunConfig
from ..data.backends import PandasBackend, DuckDBBackend
from .global_est import GlobalEstimator
from .regional import RegionalEstimator
from ..utils.monitoring import ResourceMonitor
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)

class EstimationPipeline:
    """
    Orchestrates the full estimation workflow:
    1. Load Data
    2. Global Estimation (Gamma)
    3. Regional Estimation (Alphas/Betas)
    4. Save Results
    """
    def __init__(self, config: RunConfig):
        self.config = config
        self.output_dir = Path(config.run.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        logger.info(f"Starting Pipeline: {self.config.run.name}")
        ResourceMonitor.log_usage("Start")
        
        # 1. Initialize Backend & Load Data
        backend = self._init_backend()
        
        # Load main dataset
        # For simple workflows, load everything. For large data, we might load lazily.
        # But Global Estimation usually needs full dataset (or a large sample).
        # Regional needs iteration.
        
        # Let's load the columns needed for Global Est first
        global_cols = self._get_global_columns()
        
        logger.info("Loading data for Global Estimation...")
        # If test mode, limits sample
        sample_size = self.config.execution.test_sample_size if self.config.execution.test_mode else None
        
        df_global = backend.load_data(
            columns=global_cols,
            sample_size=sample_size,
            random_seed=self.config.execution.test_seed
        )
        logger.info(f"Data Loaded: {df_global.shape}")
        ResourceMonitor.log_usage("Data Loaded")
        
        # 2. Preprocessing (Transformations & Aggregation)
        from ..data.preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor(self.config.data)
        df_global = preprocessor.process(df_global)
        
        ResourceMonitor.log_usage("Data Preprocessed")
        
        # 3. Global Estimation
        global_est = GlobalEstimator(self.config)
        global_results = global_est.estimate(df_global)
        
        # Save Global Results
        self._save_json(global_results, "global_results.json")
        gamma = global_results.get("gamma", 0.0)
        
        # 3. Regional Estimation
        logger.info("Starting Regional Estimation...")
        reg_est = RegionalEstimator(self.config, gamma=gamma)
        
        # We need to iterate over regions.
        # If dataset is small (in-memory), we can just groupby.
        # If large (DuckDB), we might iterate distinct regions and query.
        
        regions = backend.get_unique_values(column=self.config.data.columns.get("region", "region"))
        logger.info(f"Found {len(regions)} regions")
        
        # For efficiency, if using Pandas backend and df_global has everything, use it.
        # But df_global might be a sample. 
        # For regional, we typically need ALL data for that region.
        
        # If test mode, we might just run on the sampled regions in df_global
        if self.config.execution.test_mode:
            target_regions = df_global[self.config.data.columns.get("region", "region")].unique()
            df_source = df_global
        else:
            target_regions = regions
            df_source = None # Will force reload/query per region if backend requires
            
        regional_results = []
        reg_col = self.config.data.columns.get("region", "region")
        
        for region in target_regions:
            # Get data for region
            if df_source is not None:
                df_reg = df_source[df_source[reg_col] == region]
            else:
                # Query backend for specific region
                # We need all columns for regional est (including income, temp, etc.)
                all_cols = self._get_regional_columns()
                df_reg = backend.load_data(
                    columns=all_cols,
                    filters={reg_col: region}
                )
            
            if df_reg.empty:
                continue
                
            res = reg_est.estimate_region(region, df_reg)
            if res:
                regional_results.append(res)
                
        # 4. Save Regional Results
        if regional_results:
            df_results = pd.DataFrame(regional_results)
            out_path = self.output_dir / "regional_results.csv"
            df_results.to_csv(out_path, index=False)
            logger.info(f"Saved regional results to {out_path}")
            
            # --- Diagnostics ---
            logger.info("Running Diagnostics...")
            from ..diagnostics.plots import plot_gamma_distribution, plot_parameter_distributions, plot_prediction_curves
            diag_dir = self.output_dir / "diagnostics"
            diag_dir.mkdir(exist_ok=True)
            
            plot_gamma_distribution(global_results, diag_dir)
            plot_parameter_distributions(df_results, diag_dir)
            plot_prediction_curves(df_results, self.config.estimation.functional_form.formula or "alpha*x+beta*x**2", diag_dir)
            
            # --- F2 Generation ---
            # Ideally we load a separate projections dataset. 
            # For this "demo" pipeline, we will use the df_global (or reg data) as "future" data just to prove it works
            # In production, config should point to projection files.
            logger.info("Generating F2 Tables...")
            from .f2_gen import F2Generator
            f2_gen = F2Generator(self.config, df_results)
            f2_dir = self.output_dir / "f2"
            f2_dir.mkdir(exist_ok=True)
            
            # Use df_global as dummy projection input if available, else skip
            if not df_global.empty:
                # Ensure it has 'x' variable mapping
                f2_gen.generate(df_global, f2_dir)
                
        else:
            logger.warning("No regional results produced")
            
        logger.info("Pipeline Completed Successfully")
        ResourceMonitor.log_usage("End")

    def _init_backend(self):
        backend_type = self.config.estimation.regional.backend
        
        if backend_type == "auto":
            # Heuristic: if country level, pandas. If impact_region, duckdb?
            # For now default to pandas for simplicity unless DB specified
             backend_type = "pandas"
             
        if backend_type == "pandas":
            # Load from dataset_dir
            # Assume CSV for now based on example config
            # In real system, logic to find files in dir
            path = Path(self.config.data.dataset_dir)
            if path.is_file():
                return PandasBackend.from_csv(str(path))
            elif path.is_dir():
                # Look for file matching table_name or just one csv
                files = list(path.glob("*.csv"))
                if files:
                    return PandasBackend.from_csv(str(files[0]))
                else:
                    raise ValueError(f"No CSV files found in {path}")
        if backend_type == "duckdb":
            return DuckDBBackend(
                self.config.data.dataset_dir, # This assumes db file path
                self.config.data.table_name
            )
            
        raise ValueError(f"Unknown or improperly configured backend: {backend_type}")


    def _get_global_columns(self):
        # Gather all columns mapped in data config needed for global est
        # x, y, w, region, year, maybe temp binning cols
        c = self.config.data.columns
        cols = set([c.get("y"), c.get("x1"), c.get("x2"), c.get("w"), c.get("x"), c.get("region", "region"), c.get("year", "year")])
        # Add others explicitly needed?
        return [cl for cl in cols if cl]

    def _get_regional_columns(self):
        # Similar to global but maybe all mapped cols
        c = self.config.data.columns
        return list(set(c.values()) | {c.get("region", "region"), c.get("year", "year")})

    def _save_json(self, data: Dict, filename: str):
        with open(self.output_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)
