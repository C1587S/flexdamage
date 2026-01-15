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

def _run_single_region(args):
    """
    Helper for parallel execution.
    args: (region, df_reg, estimator_config, gamma, data_columns, functional_form_type, functional_form_formula, constraints)
    Since we cannot pickle the entire estimator easily if it has open connections or complex state,
    we re-instantiate a lightweight estimator or pass necessary data.
    Actually, RegionalEstimator is lightweight if initialized.
    Better: pass (region, df_reg, estimator_instance) if picklable. 
    But estimator has config which might be large? No, config is Pydantic.
    Let's rely on pickling the estimator.
    """
    region, df_reg, estimator = args
    if df_reg.empty:
        return None
    try:
        return estimator.estimate_region(region, df_reg)
    except Exception as e:
        logger.error(f"Error extracting region {region}: {e}")
        return None

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
        
        # Parallel Execution
        n_workers = self.config.execution.n_workers
        
        if n_workers > 1:
            logger.info(f"Parallelizing regional estimation with {n_workers} workers")
            from concurrent.futures import ProcessPoolExecutor
            
            # Prepare tasks
            tasks = []
            for region in target_regions:
                if df_source is not None:
                    df_reg = df_source[df_source[reg_col] == region]
                else:
                    # NOTE: Parallelizing backend queries might be bad for some backends (DuckDB concurrency)
                    # Ideally we preload data or backend handles it.
                    # For now assume backend is thread-safe or we are using Pandas
                    all_cols = self._get_regional_columns()
                    df_reg = backend.load_data(columns=all_cols, filters={reg_col: region})
                
                if not df_reg.empty:
                    tasks.append((region, df_reg, reg_est))
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = executor.map(_run_single_region, tasks)
                
            regional_results = [r for r in results if r is not None]
            
        else:
            # Serial Execution
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
            
            # --- Advanced Diagnostics (Visualizer) ---
            from ..diagnostics.visualizer import DiagnosticVisualizer
            visualizer = DiagnosticVisualizer(diag_dir)
            
            # Parameter Distributions (Refined)
            visualizer.plot_parameter_distributions(df_results)
            
            # Worst Offenders (Detailed)
            # Strategy: identify regions, load data, plot
            # Criteria: 'eta' (largest residuals/noise) is a good default for "worst fit"
            worst_regions = visualizer.select_worst_regions(df_results, criteria="eta", n=9)
            
            if worst_regions and not df_global.empty:
               # Try to filter from loaded global data first
               # Check if global data has 'region' column
               reg_col = self.config.data.columns.get("region", "region")
               if reg_col in df_global.columns:
                   df_worst = df_global[df_global[reg_col].isin(worst_regions)]
                   
                   # Map columns for plotting
                   x_col = self.config.data.columns.get("x1") or self.config.data.columns.get("x")
                   y_col = self.config.data.columns.get("y")
                   
                   
                   if x_col and y_col:
                       # Call Enhanced Plot
                       # Determine color column: preferentially GDP-like if available, else Pop
                       color_col = None
                       potential_cols = ["lgdp_delta", "loggdppc", self.config.data.columns.get("w")]
                       for c in potential_cols:
                           if c and c in df_worst.columns:
                               color_col = c
                               break
                               
                       visualizer.plot_fit_diagnostics(
                           df_results=df_results, 
                           df_data=df_worst, 
                           x_col=x_col, 
                           y_col=y_col,
                           gamma=gamma,
                           color_col=color_col,
                           regions=worst_regions.tolist() if hasattr(worst_regions, 'tolist') else worst_regions,
                           n_cols=5
                       )
            
            # --- Legacy / Alternative Diagnostics ---
            from ..diagnostics.advanced import plot_spaghetti_curves, analyze_zero_crossings
            plot_spaghetti_curves(df_results, self.config.estimation.functional_form.formula or "alpha*x+beta*x**2", diag_dir)
            analyze_zero_crossings(df_results, diag_dir)
            
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
            
        if backend_type == "polars":
            from ..data.backends import PolarsBackend
            # Similar file finding logic as pandas
            path = Path(self.config.data.dataset_dir)
            if path.is_file():
                return PolarsBackend.from_csv(str(path))
            elif path.is_dir():
                files = list(path.glob("*.csv"))
                if files:
                    return PolarsBackend.from_csv(str(files[0]))
                else:
                    raise ValueError(f"No CSV files found in {path}")
            
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
