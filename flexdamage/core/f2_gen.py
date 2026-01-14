import pandas as pd
import numpy as np
import logging
from pathlib import Path
from ..config.schema import RunConfig
from ..models.symbolic import SymbolicModel

logger = logging.getLogger(__name__)

class F2Generator:
    """
    Generates F2 tables: Projections of damages using estimated parameters 
    applied to future climate scenarios (SSP/RCP).
    Outputs columns: 'flexraw', 'rawtotal'.
    """
    def __init__(self, config: RunConfig, regional_params: pd.DataFrame):
        self.config = config
        self.regional_params = regional_params
        
    def generate(self, projections_df: pd.DataFrame, output_dir: Path):
        """
        Apply estimated parameters to projection data.
        projections_df must contain:
        - region
        - year
        - scenario (optional, if multiple)
        - variables needed for formula (e.g. temp)
        """
        logger.info("Generating F2 Tables (Projections)...")
        
        # Merge parameters into projections
        # Inner join to only predict for regions we have estimates for
        # Projections DF (left) uses the mapped column name (e.g. "iso3")
        # Regional Params DF (right) uses standardized "region" column
        reg_col = self.config.data.columns.get("region", "region")
        
        # Check if right side has the mapped name or "region"
        right_col = "region"
        if "region" not in self.regional_params.columns and reg_col in self.regional_params.columns:
            right_col = reg_col
            
        merged = pd.merge(projections_df, self.regional_params, left_on=reg_col, right_on=right_col, how="inner")
        
        if merged.empty:
            logger.warning("No overlapping regions between projections and estimates.")
            return

        # Prepare symbolic model for evaluation
        form_config = self.config.estimation.functional_form
        formula = form_config.formula
        if not formula and form_config.type == "quadratic":
             formula = "alpha * x + beta * x**2"
        
        # We need to map projection columns to formula vars
        # This mapping might differ from estimation if projection cols have different names
        # For now, assume same column names or utilize existing mapping config
        
        # Variable map: {config_alias: col_name}
        # e.g. {"x": "temperature_anomaly"}
        
        # Initialize model just for evaluation helper
        model = SymbolicModel(formula, self.config.data.columns)
        
        # Identify parameter columns in merged df
        param_cols = model.get_parameter_names()
        
        # Evaluate formula per row
        # Optimization: Vectors!
        
        # We can construct a dict of {symbol: series}
        eval_context = {}
        
        # Data variables
        for var_name in model.data_symbols:
            col_name = self.config.data.columns.get(var_name)
            if col_name and col_name in merged.columns:
                eval_context[var_name] = merged[col_name].values
            else:
                 # Fallback: maybe var_name itself is the col name
                 if var_name in merged.columns:
                     eval_context[var_name] = merged[var_name].values
                 else:
                     logger.error(f"Missing data column for variable '{var_name}' in projections.")
                     return

        # Parameter variables
        for param in model.param_symbols:
            if param in merged.columns:
                eval_context[param] = merged[param].values
            else:
                logger.error(f"Missing parameter column '{param}' in merged data.")
                return
                
        # Evaluate
        import sympy
        # Use simple lambda for vectorization
        # Caution: symbolic.py has an evaluate method but it expects params as dict of scalars if generic
        # faster to reimplement vector eval here
        
        all_syms = list(eval_context.keys())
        f = sympy.lambdify(all_syms, model.expr, modules="numpy")
        args = [eval_context[s] for s in all_syms]
        
        try:
            merged["flexraw"] = f(*args)
            # rawtotal is usually the same or just the un-referenced version.
            # Mirroring structure:
            merged["rawtotal"] = merged["flexraw"]
        except Exception as e:
            logger.error(f"Projection calculation failed: {e}")
            return
            
        # Save results
        # Split by scenario if exists
        if "scenario" in merged.columns:
            for scenario, group in merged.groupby("scenario"):
                clean_scen = str(scenario).replace("/", "_")
                out = output_dir / f"f2_{clean_scen}.csv"
                group.to_csv(out, index=False)
                logger.info(f"Saved F2 table to {out}")
        else:
            out = output_dir / "f2_projections.csv"
            merged.to_csv(out, index=False)
            logger.info(f"Saved F2 table to {out}")

