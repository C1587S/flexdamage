import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, Optional
import logging
from ..config.schema import EstimationConfig, RunConfig
from ..models.symbolic import SymbolicModel

logger = logging.getLogger(__name__)

class GlobalEstimator:
    """
    Handles global parameter estimation (e.g. Gamma for income elasticity).
    """
    def __init__(self, config: RunConfig):
        self.config = config
        self.est_config = config.estimation.global_est
        
    def estimate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run global estimation pipeline.
        Returns dictionary of estimated parameters (e.g. {'gamma': 0.123, 'gamma_se': 0.01})
        """
        logger.info(f"Starting Global Estimation (Method: {self.config.estimation.global_est.method})")
        
        # 1. Coordinate transformations if needed (e.g. centering)
        # This mirrors STEP1 logic: 
        # y = log_yield_impact, x = log(gdppc)
        # FE: y_it - y_bar_i - y_bar_t ...
        
        if self.est_config.method == "fixed_effects":
            return self._estimate_fe(df)
        elif self.est_config.method == "ols":
            return self._estimate_ols(df)
        else:
            raise NotImplementedError(f"Method {self.est_config.method} not implemented")

    def _estimate_fe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fixed Effects estimation (Within estimator).
        Mirrors STEP1 logic for Gamma estimation.
        """
        # Map columns
        y_col = self.config.data.columns.get("y", "y")
        x_col = self.config.data.columns.get("x1", self.config.data.columns.get("x")) # Usually gdppc
        w_col = self.config.data.columns.get("w", "pop")
        reg_col = self.config.data.columns.get("region", "region")
        
        # Temp bins for FE groups (Region x TempBin)
        temp_col = self.config.data.columns.get("x2", "temp") # If used for binning
        
        # For agriculture/gamma, we typically regress y on x controlling for FE
        # Model: y_it = gamma * x_it + mu_i + theta_t + eps_it
        # Where mu_i is actually Region x TempBin specific
        
        # Create groups
        bin_width = self.est_config.temperature_bins
        if temp_col in df.columns:
            df["_temp_bin"] = np.floor(df[temp_col] / bin_width)
            df["_grp"] = df[reg_col].astype(str) + "_" + df["_temp_bin"].astype(str)
        else:
            df["_grp"] = df[reg_col].astype(str)
            
        # De-mean (Within transformation)
        # Fast groupby transform
        for col in [y_col, x_col]:
            # Region-Bin FE
            df[f"_{col}_demean"] = df[col] - df.groupby("_grp")[col].transform("mean")
            # Year FE
            df[f"_{col}_demean"] = df[f"_{col}_demean"] - df.groupby("year")[f"_{col}_demean"].transform("mean")
            
        # Weighted Least Squares on de-meaned data
        y = df[f"_{y_col}_demean"]
        X = sm.add_constant(df[f"_{x_col}_demean"])
        
        if w_col in df.columns:
            weights = df[w_col]
        else:
            weights = 1.0
            
        mod = sm.WLS(y, X, weights=weights).fit()
        
        # Extract Gamma (slope of x)
        # Params: [const, x_col]
        gamma = mod.params[1]
        gamma_se = mod.bse[1]
        
        logger.info(f"Estimated Gamma: {gamma:.6f} (SE: {gamma_se:.6f})")
        
        # Capture residuals for diagnostics
        # mod.resid index matches the data used (df)
        df["residuals"] = mod.resid
        
        # Prepare diagnostic dataframe (Year, Residuals, + Categorical cols if present)
        diag_cols = ["year", "residuals"]
        for cat_col in ["ssp", "rcp", "model"]:
            if cat_col in df.columns:
                diag_cols.append(cat_col)
                
        # Drop duplicates if multiple points per year/region? 
        # Regression is on (Region, Year) level usually.
        # Just return the relevant columns.
        diag_df = df[diag_cols].copy()
        
        return {
            "gamma": gamma,
            "gamma_se": gamma_se,
            "r_squared": mod.rsquared,
            "n_obs": int(mod.nobs),
            "diagnostics": diag_df.to_dict(orient="list") # return as list dict for easy serialization/reconstruction
        }

    def _estimate_ols(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple OLS for testing or alternative specs"""
        y_col = self.config.data.columns.get("y")
        x_col = self.config.data.columns.get("x1", self.config.data.columns.get("x"))
        
        X = sm.add_constant(df[x_col])
        mod = sm.OLS(df[y_col], X).fit()
        
        return {
            "gamma": mod.params[x_col],
            "gamma_se": mod.bse[x_col],
            "r_squared": mod.rsquared
        }
