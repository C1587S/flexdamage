import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from ..config.schema import RunConfig
from ..models.symbolic import SymbolicModel

logger = logging.getLogger(__name__)

class RegionalEstimator:
    """
    Handles regional estimation (STEP2).
    Fits functional forms (e.g. polynomials) for each region.
    """
    def __init__(self, config: RunConfig, gamma: float):
        self.config = config
        self.gamma = gamma
        self.est_config = config.estimation.regional
        
        # Initialize symbolic model
        form = config.estimation.functional_form
        if form.type in ["quadratic", "explicit"]:
             # If explicit, use user formula
             # If quadratic, use default: alpha*T + beta*T^2
             if form.type == "quadratic":
                 formula = "alpha * x + beta * x**2"
                 # Map standard vars if not explicitly mapped
                 vars_map = {"x": config.data.columns.get("x", "temp")}
             else:
                 formula = form.formula
                 vars_map = config.data.columns # Use all columns mapping
                 
             self.model = SymbolicModel(formula, vars_map)
        else:
            raise NotImplementedError(f"Functional form {form.type} not yet supported")
            
    def estimate_region(self, region_id: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Estimate parameters for a single region.
        """
        # 1. Normalize outcome: y_norm = y / (exp(log_gdppc * gamma))
        # Or more generally: y_norm = y - gamma * log(gdppc) if log-linear
        # The agriculture code used: y_norm = y - gamma * x
        
        y_col = self.config.data.columns.get("y", "y")
        x_gdppc = self.config.data.columns.get("x1", "loggdppc") # Assuming x1 is log gdp
        
        # Check if we are doing log-linear adjustment or ratio adjustment
        # Step 2 original: y_norm = df["y"] - gamma * df["x"]
        
        # Calculate normalized y
        # We need to know which column is "x" (log gdppc)
        # For flexibility, let's assume one of the columns is the income variable
        # If not found, assume y is already normalized?
        
        # For now, replicate standard logic: y_norm = y - gamma * log_gdppc
        # But we need column name for log_gdppc.
        # Let's use the 'x' from config.data.columns if mapped, or 'x1'
        
        # In schema defaults: x=temp, y=yield. We need income.
        # Let's look for 'income' or 'x1' in columns
        inc_col = self.config.data.columns.get("income", self.config.data.columns.get("x1"))
        
        # Adjust outcome
        if inc_col and inc_col in df.columns:
            y_norm = df[y_col] - self.gamma * df[inc_col]
        else:
            # Fallback or warning
            y_norm = df[y_col] 
            
        # 2. Prepare Design Matrix using Symbolic Model
        try:
            X, param_names = self.model.prepare_design_matrix(df)
        except Exception as e:
            logger.warning(f"Region {region_id}: Failed to prepare design matrix: {e}")
            return None
            
        # 3. Fit OLS
        # Remove NaNs
        mask = np.isfinite(y_norm) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < self.est_config.min_observations:
            return None
            
        y_clean = y_norm[mask]
        X_clean = X[mask]
        
        # Fit
        # coeffs = (X'X)^-1 X'y
        try:
            coeffs, resid, rank, s = np.linalg.lstsq(X_clean, y_clean, rcond=None)
        except Exception:
            return None
            
        # 4. Check Constraints (e.g. Concavity: beta <= 0)
        # This assumes we know which param is 'beta'
        # Symbolic model knows param names
        
        # Naive implementation of concavity for quadratic:
        # if beta > 0, set beta=0 and re-estimate alpha
        # Generalizing this is hard without an optimizer (scipy.optimize)
        # For now, let's just implement the specific agriculture logic if param 'beta' exists
        
        params = dict(zip(param_names, coeffs))
        
        # Check specific constraint: "concavity" (beta <= 0) or "convexity" (beta >= 0) on "beta"
        beta_idx = -1
        if "beta" in param_names:
            beta_idx = param_names.index("beta")
            
        if beta_idx >= 0:
            beta_val = params["beta"]
            re_estimate = False
            
            # Check Concavity (Beta must be <= 0, if > 0 then constraint violated)
            if beta_val > 0 and any(c.type == "concavity" and c.parameter == "beta" for c in self.config.estimation.constraints):
                re_estimate = True
                
            # Check Convexity (Beta must be >= 0, if < 0 then constraint violated)
            # Reference script: if coef(mod)[3] < 0 { mod <- lm(... ~ delta_temp) ... } -> this forces linear if concave
            elif beta_val < 0 and any(c.type == "convexity" and c.parameter == "beta" for c in self.config.estimation.constraints):
                re_estimate = True
            
            if re_estimate:
                # Re-estimate with beta=0
                # Drop beta column from X
                X_constr = np.delete(X_clean, beta_idx, axis=1)
                coeffs_constr, _, _, _ = np.linalg.lstsq(X_constr, y_clean, rcond=None)
                
                # Reconstruct full params
                new_coeffs = []
                c_ptr = 0
                for i in range(len(param_names)):
                    if i == beta_idx:
                        new_coeffs.append(0.0)
                    else:
                        new_coeffs.append(coeffs_constr[c_ptr])
                        c_ptr += 1
                coeffs = np.array(new_coeffs)
                params = dict(zip(param_names, coeffs))
                
        # 5. Calculate Heteroskedasticity / Diagnostics
        # Residuals
        y_pred = X_clean @ coeffs
        residuals = y_clean - y_pred
        
        # Compute r-squared, etc.
        sst = np.sum((y_clean - y_clean.mean())**2)
        ssr = np.sum(residuals**2)
        r2 = 1 - (ssr/sst) if sst > 0 else 0
        
        # Return results
        res = {"region": region_id, "n_obs": int(mask.sum()), "rsqr": r2}
        res.update(params)
        
        return res
