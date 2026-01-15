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
        
        # 4. Apply Constraints
        coeffs = self._apply_constraints(X_clean, y_clean, param_names, coeffs)
        params = dict(zip(param_names, coeffs))
                
        # 5. Calculate Heteroskedasticity / Diagnostics
        # Residuals
        y_pred = X_clean @ coeffs
        residuals = y_clean - y_pred
        
        # Calculate standard metrics
        sst = np.sum((y_clean - y_clean.mean())**2)
        ssr = np.sum(residuals**2)
        rsqr1 = 1 - (ssr/sst) if sst > 0 else 0
        
        # Calculate s2 (variance of residuals)
        dof = len(y_clean) - len(coeffs)
        s2 = ssr / dof if dof > 0 else 0
        
        # --- Advanced R-script Metrics (zeta, eta) ---
        # "mod2 <- lm(totalsd_scaled ~ 0 + delta_temp, data = subdf)"
        # totalsd_scaled <- sqrt(subdf$resids^2) (which is just abs(residuals))
        # We regression abs(residuals) on temperature (column 0 of X is usually constant? No, X is symbolic)
        # We need the 'temperature' column specifically. 
        # In symbolic design matrix, we might have 'x', 'x^2'. 
        # We need 'x' (or 'delta_temp' equivalent).
        # Let's try to extract 'x' column from design matrix if possible, or use df column using vars_map.
        
        # Assume 'x' in vars_map is temperature
        temp_col = self.config.data.columns.get("x1") or self.config.data.columns.get("x")
        
        zeta = np.nan
        eta = np.nan
        rsqr2 = np.nan
        
        if temp_col and temp_col in df.columns:
            # Re-extract temp corresponding to the cleaned mask
            # Note: X_clean and y_clean used mask. We need temp for same rows.
            temp_vals = df.loc[df.index[mask], temp_col].values
            
            # Target for mod2: abs(residuals)
            dependent_var = np.abs(residuals)
            
            # Regressor: 0 + delta_temp (just temp, no intercept)
            # Reshape for OLS
            X_mod2 = temp_vals.reshape(-1, 1)
            
            try:
                # fit mod2
                # coeff, resid, rank, s
                c2, res2, _, _ = np.linalg.lstsq(X_mod2, dependent_var, rcond=None)
                zeta = c2[0]
                
                # eta = sd(residuals(mod2))
                # residuals of mod2
                pred2 = X_mod2 @ c2
                resid2 = dependent_var - pred2
                eta = np.std(resid2) # using population or sample sd? R usually sample
                eta = np.std(resid2, ddof=1) if len(resid2) > 1 else 0
                
                # rsqr2
                sst2 = np.sum((dependent_var - dependent_var.mean())**2)
                ssr2 = np.sum(resid2**2)
                rsqr2 = 1 - (ssr2/sst2) if sst2 > 0 else 0
                
            except Exception:
                pass

        # Return results
        # 'n' is n_obs
        res = {
            "region": region_id, 
            "n": int(mask.sum()), 
            "rsqr1": rsqr1,
            "rsqr2": rsqr2,
            "s2": s2,
            "zeta": zeta,
            "eta": eta,
            # Placeholder for rho (requires global residuals correlation, hard to do inside single region estimation without global context passed in)
            "rho": 0.0 
        }
        res.update(params)
        
        return res

    def _apply_constraints(self, X, y, param_names, initial_coeffs):
        """
        Check and enforce constraints. If violated, re-estimate with fixed parameter.
        Supports: convexity, concavity, and explicit formulas (e.g. "beta >= 0").
        """
        current_coeffs = initial_coeffs.copy()
        params = dict(zip(param_names, current_coeffs))
        
        import re

        for constraint in self.config.estimation.constraints:
            violated = False
            fix_val = 0.0
            param_to_fix = None
            
            # 1. Pre-defined types
            if constraint.type == "convexity":
                # beta >= 0 (default)
                p = constraint.parameter or "beta"
                if p in params and params[p] < 0:
                    violated = True
                    param_to_fix = p
                    fix_val = 0.0
                    
            elif constraint.type == "concavity":
                # beta <= 0 (default)
                p = constraint.parameter or "beta"
                if p in params and params[p] > 0:
                    violated = True
                    param_to_fix = p
                    fix_val = 0.0
            
            # 2. Explicit Formula (e.g. "beta >= 0", "alpha <= 1.5")
            elif constraint.type == "formula" and constraint.expression:
                expr = constraint.expression.strip()
                # Parse "param >= val" or "param <= val"
                # Regex for: param, op, val (float)
                match = re.match(r"^([a-zA-Z0-9_]+)\s*(>=|<=|>|<)\s*([-+]?[0-9]*\.?[0-9]+)$", expr)
                if match:
                    p, op, val_str = match.groups()
                    thresh = float(val_str)
                    
                    if p in params:
                        val = params[p]
                        if op == ">=" and val < thresh:
                            violated = True
                            param_to_fix = p
                            fix_val = thresh
                        elif op == ">" and val <= thresh:
                            violated = True
                            param_to_fix = p
                            fix_val = thresh # Boundary solution
                        elif op == "<=" and val > thresh:
                            violated = True
                            param_to_fix = p
                            fix_val = thresh
                        elif op == "<" and val >= thresh:
                            violated = True
                            param_to_fix = p
                            fix_val = thresh # Boundary solution
                else:
                    logger.warning(f"Could not parse constraint formula: {expr}")

            if violated and param_to_fix is not None:
                logger.info(f"Constraint violated: {param_to_fix} ({params[param_to_fix]:.4f}) violates {constraint.type}/{constraint.expression}. Fixing to {fix_val}.")
                return self._refit_with_fixed_param(X, y, param_names, param_to_fix, fix_val)
                
        return current_coeffs

    def _refit_with_fixed_param(self, X, y, param_names, param_to_fix, fix_val):
        """
        Re-estimate OLS with one parameter fixed to a specific value.
        y_adj = y - fix_val * X[:, idx]
        Fit y_adj ~ X_reduced
        """
        try:
            idx = param_names.index(param_to_fix)
        except ValueError:
            return np.zeros(len(param_names)) # Should not happen

        # Adjust y
        y_adj = y - fix_val * X[:, idx]
        
        # Remove column from X
        X_reduced = np.delete(X, idx, axis=1)
        
        # Refit
        coeffs_constr, _, _, _ = np.linalg.lstsq(X_reduced, y_adj, rcond=None)
        
        # Reconstruct full coefficients list
        new_coeffs = []
        c_ptr = 0
        for i in range(len(param_names)):
            if i == idx:
                new_coeffs.append(fix_val)
            else:
                new_coeffs.append(coeffs_constr[c_ptr])
                c_ptr += 1
                
        return np.array(new_coeffs)
