import sympy
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pydantic import BaseModel

class SymbolicModel:
    """
    Parses a user-defined formula and prepares data for estimation.
    Example: "alpha * x1 + beta * x1**2"
    """
    def __init__(self, formula: str, variables: Dict[str, str]):
        self.formula_str = formula
        self.variables_map = variables  # {config_name: dataset_col}
        
        # Identify tokenizer or simply predefined override for common params
        # prevent 'beta', 'gamma', etc. from being parsed as functions
        common_params = ['alpha', 'beta', 'gamma', 'delta', 'eta', 'theta', 'zeta', 'lambda', 'mu', 'nu', 'rho', 'sigma', 'tau', 'phi', 'chi', 'psi', 'omega']
        
        local_dict = {}
        for p in common_params:
            local_dict[p] = sympy.Symbol(p)
            
        # Parse formula using local overrides
        self.expr = sympy.sympify(formula, locals=local_dict)
        
        # Extract symbols
        self.symbols = self.expr.free_symbols
        self.param_symbols = []
        self.data_symbols = []
        
        # Simple heuristic: single letters or greek names are params, others data
        # In a real system, we might be more explicit
        # For now, let's assume 'x', 'x1', 'x2', 'w', 'T' are data, others params
        # Better: checking against config variables map
        
        # Inverted map: formula_var -> dataset_col
        # Config provides: { "x": "temp", "y": "yield" }
        # Formula uses keys of variables_map
        
        known_vars = set(variables.keys())
        
        for s in self.symbols:
            if str(s) in known_vars:
                self.data_symbols.append(str(s))
            else:
                self.param_symbols.append(str(s))
                
        self.param_symbols.sort()
        self.data_symbols.sort()
        
    def get_parameter_names(self) -> List[str]:
        return self.param_symbols
        
    def prepare_design_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (X, param_names) where y ~ X @ params
        This assumes the formula is linear in parameters!
        e.g. alpha*x + beta*x^2 -> atoms are [x, x^2], params [alpha, beta]
        """
        # 1. Map dataset columns to formula variables
        eval_context = {}
        # Only map variables that are actually in the symbols
        needed_vars = set(self.data_symbols)
        
        for var_name, col_name in self.variables_map.items():
            if var_name in needed_vars:
                if col_name in df.columns:
                    eval_context[var_name] = df[col_name].values
                else:
                    raise ValueError(f"Column {col_name} mapped to {var_name} not found in data")
            elif var_name in self.param_symbols:
                 # It's a param, we don't need data for it
                 pass
                
        # 2. Decompose expression linear in parameters
        # expr = p1*term1 + p2*term2 + ...
        # We need to extract term_i for each p_i
        
        coeffs = self.expr.as_coefficients_dict() 
        # as_coefficients_dict returns {term: coeff}
        # But for "alpha*x", alpha is a symbol, not a number.
        # So we use .collect() or .coeff()
        
        X_cols = []
        
        for param in self.param_symbols:
            p_sym = sympy.Symbol(param)
            term = self.expr.coeff(p_sym)
            
            if term == 0:
                # Parameter not found in expression (linear dependence assumption)
                # Maybe inside a non-linear function? 
                # For this version, we require linearity in parameters.
                raise ValueError(f"Formula {self.formula_str} must be linear in parameter {param}")
            
            # Evaluate term with data
            # Use sympy.lambdify for speed
            f = sympy.lambdify(self.data_symbols, term, modules="numpy")
            
            # Prepare args based on data_symbols order
            args = [eval_context[s] for s in self.data_symbols]
            
            try:
                col_values = f(*args)
                
                # Broadcast scalar if needed (e.g. intercept term derived from coeff 1)
                if np.isscalar(col_values):
                    col_values = np.full(len(df), col_values)
                    
                X_cols.append(col_values)
                
            except Exception as e:
                raise ValueError(f"Failed to evaluate term for {param}: {e}")
                
        X = np.column_stack(X_cols)
        return X, self.param_symbols

    def evaluate(self, df: pd.DataFrame, params: Dict[str, float]) -> np.ndarray:
        """Evaluate full formula with given parameters"""
        # Map variables
        eval_context = {}
        for var_name, col_name in self.variables_map.items():
            eval_context[var_name] = df[col_name].values
            
        # Add params to context
        eval_context.update(params)
        
        # Lambdify full expression
        all_syms = self.data_symbols + self.param_symbols
        f = sympy.lambdify(all_syms, self.expr, modules="numpy")
        
        args = [eval_context[s] for s in all_syms]
        return f(*args)
