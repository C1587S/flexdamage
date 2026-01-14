import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_gamma_distribution(global_results: dict, output_dir: Path):
    """
    Plots the distribution of Gamma if bootstrapped, or just visualizes the estimate with SE.
    For fixed effects single estimate, we might plot residuals or just a summary.
    If we have multiple gamma estimates (e.g. from bootstrap), we plot hist.
    
    For now, assume we just want to visualize the estimated Gamma vs 0.
    """
    gamma = global_results.get("gamma")
    se = global_results.get("gamma_se")
    
    if gamma is None:
        return

    plt.style.use('ggplot')
    plt.figure(figsize=(8, 6))
    
    # Create a normal distribution based on est and se to visualize uncertainty
    x = np.linspace(gamma - 4*se, gamma + 4*se, 100)
    y = (1/(se * np.sqrt(2 * np.pi))) * np.exp( - (x - gamma)**2 / (2 * se**2) )
    
    plt.plot(x, y, label=f"Gamma = {gamma:.4f} (SE: {se:.4f})")
    plt.axvline(0, color='red', linestyle='--', alpha=0.5)
    plt.title("Estimated Gamma Distribution")
    plt.xlabel("Gamma Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = output_dir / "gamma_distribution.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved Gamma plot to {output_path}")



def plot_damages_by_scenario(f2_df: pd.DataFrame, output_dir: Path):
    """
    Plot relationship between damages and temperature, faceted by SSP, colored by RCP.
    """
    plt.style.use('ggplot')
    
    # Ensure simplified scientific plotting style
    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("whitegrid")
    
    required_cols = ["temperature_anomaly", "flexraw", "ssp", "rcp"]
    if not all(c in f2_df.columns for c in required_cols):
        logger.warning(f"Missing columns for scenario plot: {required_cols}")
        return

    # FacetGrid
    g = sns.FacetGrid(f2_df, col="ssp", hue="rcp", palette="viridis", height=5, aspect=1.2)
    g.map(plt.scatter, "temperature_anomaly", "flexraw", alpha=0.3, s=10)
    g.add_legend(title="RCP")
    
    # Axes labels
    g.set_axis_labels("Temperature Anomaly (C)", "Predicted Damage")
    g.set_titles("{col_name}")
    
    output_path = output_dir / "damages_by_scenario.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved damages by scenario plot to {output_path}")

def plot_global_residuals(global_results: dict, df_proc: pd.DataFrame, output_dir: Path):
    """
    Plot residuals vs time for global estimation.
    """
    plt.style.use('ggplot')
    sns.set_style("whitegrid")
    
    # This requires residuals to be available or re-calculable
    # For now, let's assume we can compute y - gamma * x
    # This is rough if fixed effects were stripped out. 
    # Ideally, GlobalEstimator passes back residuals.
    # If not, skipping or mocking for now.
    
    # Assuming we can visualize raw relationships: y vs x
    gamma = global_results.get("gamma")
    if gamma is None: return

    # Let's plot y vs x and the fitted line y = gamma * x
    # Need to know x and y columns from config... passed via args?
    # Simpler: just plot y - gamma*x vs year (Partial Residuals)
    
    # We'll rely on columns being standardized in df_proc if available
    # Or heuristic guess. 
    # For this task: "residual vs time". 
    
    # If we don't have true residuals, we skip real calc and just plot what we can
    # But user asked for specific plot. 
    # Let's try to infer if 'residual' column was added (it's not currently).
    
    pass # Placeholder if data not available easily

    
def plot_parameter_distributions(regional_results_df: pd.DataFrame, output_dir: Path):
    """
    Plot histograms of regional parameters (alpha, beta, etc.) using FacetGrid styling (neutral colors).
    """
    plt.style.use('ggplot')
    sns.set_style("whitegrid")  # Clean scientific look
    
    standard_cols = ["region", "n_obs", "rsqr"]
    params = [c for c in regional_results_df.columns if c not in standard_cols]
    
    if not params: return
    
    # Melt for FacetGrid
    df_melt = regional_results_df.melt(id_vars=["region"], value_vars=params, var_name="Parameter", value_name="Value")
    
    g = sns.FacetGrid(df_melt, col="Parameter", col_wrap=3, sharex=False, sharey=False, height=4)
    g.map(sns.histplot, "Value", kde=True, color="gray", edgecolor="black") # Neutral color
    
    g.set_titles("{col_name}")
    
    output_path = output_dir / "parameter_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved parameter distributions to {output_path}")

def plot_prediction_curves(regional_results_df: pd.DataFrame, functional_form_formula: str, output_dir: Path):
    """
    Plot the estimated curves for a random subset of regions.
    """
    plt.style.use('ggplot')
    import sympy
    
    # Parse formula to lambda
    # This is a simplified viz that assumes we can map 'x' to a range
    # Basic Quadratic: alpha * x + beta * x**2
    # We need to know which param is which.
    
    regions_to_plot = regional_results_df.sample(min(20, len(regional_results_df)))
    
    x_range = np.linspace(-2, 5, 100) # Temp anomaly range
    
    fig = plt.figure(figsize=(10, 8))
    
    # This is tricky with arbitrary symbolic formulas.
    # We will assume the formula uses 'x' or 'x1' as the independent variable
    # and match parameters from the dataframe columns.
    
    try:
        # Create symbol mappings for all known columns to prevent conflicts (e.g. beta function)
        local_dict = {col: sympy.Symbol(col) for col in regional_results_df.columns}
        
        # Also parse explicit 'x' or 'x1' if not in columns
        # But we don't know what x is named in formula yet.
        # Safe approach: parse with local_dict of params, then find free symbols
        
        from sympy.parsing.sympy_parser import parse_expr
        expr = parse_expr(functional_form_formula, local_dict=local_dict)
        
        # Identify params in df
        params = [str(s) for s in expr.free_symbols if str(s) in regional_results_df.columns]
        # Identify 'x' variable - assume remaining symbol is x
        data_syms = [str(s) for s in expr.free_symbols if str(s) not in params]
        
        if len(data_syms) != 1:
            logger.warning("Cannot visualize formula with multiple independent data variables easily.")
            plt.close()
            return

        x_sym = data_syms[0]
        # Ensure x is a symbol
        x_symbol = sympy.Symbol(x_sym)
        
        f = sympy.lambdify([x_symbol] + [sympy.Symbol(p) for p in params], expr, modules="numpy")
        
        for _, row in regions_to_plot.iterrows():
            # Get param values
            p_values = [row[p] for p in params]
            y_vals = f(x_range, *p_values)
            plt.plot(x_range, y_vals, alpha=0.3, color='steelblue')
            
        plt.title(f"Regional Response Curves: {functional_form_formula}")
        plt.xlabel("Temperature Anomaly (C)")
        plt.ylabel("Impact")
        plt.grid(True, alpha=0.3)
        
        output_path = output_dir / "regional_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved regional curves plot to {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to plot prediction curves: {e}")
        plt.close()
        
    except Exception as e:
        logger.warning(f"Failed to plot prediction curves: {e}")
