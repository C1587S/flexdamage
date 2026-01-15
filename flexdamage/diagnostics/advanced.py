
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

def set_style():
    plt.style.use('ggplot')
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette("colorblind")

def plot_spaghetti_curves(
    df_results: pd.DataFrame, 
    formula: str, 
    output_dir: Path,
    x_range: tuple = (-5, 10),
    n_points: int = 100,
    highlight_regions: Optional[List[str]] = None
):
    """
    Plot all regional curves on a single plot to show distribution.
    """
    set_style()
    
    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Parse formula coarsely if simple
    # Assumes alpha*x + beta*x**2 structure if params present
    # Or uses eval with dict
    
    plotted = 0
    for _, row in df_results.iterrows():
        params = row.to_dict()
        # Evaluate y
        # Basic quadratic fallback
        alpha = params.get("alpha", 0)
        beta = params.get("beta", 0)
        
        y_vals = alpha * x_vals + beta * (x_vals**2)
        
        # Color specific or neutral
        if highlight_regions is not None and row.get("region") in highlight_regions:
            color = "red"
            alpha_val = 1.0
            lw = 2
            zorder = 10
        else:
            color = "gray"
            alpha_val = 0.1
            lw = 1
            zorder = 1
            
        ax.plot(x_vals, y_vals, color=color, alpha=alpha_val, linewidth=lw, zorder=zorder)
        plotted += 1
        
    ax.set_xlabel("Temperature Anomaly (C)")
    ax.set_ylabel("Damage (Impact)")
    ax.set_title(f"Regional Damage Functions (N={plotted})")
    ax.axhline(0, color="black", linestyle="--")
    ax.axvline(0, color="black", linestyle="--")
    
    out_path = output_dir / "spaghetti_curves.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved spaghetti plot to {out_path}")

def plot_worst_offenders(
    df_results: pd.DataFrame,
    output_dir: Path,
    criteria: str = "max_slope",
    n: int = 9
):
    """
    Identify and plot 'worst' regions based on criteria (e.g. extreme slopes).
    """
    set_style()
    
    df = df_results.copy()
    
    # Calculate metric
    # E.g. slope at T=5
    # slope = alpha + 2*beta*T
    T_eval = 5
    if "alpha" in df.columns and "beta" in df.columns:
        df["slope_at_5"] = df["alpha"] + 2 * df["beta"] * T_eval
        df["abs_slope"] = df["slope_at_5"].abs()
        
        # Sort by metric
        offenders = df.sort_values("abs_slope", ascending=False).head(n)
        
        # Plot FacetGrid of these regions
        # We need to generate curve data for each
        plot_data = []
        x_vals = np.linspace(-2, 8, 50)
        
        for _, row in offenders.iterrows():
            alpha = row["alpha"]
            beta = row["beta"]
            region = row["region"]
            y_vals = alpha * x_vals + beta * (x_vals**2)
            
            tmp_df = pd.DataFrame({"temp": x_vals, "damage": y_vals})
            tmp_df["region"] = region
            plot_data.append(tmp_df)
            
        if not plot_data:
            return

        plot_df = pd.concat(plot_data)
        
        g = sns.FacetGrid(plot_df, col="region", col_wrap=3, sharey=False)
        g.map(plt.plot, "temp", "damage")
        g.map(plt.axhline, y=0, color="k", linestyle="--")
        g.set_axis_labels("Temp Anomaly", "Damage")
        g.fig.suptitle(f"Top {n} Deviant Regions (Slope at 5C)", y=1.02)
        
        out_path = output_dir / "worst_offenders.png"
        g.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(g.fig)
        logger.info(f"Saved worst offenders plot to {out_path}")

def analyze_zero_crossings(df_results: pd.DataFrame, output_dir: Path):
    """
    Histogram of where curves cross zero (roots).
    alpha*x + beta*x^2 = 0 => x(alpha + beta*x) = 0
    Roots: x=0, x = -alpha/beta
    """
    set_style()
    
    roots = []
    for _, row in df_results.iterrows():
        alpha = row.get("alpha")
        beta = row.get("beta")
        
        if beta and abs(beta) > 1e-9:
            root = -alpha / beta
            # Filter reasonable range
            if -10 < root < 20: 
                roots.append(root)
                
    if roots:
        fig, ax = plt.subplots()
        sns.histplot(roots, kde=True, ax=ax)
        ax.set_title("Distribution of Zero Crossings (Non-zero root)")
        ax.set_xlabel("Temperature (C)")
        
        out_path = output_dir / "zero_crossings.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved zero crossings plot to {out_path}")
