
import json
from pathlib import Path

def refactor_notebook():
    nb_path = Path("notebooks/tutorial_walkthrough.ipynb")
    with open(nb_path, "r") as f:
        nb = json.load(f)

    # USER-REQUESTED REPLACEMENTS

    # 1. Gamma Distribution (New R-Style)
    md_gamma = {
        "cell_type": "markdown",
        "id": "diag_gamma_md",
        "metadata": {},
        "source": ["### 6b.5 Gamma Distribution (R-Style)\n", "Global gamma parameters with R-style plot."]
    }
    code_gamma = {
        "cell_type": "code",
        "execution_count": None,
        "id": "diag_gamma_code_new",
        "metadata": {},
        "outputs": [],
        "source": [
            "visualizer = DiagnosticVisualizer(diag_dir, style=\"seaborn-v0_8-paper\")\n",
            "visualizer.plot_gamma_distribution(global_results)\n",
            "if (diag_dir / \"gamma_distribution_rstyle.png\").exists():\n",
            "    display(Image(filename=diag_dir / \"gamma_distribution_rstyle.png\"))"
        ]
    }

    # 2. Parameter Distributions (New R-Style)
    md_dist = {
        "cell_type": "markdown",
        "id": "diag_dist_md",
        "metadata": {},
        "source": ["### 6b.6 Parameter Distributions (R-Style)\n", "Histograms of regional coefficients and metrics with mean/median lines."]
    }
    code_dist = {
        "cell_type": "code",
        "execution_count": None,
        "id": "diag_dist_code_new",
        "metadata": {},
        "outputs": [],
        "source": [
            "visualizer.plot_parameter_distributions(df_results)\n",
            "if (diag_dir / \"01_param_distributions.png\").exists():\n",
            "    display(Image(filename=diag_dir / \"01_param_distributions.png\"))"
        ]
    }

    # 3. New R-Style Metrics
    md_poly = {
        "cell_type": "markdown",
        "id": "diag_poly_md",
        "metadata": {},
        "source": ["### 6b.7 Polynomial Summary Curve\n", "Mean damage function with IQR shading."]
    }
    code_poly = {
        "cell_type": "code",
        "execution_count": None,
        "id": "diag_poly_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "visualizer.plot_polynomial_summary(df_results, t_range=(0, 20))\n",
            "if (diag_dir / \"02_polynomial_summary.png\").exists():\n",
            "    display(Image(filename=diag_dir / \"02_polynomial_summary.png\"))"
        ]
    }

    md_cross = {
        "cell_type": "markdown",
        "id": "diag_cross_md",
        "metadata": {},
        "source": ["### 6b.8 Crossing Zero (Convex Regions)\n", "Temperature where damage crosses zero for convex curves."]
    }
    code_cross = {
        "cell_type": "code",
        "execution_count": None,
        "id": "diag_cross_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "visualizer.plot_crossing_zero(df_results)\n",
            "if (diag_dir / \"03_crossing_zero.png\").exists():\n",
            "    display(Image(filename=diag_dir / \"03_crossing_zero.png\"))"
        ]
    }

    md_slope = {
        "cell_type": "markdown",
        "id": "diag_slope_md",
        "metadata": {},
        "source": ["### 6b.9 Maximum Slope\n", "Distribution of max slope in [0, 10]C range."]
    }
    code_slope = {
        "cell_type": "code",
        "execution_count": None,
        "id": "diag_slope_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "visualizer.plot_max_slope(df_results, t_range=(0, 10))\n",
            "if (diag_dir / \"04_maxslope_hist.png\").exists():\n",
            "    display(Image(filename=diag_dir / \"04_maxslope_hist.png\"))"
        ]
    }
    
    new_blocks = [
        (md_gamma, code_gamma),
        (md_dist, code_dist),
        (md_poly, code_poly),
        (md_cross, code_cross),
        (md_slope, code_slope)
    ]

    final_cells = []
    inserted = False
    
    # We want to remove OLD instances of these if they exist (duplicates or deprecated)
    # AND remove cells containing the DEPRECATED calls mentioned by user
    skipped_ids = [c["id"] for b in new_blocks for c in b]
    
    deprecated_signatures = [
        "plot_gamma_distribution(global_results, diag_dir)", # Old signature
        "plot_parameter_distributions(df_results, diag_dir)" # Old signature
    ]
    
    for cell in nb["cells"]:
        # Skip if ID matches new blocks (we re-insert them)
        if cell.get("id") in skipped_ids:
            continue
            
        # Skip deprecated code cells
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if any(sig in src for sig in deprecated_signatures):
                print(f"Removing deprecated cell: {src[:50]}...")
                continue
        
        final_cells.append(cell)
        
        # Insert New Blocks AFTER the Residuals Diagnostic cell
        # (Assuming it exists, otherwise we might append at end)
        if cell["cell_type"] == "code" and "residuals_diagnostic.png" in "".join(cell["source"]):
             if not inserted:
                 for md, code in new_blocks:
                     final_cells.append(md)
                     final_cells.append(code)
                 inserted = True
    
    # Fallback: if not inserted, append at end
    if not inserted:
         for md, code in new_blocks:
             final_cells.append(md)
             final_cells.append(code)
    
    nb["cells"] = final_cells
    
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook updated: Deprecated cells removed, R-style blocks inserted.")

if __name__ == "__main__":
    refactor_notebook()
