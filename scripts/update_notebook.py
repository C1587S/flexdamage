
import json
from pathlib import Path

def update_notebook():
    nb_path = Path("notebooks/tutorial_walkthrough.ipynb")
    if not nb_path.exists():
        print("Notebook not found")
        return

    with open(nb_path, "r") as f:
        nb = json.load(f)

    # Define new cells
    markdown_cell = {
        "cell_type": "markdown",
        "id": "adv_diag_md",
        "metadata": {},
        "source": [
            "### 6b. Advanced Diagnostics\n",
            "\n",
            "New visualization tools for deep-diving into model behavior:\n",
            "- **Spaghetti Plots**: View all regional curves simultaneously.\n",
            "- **Zero Crossings**: Analyze where response curves cross zero.\n",
            "- **Worst Offenders**: Identify regions with extreme slopes or erratic behavior."
        ]
    }

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "adv_diag_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "from flexdamage.diagnostics.advanced import plot_spaghetti_curves, plot_worst_offenders, analyze_zero_crossings\n",
            "\n",
            "# Spaghetti Plot\n",
            "plot_spaghetti_curves(df_results, config.estimation.functional_form.formula, diag_dir, highlight_regions=regions[:5])\n",
            "display(Image(filename=diag_dir / \"spaghetti_curves.png\"))\n",
            "\n",
            "# Zero Crossings\n",
            "analyze_zero_crossings(df_results, diag_dir)\n",
            "if (diag_dir / \"zero_crossings.png\").exists():\n",
            "    display(Image(filename=diag_dir / \"zero_crossings.png\"))\n",
            "\n",
            "# Worst Offenders\n",
            "plot_worst_offenders(df_results, diag_dir)\n",
            "if (diag_dir / \"worst_offenders.png\").exists():\n",
            "    display(Image(filename=diag_dir / \"worst_offenders.png\"))"
        ]
    }

    # Find insertion point: after cell with id "8dbeed5b" (Basic Diagnostics)
    cells = nb["cells"]
    insert_idx = -1
    for i, cell in enumerate(cells):
        if cell.get("id") == "8dbeed5b":
            insert_idx = i + 1
            break
    
    if insert_idx != -1:
        # Check if already inserted to avoid dupes
        if cells[insert_idx].get("id") == "adv_diag_md":
            print("Cells already present.")
        else:
            cells.insert(insert_idx, code_cell)
            cells.insert(insert_idx, markdown_cell)
            print("Inserted advanced diagnostics cells.")
            
            with open(nb_path, "w") as f:
                json.dump(nb, f, indent=1)
    else:
        print("Target cell not found.")

if __name__ == "__main__":
    update_notebook()
