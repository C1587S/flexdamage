"""
Professional Diagnostic Visualizer.

Features:
- Tufte-inspired design principles
- Configurable style system ('scientific' or 'presentation')
- Enhanced multi-dimensional visualization
- Relationship plots with regression lines (ggplot-style)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union, Literal
import logging
import warnings

from .styles import (
    StyleConfig,
    get_style,
    apply_style,
    get_colormap,
    get_categorical_colors,
    PARAM_LABELS,
    format_stats_compact,
    GREEK_STATS,
)

logger = logging.getLogger(__name__)


class DiagnosticVisualizer:
    """
    Professional diagnostic visualizer with customizable aesthetics.

    Follows Tufte principles:
    - Maximize data-ink ratio
    - Remove chartjunk
    - Use subtle, purposeful color
    - Clear typography

    Args:
        output_dir: Directory to save plots
        style: Style name ('scientific' or 'presentation') or StyleConfig instance
    """

    def __init__(
        self,
        output_dir: Path,
        style: Union[str, StyleConfig] = "scientific"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize style
        if isinstance(style, str):
            self.style = get_style(style)
        else:
            self.style = style

        self._apply_style()
        self.cmap = get_colormap(self.style, "sequential")

    def _apply_style(self):
        """Apply the current style configuration."""
        apply_style(self.style)

    def set_style(self, style: Union[str, StyleConfig]):
        """Change the visualization style."""
        if isinstance(style, str):
            self.style = get_style(style)
        else:
            self.style = style
        self._apply_style()
        self.cmap = get_colormap(self.style, "sequential")

    # =========================================================================
    # NEW: Relationship Plots (Temperature/Economic vs Damages)
    # =========================================================================

    def plot_relationship(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        shape_col: Optional[str] = None,
        facet_col: Optional[str] = None,
        facet_row: Optional[str] = None,
        add_regression: bool = True,
        regression_type: Literal["linear", "lowess", "polynomial"] = "lowess",
        polynomial_degree: int = 2,
        ci: Optional[int] = 95,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        alpha: float = 0.6,
        size_col: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Plot relationship between two variables with optional regression line.

        This is a general-purpose plot for exploring relationships like:
        - Temperature vs Damages
        - Economic variables vs Damages
        - Any X vs Y relationship

        Supports ggplot-style regression overlays.

        Args:
            df: Input DataFrame
            x_col: Column name for X axis (e.g., 'temperature_anomaly')
            y_col: Column name for Y axis (e.g., 'mortality_impact')
            color_col: Column for color encoding (categorical or continuous)
            shape_col: Column for marker shape (categorical)
            facet_col: Column for faceting horizontally
            facet_row: Column for faceting vertically
            add_regression: Whether to add regression line
            regression_type: Type of regression ('linear', 'lowess', 'polynomial')
            polynomial_degree: Degree for polynomial regression
            ci: Confidence interval percentage (None to disable)
            xlabel: Custom X axis label
            ylabel: Custom Y axis label
            title: Plot title
            alpha: Point transparency
            size_col: Column for point size encoding
            filename: Custom output filename

        Returns:
            Path to saved figure
        """
        self._apply_style()

        if df.empty:
            logger.warning("Empty dataframe for relationship plot.")
            return None

        # Validate columns
        actual_color = color_col if color_col and color_col in df.columns else None
        actual_shape = shape_col if shape_col and shape_col in df.columns else None
        actual_facet_col = facet_col if facet_col and facet_col in df.columns else None
        actual_facet_row = facet_row if facet_row and facet_row in df.columns else None
        actual_size = size_col if size_col and size_col in df.columns else None

        # Determine if color is categorical
        color_is_categorical = False
        palette = None
        if actual_color:
            if df[actual_color].dtype in ['object', 'category'] or df[actual_color].nunique() <= 8:
                color_is_categorical = True
                palette = get_categorical_colors(self.style, df[actual_color].nunique())
            else:
                palette = self.cmap

        # Create figure based on faceting
        if actual_facet_col or actual_facet_row:
            # Use FacetGrid for faceted plots
            g = sns.FacetGrid(
                df,
                col=actual_facet_col,
                row=actual_facet_row,
                col_wrap=3 if actual_facet_col and not actual_facet_row else None,
                height=2.5,
                aspect=1.2,
                sharex=True,
                sharey=True,
            )

            # Scatter plot
            scatter_kws = {
                "alpha": alpha,
                "s": self.style.marker_size if not actual_size else None,
                "edgecolor": "white",
                "linewidth": 0.3,
            }

            g.map_dataframe(
                sns.scatterplot,
                x=x_col,
                y=y_col,
                hue=actual_color,
                style=actual_shape,
                size=actual_size,
                palette=palette,
                **scatter_kws
            )

            # Add regression per facet
            if add_regression:
                for ax in g.axes.flatten():
                    self._add_regression_to_ax(
                        ax, df, x_col, y_col, regression_type,
                        polynomial_degree, ci, actual_color, palette, color_is_categorical
                    )

            # Style facet headers
            self._apply_facet_headers_combined(g, actual_facet_col, actual_facet_row)

            # Labels
            x_lab = xlabel or x_col.replace("_", " ").title()
            y_lab = ylabel or y_col.replace("_", " ").title()
            g.set_axis_labels(x_lab, y_lab)

            if title:
                g.fig.suptitle(title, y=1.02, fontsize=self.style.typography.title_size,
                               fontweight=self.style.typography.title_weight)

            # Legend
            if actual_color or actual_shape:
                g.add_legend(bbox_to_anchor=(1.02, 0.5), loc="center left",
                             fontsize=self.style.typography.legend_size)

            fig = g.fig

        else:
            # Single plot
            fig, ax = plt.subplots(figsize=(6, 4.5))

            scatter_kws = {
                "alpha": alpha,
                "s": self.style.marker_size if not actual_size else None,
                "edgecolor": "white",
                "linewidth": 0.3,
            }

            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=actual_color,
                style=actual_shape,
                size=actual_size,
                palette=palette,
                ax=ax,
                **scatter_kws
            )

            # Add regression
            if add_regression:
                self._add_regression_to_ax(
                    ax, df, x_col, y_col, regression_type,
                    polynomial_degree, ci, actual_color, palette, color_is_categorical
                )

            # Labels
            x_lab = xlabel or x_col.replace("_", " ").title()
            y_lab = ylabel or y_col.replace("_", " ").title()
            ax.set_xlabel(x_lab, fontsize=self.style.typography.label_size)
            ax.set_ylabel(y_lab, fontsize=self.style.typography.label_size)

            if title:
                ax.set_title(title, fontsize=self.style.typography.title_size,
                             fontweight=self.style.typography.title_weight)

            # Legend outside
            if actual_color or actual_shape or actual_size:
                ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left",
                          fontsize=self.style.typography.legend_size)

        # Zero line for reference
        if actual_facet_col or actual_facet_row:
            for ax in g.axes.flatten():
                ax.axhline(0, color=self.style.colors.text_light, linestyle="--",
                           linewidth=0.8, alpha=0.5)
        else:
            ax.axhline(0, color=self.style.colors.text_light, linestyle="--",
                       linewidth=0.8, alpha=0.5)

        # Save
        out_name = filename or f"relationship_{x_col}_vs_{y_col}.png"
        out_path = self.output_dir / out_name
        fig.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved relationship plot to {out_path}")
        return out_path

    def _add_regression_to_ax(
        self,
        ax,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        regression_type: str,
        polynomial_degree: int,
        ci: Optional[int],
        color_col: Optional[str],
        palette,
        color_is_categorical: bool
    ):
        """Add regression line to an axis."""
        try:
            if color_col and color_is_categorical:
                # Add regression per color group
                groups = df[color_col].unique()
                colors = palette if isinstance(palette, list) else get_categorical_colors(self.style, len(groups))
                for i, group in enumerate(groups):
                    group_df = df[df[color_col] == group]
                    color = colors[i % len(colors)]
                    self._plot_single_regression(ax, group_df, x_col, y_col,
                                                  regression_type, polynomial_degree, ci, color)
            else:
                # Single regression line
                color = self.style.colors.secondary
                self._plot_single_regression(ax, df, x_col, y_col,
                                              regression_type, polynomial_degree, ci, color)
        except Exception as e:
            logger.warning(f"Could not add regression: {e}")

    def _plot_single_regression(
        self,
        ax,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        regression_type: str,
        polynomial_degree: int,
        ci: Optional[int],
        color: str
    ):
        """Plot a single regression line."""
        x = df[x_col].dropna()
        y = df[y_col].dropna()

        # Align x and y
        valid_idx = x.index.intersection(y.index)
        x = x.loc[valid_idx].values
        y = y.loc[valid_idx].values

        if len(x) < 3:
            return

        x_range = np.linspace(x.min(), x.max(), 100)

        if regression_type == "linear":
            # Linear regression
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x_range)
            ax.plot(x_range, y_pred, color=color, linewidth=1.5, alpha=0.9)

            # CI band
            if ci:
                self._add_ci_band(ax, x, y, x_range, y_pred, coeffs, ci, color, degree=1)

        elif regression_type == "polynomial":
            # Polynomial regression
            coeffs = np.polyfit(x, y, polynomial_degree)
            y_pred = np.polyval(coeffs, x_range)
            ax.plot(x_range, y_pred, color=color, linewidth=1.5, alpha=0.9)

            if ci:
                self._add_ci_band(ax, x, y, x_range, y_pred, coeffs, ci, color, degree=polynomial_degree)

        elif regression_type == "lowess":
            # LOWESS smoothing
            try:
                import statsmodels.api as sm
                lowess = sm.nonparametric.lowess(y, x, frac=0.3)
                ax.plot(lowess[:, 0], lowess[:, 1], color=color, linewidth=1.5, alpha=0.9)
            except ImportError:
                # Fallback to polynomial
                coeffs = np.polyfit(x, y, 2)
                y_pred = np.polyval(coeffs, x_range)
                ax.plot(x_range, y_pred, color=color, linewidth=1.5, alpha=0.9)

    def _add_ci_band(self, ax, x, y, x_range, y_pred, coeffs, ci, color, degree):
        """Add confidence interval band."""
        try:
            n = len(x)
            y_fit = np.polyval(coeffs, x)
            residuals = y - y_fit
            se = np.sqrt(np.sum(residuals**2) / (n - degree - 1))

            from scipy import stats
            t_val = stats.t.ppf((1 + ci/100) / 2, n - degree - 1)

            # Simplified CI (constant width)
            ci_width = t_val * se
            ax.fill_between(x_range, y_pred - ci_width, y_pred + ci_width,
                            color=color, alpha=0.15)
        except Exception:
            pass

    def plot_temperature_damage(
        self,
        df: pd.DataFrame,
        temp_col: str = "temperature_anomaly",
        damage_col: str = "mortality_impact",
        **kwargs
    ) -> Optional[Path]:
        """
        Convenience method for temperature vs damage relationship.

        Args:
            df: Input DataFrame
            temp_col: Temperature column name
            damage_col: Damage/impact column name
            **kwargs: Additional arguments passed to plot_relationship

        Returns:
            Path to saved figure
        """
        defaults = {
            "xlabel": "Temperature Anomaly (\u00B0C)",
            "ylabel": "Damage Impact",
            "title": "Temperature-Damage Relationship",
            "filename": "temperature_damage_relationship.png",
        }
        defaults.update(kwargs)
        return self.plot_relationship(df, temp_col, damage_col, **defaults)

    def plot_economic_damage(
        self,
        df: pd.DataFrame,
        econ_col: str = "lgdp_delta",
        damage_col: str = "mortality_impact",
        **kwargs
    ) -> Optional[Path]:
        """
        Convenience method for economic variable vs damage relationship.

        Args:
            df: Input DataFrame
            econ_col: Economic variable column name
            damage_col: Damage/impact column name
            **kwargs: Additional arguments passed to plot_relationship

        Returns:
            Path to saved figure
        """
        defaults = {
            "xlabel": "Log GDP Delta",
            "ylabel": "Damage Impact",
            "title": "Economic-Damage Relationship",
            "filename": "economic_damage_relationship.png",
        }
        defaults.update(kwargs)
        return self.plot_relationship(df, econ_col, damage_col, **defaults)

    # =========================================================================
    # Core Diagnostic Plots
    # =========================================================================

    def plot_fit_diagnostics(
        self,
        df_results: pd.DataFrame,
        df_data: pd.DataFrame,
        x_col: str,
        y_col: str,
        gamma: float = 0.0,
        color_col: Optional[str] = None,
        regions: Optional[List[str]] = None,
        n_cols: int = 5,
        region_col: str = "region",
        show_curve: bool = True,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        n_worst: Optional[int] = None,
        worst_criteria: str = "eta",
        extra_dims: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Create a FacetGrid diagnostic plot with actual vs predicted values.

        FIX: Title no longer shows single region name for multi-region grids.
        """
        self._apply_style()

        # Auto-select worst regions if needed
        if regions is None and n_worst is not None:
            regions = self.select_worst_regions(df_results, worst_criteria, n_worst)

        # Filter data
        if regions:
            plot_df = df_data[df_data[region_col].isin(regions)].copy()
        else:
            plot_df = df_data.copy()

        if plot_df.empty:
            logger.warning("No data for fit diagnostics plot.")
            return None

        # Merge parameters
        cols_needed = ["region", "alpha", "beta"]
        if not all(c in df_results.columns for c in cols_needed):
            logger.warning(f"Results missing required columns: {cols_needed}")
            return None

        res_subset = df_results[cols_needed]
        plot_df = plot_df.merge(
            res_subset,
            left_on=region_col,
            right_on="region",
            how="inner"
        )

        if plot_df.empty:
            logger.warning("Merge with results produced empty dataframe.")
            return None

        # Calculate predictions
        response = plot_df["alpha"] * plot_df[x_col] + plot_df["beta"] * (plot_df[x_col] ** 2)

        # Apply gamma scaling
        scaling = 1.0
        if color_col and color_col in plot_df.columns and abs(gamma) > 1e-9:
            scaling = np.exp(plot_df[color_col] * gamma)

        pred_col = "_predicted"
        plot_df[pred_col] = response * scaling

        # Prepare for melting
        id_vars = [region_col, x_col]
        if color_col and color_col in plot_df.columns:
            id_vars.append(color_col)

        # Add extra dimensions for richer visualization
        if extra_dims:
            for dim in extra_dims:
                if dim in plot_df.columns and dim not in id_vars:
                    id_vars.append(dim)

        melted = plot_df.melt(
            id_vars=id_vars,
            value_vars=[y_col, pred_col],
            var_name="series",
            value_name="value"
        )

        melted["series"] = melted["series"].map({
            y_col: "Observed",
            pred_col: "Predicted"
        })

        if not show_curve:
            melted = melted[melted["series"] == "Observed"]

        if melted.empty:
            logger.warning("No data left after filtering.")
            return None

        # Create FacetGrid
        n_unique = len(melted[region_col].unique())
        if n_unique == 0:
            return None

        # Color normalization
        norm = None
        if color_col and color_col in melted.columns:
            vmin, vmax = melted[color_col].min(), melted[color_col].max()
            norm = plt.Normalize(vmin, vmax)

        g = sns.FacetGrid(
            melted,
            col=region_col,
            col_wrap=max(1, min(n_cols, n_unique)),
            sharex=False,
            sharey=False,
            height=2.0,
            aspect=1.15,
        )

        # Plot observed data as scatter points
        obs_melted = melted[melted["series"] == "Observed"]
        g.map_dataframe(
            sns.scatterplot,
            x=x_col,
            y="value",
            hue=color_col if color_col and color_col in obs_melted.columns else None,
            hue_norm=norm,
            palette=self.cmap if color_col else None,
            marker="^",
            s=self.style.marker_size,
            alpha=self.style.marker_alpha,
            edgecolor=self.style.colors.text_light,
            linewidth=self.style.marker_edgewidth,
            legend=False
        )

        # FIX: Add fitted curve as a smooth line (not scatter points)
        if show_curve:
            def plot_fitted_curve(data, **kws):
                if data.empty:
                    return
                reg = data[region_col].iloc[0]
                res_row = df_results[df_results["region"] == reg]
                if res_row.empty:
                    return
                res_row = res_row.iloc[0]

                alpha_val = res_row.get("alpha", 0)
                beta_val = res_row.get("beta", 0)

                # Get x range from data
                x_min, x_max = data[x_col].min(), data[x_col].max()
                x_range = np.linspace(x_min, x_max, 100)
                y_fit = alpha_val * x_range + beta_val * (x_range ** 2)

                plt.plot(x_range, y_fit, color=self.style.colors.secondary,
                         linewidth=2, linestyle="-", alpha=0.9)
                plt.axhline(0, color=self.style.colors.text_light,
                            linestyle=":", linewidth=0.8, alpha=0.6)

            g.map_dataframe(plot_fitted_curve)

        # Apply clean facet headers
        self._apply_facet_headers(g, region_col)

        # Axis labels
        x_lab = xlabel or x_col.replace("_", " ").title()
        y_lab = ylabel or y_col.replace("_", " ").title()

        g.set_axis_labels("", "")
        g.fig.supxlabel(x_lab, y=0.02, fontsize=self.style.typography.label_size)
        g.fig.supylabel(y_lab, x=0.02, fontsize=self.style.typography.label_size)

        # FIX: Title should be descriptive, not region-specific
        if title:
            g.fig.suptitle(
                title,
                fontsize=self.style.typography.title_size,
                fontweight=self.style.typography.title_weight,
                y=0.98
            )

        plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.88, wspace=0.25, hspace=0.45)

        # Legend outside plot (FIX from notes)
        handles = []
        handles.append(Line2D([], [], color=self.style.colors.primary, marker='^',
                              linestyle='None', label='Observed', markersize=6))
        if show_curve:
            handles.append(Line2D([], [], color=self.style.colors.secondary, linewidth=2,
                                  linestyle='-', label='Fitted Curve'))

        g.fig.legend(
            handles=handles,
            loc='upper right',
            bbox_to_anchor=(0.99, 0.92),
            frameon=self.style.legend_frameon,
            fontsize=self.style.typography.legend_size
        )

        # Colorbar
        if color_col and color_col in melted.columns:
            cbar_ax = g.fig.add_axes([0.92, 0.25, 0.012, 0.45])
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            sm.set_array([])
            cbar = g.fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(
                color_col.replace("_", " ").title(),
                rotation=270,
                labelpad=12,
                fontsize=self.style.typography.annotation_size
            )
            cbar.ax.tick_params(labelsize=self.style.typography.tick_size - 1)

        # Save
        n_reg = len(regions) if regions else "all"
        out_name = f"fit_diagnostics_{n_reg}.png"
        out_path = self.output_dir / out_name
        plt.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved diagnostic plot to {out_path}")
        return out_path

    def plot_single_region(
        self,
        region: str,
        df_results: pd.DataFrame,
        df_data: pd.DataFrame,
        region_col: str = "region",
        title: Optional[str] = None,
        **kwargs
    ) -> Optional[Path]:
        """Plot diagnostics for a single region."""
        final_title = title or f"Region: {region}"
        return self.plot_fit_diagnostics(
            df_results,
            df_data,
            regions=[region],
            n_cols=1,
            region_col=region_col,
            title=final_title,
            **kwargs
        )

    # =========================================================================
    # Parameter Distribution Plots (FIXED: 3x2 layout, table below, no title)
    # =========================================================================

    def plot_parameter_distributions(
        self,
        df_results: pd.DataFrame,
        show_table: bool = True
    ) -> Optional[Path]:
        """
        Plot parameter distributions with statistics table below.

        FIX: 3x2 layout, stats in table below plots, no main title.

        Args:
            df_results: DataFrame with parameter columns
            show_table: Whether to show statistics table below (default True)

        Returns:
            Path to saved figure
        """
        self._apply_style()

        cols = ["alpha", "beta", "rsqr1", "zeta", "eta", "rsqr2"]
        available_cols = [c for c in cols if c in df_results.columns]

        if not available_cols:
            logger.warning("No parameter columns found for distribution plot.")
            return None

        n_params = len(available_cols)

        # FIX: Force 3x2 layout
        n_cols_grid = 3
        n_rows = 2

        # Create figure with GridSpec for table below (only if show_table is True)
        if show_table:
            fig = plt.figure(figsize=(9, 8.5))  # Taller to accommodate table with more space
            gs = GridSpec(n_rows + 1, n_cols_grid, figure=fig,
                          height_ratios=[1, 1, 0.5], hspace=0.6, wspace=0.3)
        else:
            fig = plt.figure(figsize=(9, 5.5))
            gs = GridSpec(n_rows, n_cols_grid, figure=fig,
                          hspace=0.35, wspace=0.3)

        # Collect stats for table
        stats_data = []

        for idx, col in enumerate(available_cols[:6]):  # Max 6 for 3x2
            row = idx // n_cols_grid
            col_idx = idx % n_cols_grid

            ax = fig.add_subplot(gs[row, col_idx])
            series = df_results[col].dropna()

            if len(series) == 0:
                continue

            # Histogram
            sns.histplot(
                series,
                bins=35,
                color=self.style.colors.primary,
                element="step",
                fill=True,
                alpha=0.6,
                ax=ax,
                linewidth=0.8
            )

            # Mean and median lines
            mean_val = series.mean()
            median_val = series.median()
            sd_val = series.std()

            ax.axvline(mean_val, color=self.style.colors.mean_color,
                       linestyle="--", lw=1.2, alpha=0.9)
            ax.axvline(median_val, color=self.style.colors.median_color,
                       linestyle=":", lw=1.2, alpha=0.9)

            # Clean header
            label = PARAM_LABELS.get(col, col)
            self._add_facet_header(ax, label, height=0.12)

            ax.set_xlabel("")
            ax.set_ylabel("")

            # Collect stats
            stats_data.append({
                "param": label,
                "mean": mean_val,
                "median": median_val,
                "sd": sd_val,
                "n": len(series)
            })

        # FIX: Add stats table below plots (only if show_table is True)
        if show_table and stats_data:
            table_ax = fig.add_subplot(gs[-1, :])
            table_ax.axis("off")

            # Create table
            table_data = []
            headers = ["Parameter", "Mean", "Median", "Std Dev", "N"]
            for s in stats_data:
                table_data.append([
                    s["param"],
                    f"{s['mean']:.4f}",
                    f"{s['median']:.4f}",
                    f"{s['sd']:.4f}",
                    str(s["n"])
                ])

            table = table_ax.table(
                cellText=table_data,
                colLabels=headers,
                loc="center",
                cellLoc="center",
                colColours=[self.style.colors.header_bg] * len(headers),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(self.style.typography.annotation_size)
            table.scale(1, 1.4)

            # Style table
            for key, cell in table.get_celld().items():
                cell.set_edgecolor(self.style.colors.header_border)
                if key[0] == 0:  # Header row
                    cell.set_text_props(fontweight="bold")

        # Legend for lines
        handles = [
            Line2D([0], [0], color=self.style.colors.mean_color, lw=1.5, ls="--", label="Mean"),
            Line2D([0], [0], color=self.style.colors.median_color, lw=1.5, ls=":", label="Median"),
        ]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98),
                   ncol=1, frameon=False, fontsize=self.style.typography.legend_size)

        # FIX: No main title as requested
        plt.subplots_adjust(top=0.92, bottom=0.08)

        out_path = self.output_dir / "param_distributions.png"
        fig.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved parameter distributions to {out_path}")
        return out_path

    # =========================================================================
    # Crossing Zero and Max Slope (FIXED: integrated legend with colored values)
    # =========================================================================

    def plot_crossing_zero(self, df_results: pd.DataFrame) -> Optional[Path]:
        """
        Plot histogram of zero-crossing temperatures for convex regions.

        FIX: Legend integrated with values, colored text (no Greek symbols).
        """
        self._apply_style()

        if "alpha" not in df_results.columns or "beta" not in df_results.columns:
            logger.warning("Missing alpha/beta columns for crossing zero plot.")
            return None

        df = df_results.copy()
        df = df[df["beta"] > 1e-9]  # Convex only
        df["cross0"] = -df["alpha"] / df["beta"]

        return self._plot_metric_histogram_v2(
            df,
            col="cross0",
            xlabel="Temperature at Zero Crossing (\u00B0C)",
            title="Zero Crossing Temperature (Convex Regions)",
            filename="crossing_zero.png"
        )

    def plot_max_slope(
        self,
        df_results: pd.DataFrame,
        t_range: Tuple[float, float] = (0, 10)
    ) -> Optional[Path]:
        """
        Plot histogram of maximum slopes across temperature range.

        FIX: Legend integrated with values, colored text.
        """
        self._apply_style()

        if "alpha" not in df_results.columns or "beta" not in df_results.columns:
            logger.warning("Missing alpha/beta columns for max slope plot.")
            return None

        df = df_results.copy()
        s0 = df["alpha"] + 2 * df["beta"] * t_range[0]
        s1 = df["alpha"] + 2 * df["beta"] * t_range[1]
        df["maxslope"] = np.maximum(s0, s1)

        return self._plot_metric_histogram_v2(
            df,
            col="maxslope",
            xlabel="Maximum Slope Value",
            title=f"Maximum Slope ({t_range[0]}\u2013{t_range[1]}\u00B0C)",
            filename="maxslope_hist.png"
        )

    def _plot_metric_histogram_v2(
        self,
        df: pd.DataFrame,
        col: str,
        xlabel: str,
        title: str,
        filename: str
    ) -> Optional[Path]:
        """
        Create a professional histogram with integrated legend.

        FIX: Legend integrated with values in single box, colored text.
        """
        series = df[col].dropna()
        n = len(series)
        if n == 0:
            return None

        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        # Histogram
        sns.histplot(
            series,
            bins=25,
            color=self.style.colors.primary,
            element="step",
            fill=True,
            alpha=0.6,
            ax=ax,
            linewidth=0.8
        )

        # Statistical lines
        mean_val = series.mean()
        median_val = series.median()
        sd_val = series.std()

        ax.axvline(mean_val, color=self.style.colors.mean_color,
                   linestyle="--", lw=1.5, alpha=0.9)
        ax.axvline(median_val, color=self.style.colors.median_color,
                   linestyle=":", lw=1.5, alpha=0.9)

        # Header
        self._add_facet_header(ax, title, height=0.10)

        # Labels
        ax.set_xlabel(xlabel, fontsize=self.style.typography.label_size)
        ax.set_ylabel("Count", fontsize=self.style.typography.label_size)

        # FIX: Integrated legend with colored values - more compact spacing
        # Colored mean/median/sd annotations with tighter vertical spacing
        y_start = 0.95
        line_height = 0.08  # Reduced spacing between lines

        ax.annotate(
            f"n = {n}",
            xy=(0.97, y_start),
            xycoords="axes fraction",
            fontsize=self.style.typography.annotation_size,
            ha="right",
            va="top",
            color=self.style.colors.text,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=self.style.colors.header_border, alpha=0.9)
        )

        ax.annotate(
            f"Mean = {mean_val:.2f}",
            xy=(0.97, y_start - line_height),
            xycoords="axes fraction",
            fontsize=self.style.typography.annotation_size,
            ha="right",
            va="top",
            color=self.style.colors.mean_color,
            fontweight="medium"
        )

        ax.annotate(
            f"Median = {median_val:.2f}",
            xy=(0.97, y_start - 2 * line_height),
            xycoords="axes fraction",
            fontsize=self.style.typography.annotation_size,
            ha="right",
            va="top",
            color=self.style.colors.median_color,
            fontweight="medium"
        )

        ax.annotate(
            f"Std Dev = {sd_val:.2f}",
            xy=(0.97, y_start - 3 * line_height),
            xycoords="axes fraction",
            fontsize=self.style.typography.annotation_size,
            ha="right",
            va="top",
            color=self.style.colors.text_light,
        )

        plt.subplots_adjust(top=0.88, bottom=0.15, right=0.85)

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {filename}")
        return out_path

    # =========================================================================
    # Residuals Diagnostic (Enhanced with extra dimensions)
    # =========================================================================

    def plot_residuals_diagnostic(
        self,
        df_results: pd.DataFrame,
        df_data: pd.DataFrame,
        x_col: str,
        y_col: str,
        gamma: float = 0.0,
        regions: Optional[List[str]] = None,
        region_col: str = "region",
        n_cols: int = 5,
        color_col: Optional[str] = None,
        shape_col: Optional[str] = None,
        extra_dims: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Plot residuals vs temperature for diagnostic purposes.

        FIX: Support for additional dimensions (rcp, ssp, lgdp) via color/shape.
        """
        self._apply_style()

        # Filter regions
        if regions:
            plot_df = df_data[df_data[region_col].isin(regions)].copy()
        else:
            plot_df = df_data.copy()

        if plot_df.empty:
            return None

        # Merge parameters
        cols_needed = ["region", "alpha", "beta"]
        res_subset = df_results[cols_needed]
        plot_df = plot_df.merge(res_subset, left_on=region_col, right_on="region", how="left")

        # Calculate residuals
        scaling = 1.0
        gdp_col = "lgdp_delta" if "lgdp_delta" in plot_df.columns else None
        if gdp_col and abs(gamma) > 1e-9:
            scaling = np.exp(plot_df[gdp_col] * gamma)

        pred = (plot_df["alpha"] * plot_df[x_col] +
                plot_df["beta"] * (plot_df[x_col] ** 2)) * scaling
        plot_df["_residual"] = plot_df[y_col] - pred

        # Determine color/shape columns
        actual_color = color_col if color_col and color_col in plot_df.columns else None
        actual_shape = shape_col if shape_col and shape_col in plot_df.columns else None

        n_regions = len(plot_df[region_col].unique())
        if n_regions == 0:
            return None

        # Setup palette
        palette = None
        if actual_color:
            if plot_df[actual_color].dtype in ['object', 'category'] or plot_df[actual_color].nunique() <= 10:
                palette = get_categorical_colors(self.style, plot_df[actual_color].nunique())
            else:
                palette = self.cmap

        g = sns.FacetGrid(
            plot_df,
            col=region_col,
            col_wrap=min(n_cols, n_regions),
            sharex=False,
            sharey=False,
            height=2.0,
            aspect=1.2
        )

        # Scatter with optional dimensions
        scatter_kws = {
            "alpha": self.style.marker_alpha,
            "s": self.style.marker_size * 0.8,
            "edgecolor": self.style.colors.text_light,
            "linewidth": 0.3
        }

        if actual_color or actual_shape:
            g.map_dataframe(
                sns.scatterplot,
                x=x_col,
                y="_residual",
                hue=actual_color,
                style=actual_shape,
                palette=palette,
                **scatter_kws
            )
        else:
            g.map_dataframe(
                sns.scatterplot,
                x=x_col,
                y="_residual",
                color=self.style.colors.primary,
                **scatter_kws
            )

        # Zero line
        for ax in g.axes.flatten():
            ax.axhline(0, color=self.style.colors.secondary, linestyle="--",
                       linewidth=1, alpha=0.7)

        # Apply headers
        self._apply_facet_headers(g, region_col)

        g.set_axis_labels(
            x_col.replace("_", " ").title(),
            "Residuals",
            fontsize=self.style.typography.label_size
        )

        g.fig.suptitle(
            "Residuals vs Temperature (Model Fit Check)",
            y=1.02,
            fontsize=self.style.typography.title_size,
            fontweight=self.style.typography.title_weight
        )

        # Add legend if we have color/shape
        if actual_color or actual_shape:
            g.add_legend(
                title="",
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
                fontsize=self.style.typography.legend_size
            )

        out_path = self.output_dir / "residuals_diagnostic.png"
        g.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved residuals diagnostic to {out_path}")
        return out_path

    # =========================================================================
    # Worst Offenders (FIXED: shows actual values alongside predicted)
    # =========================================================================

    def plot_worst_offenders(
        self,
        df_results: pd.DataFrame,
        df_data: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        shape_col: Optional[str] = None,
        criteria: str = "eta",
        n_regions: int = 9,
        region_col: str = "region",
        extra_dims: Optional[List[str]] = None,
        show_fit: bool = True,
        show_observed: bool = True,
    ) -> Optional[Path]:
        """
        Plot data and fits for worst-performing regions.

        FIX: Now shows both observed data points AND fitted curve for comparison.
        """
        self._apply_style()

        regions = self.select_worst_regions(df_results, criteria, n_regions)
        plot_df = df_data[df_data[region_col].isin(regions)].copy()

        if plot_df.empty:
            logger.warning("No data for worst offenders.")
            return None

        # Check available dimensions
        actual_color = color_col if color_col and color_col in plot_df.columns else None
        actual_shape = shape_col if shape_col and shape_col in plot_df.columns else None

        # Determine if color is categorical or continuous
        color_is_categorical = False
        palette = None
        if actual_color:
            if plot_df[actual_color].dtype in ['object', 'category'] or plot_df[actual_color].nunique() <= 8:
                color_is_categorical = True
                palette = get_categorical_colors(self.style, plot_df[actual_color].nunique())
            else:
                palette = self.cmap

        n_cols_grid = min(3, n_regions)
        g = sns.FacetGrid(
            plot_df,
            col=region_col,
            col_wrap=n_cols_grid,
            sharey=False,
            sharex=False,
            height=2.4,
            aspect=1.15
        )

        # FIX: Scatter plot showing OBSERVED data points
        if show_observed:
            scatter_kws = {
                "alpha": self.style.marker_alpha,
                "s": self.style.marker_size,
                "edgecolor": "white",
                "linewidth": 0.3
            }

            g.map_dataframe(
                sns.scatterplot,
                x=x_col,
                y=y_col,
                hue=actual_color,
                style=actual_shape,
                palette=palette,
                **scatter_kws
            )

        # Fitted curves
        if show_fit:
            def plot_curve(data, **kws):
                if data.empty:
                    return
                reg = data[region_col].iloc[0]
                res_row = df_results[df_results["region"] == reg]
                if res_row.empty:
                    return
                res_row = res_row.iloc[0]

                alpha = res_row.get("alpha", 0)
                beta = res_row.get("beta", 0)

                x_range = np.linspace(data[x_col].min(), data[x_col].max(), 100)
                y_fit = alpha * x_range + beta * (x_range ** 2)

                plt.plot(x_range, y_fit, color=self.style.colors.text,
                         linewidth=2, linestyle="-", alpha=0.9, label="Fitted")
                plt.axhline(0, color=self.style.colors.text_light,
                            linestyle=":", linewidth=0.8, alpha=0.6)

            g.map_dataframe(plot_curve)

        # Apply headers
        self._apply_facet_headers(g, region_col)

        g.set_axis_labels(
            x_col.replace("_", " ").title(),
            y_col.replace("_", " ").title(),
            fontsize=self.style.typography.label_size
        )

        # Legend outside with both markers and fit line
        handles = []
        if show_observed:
            handles.append(Line2D([], [], color=self.style.colors.primary, marker='o',
                                  linestyle='None', label='Observed', markersize=6,
                                  markeredgecolor='white', markeredgewidth=0.3))
        if show_fit:
            handles.append(Line2D([], [], color=self.style.colors.text, linewidth=2,
                                  linestyle='-', label='Fitted Curve'))

        if actual_color or actual_shape:
            g.add_legend(
                title="",
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
                fontsize=self.style.typography.legend_size
            )
        elif handles:
            g.fig.legend(
                handles=handles,
                loc='upper right',
                bbox_to_anchor=(0.99, 0.98),
                frameon=self.style.legend_frameon,
                fontsize=self.style.typography.legend_size
            )

        g.fig.suptitle(
            f"Worst Offenders by {criteria.upper()} (Top {n_regions})",
            y=1.02,
            fontsize=self.style.typography.title_size,
            fontweight=self.style.typography.title_weight
        )

        out_path = self.output_dir / "worst_offenders_detail.png"
        g.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved worst offenders to {out_path}")
        return out_path

    # =========================================================================
    # Global Residuals (FIXED: horizontal zero line added)
    # =========================================================================

    def plot_global_residuals(
        self,
        global_results: Dict[str, Any],
        facet_col: Optional[str] = None,
        color_col: Optional[str] = None,
        shape_col: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Plot global model residuals over time.

        FIX: Added horizontal zero line for reference.
        """
        self._apply_style()

        diag_data = global_results.get("diagnostics")
        if diag_data is None:
            logger.warning("No diagnostic data in global results.")
            return None

        df = pd.DataFrame(diag_data)
        if "year" not in df.columns or "residuals" not in df.columns:
            logger.warning("Diagnostic data missing 'year' or 'residuals'.")
            return None

        df = df.dropna(subset=["year", "residuals"])
        if df.empty:
            return None

        # Validate columns
        actual_facet = facet_col if facet_col and facet_col in df.columns else None
        actual_color = color_col if color_col and color_col in df.columns else None
        actual_shape = shape_col if shape_col and shape_col in df.columns else None

        if actual_facet:
            df = df.dropna(subset=[actual_facet])
            if df.empty:
                return None

        # Determine palette
        palette = None
        if actual_color:
            if df[actual_color].dtype in ['object', 'category'] or df[actual_color].nunique() <= 8:
                palette = get_categorical_colors(self.style, df[actual_color].nunique())
            else:
                palette = self.cmap

        try:
            if actual_facet:
                n_facets = df[actual_facet].nunique()
                n_cols = min(3, n_facets)
                g = sns.FacetGrid(
                    df,
                    col=actual_facet,
                    col_wrap=n_cols,
                    height=2.2,
                    aspect=1.3,
                    sharex=True,
                    sharey=True
                )

                scatter_kws = {
                    "alpha": 0.5,
                    "s": self.style.marker_size * 0.6,
                    "edgecolor": "none"
                }

                g.map_dataframe(
                    sns.scatterplot,
                    x="year",
                    y="residuals",
                    hue=actual_color,
                    style=actual_shape,
                    palette=palette,
                    **scatter_kws
                )

                # FIX: Add horizontal zero line to each facet
                for ax in g.axes.flatten():
                    ax.axhline(0, color=self.style.colors.secondary, linestyle="--",
                               linewidth=1.2, alpha=0.7, zorder=0)

                    # LOESS smoothing
                    try:
                        import statsmodels.api as sm
                        # Get data from this facet
                        x_data = df["year"].values
                        y_data = df["residuals"].values
                        if len(x_data) > 10:
                            lowess = sm.nonparametric.lowess(y_data, x_data, frac=0.3)
                            ax.plot(lowess[:, 0], lowess[:, 1],
                                    color=self.style.colors.tertiary,
                                    linewidth=1.5, alpha=0.8)
                    except ImportError:
                        pass

                self._apply_facet_headers(g, actual_facet)
                g.set_axis_labels("Year", "Residuals")

                if actual_color or actual_shape:
                    g.add_legend(
                        bbox_to_anchor=(1.02, 0.5),
                        loc="center left",
                        fontsize=self.style.typography.legend_size
                    )

                fig = g.fig
            else:
                # Single plot
                fig, ax = plt.subplots(figsize=(6, 4))

                sns.scatterplot(
                    data=df,
                    x="year",
                    y="residuals",
                    hue=actual_color,
                    style=actual_shape,
                    palette=palette,
                    alpha=0.5,
                    s=self.style.marker_size * 0.6,
                    edgecolor="none",
                    ax=ax
                )

                # FIX: Add horizontal zero line
                ax.axhline(0, color=self.style.colors.secondary, linestyle="--",
                           linewidth=1.2, alpha=0.7, zorder=0, label="Zero Reference")

                # LOESS
                try:
                    import statsmodels.api as sm
                    lowess = sm.nonparametric.lowess(
                        df["residuals"].values,
                        df["year"].values,
                        frac=0.3
                    )
                    ax.plot(lowess[:, 0], lowess[:, 1],
                            color=self.style.colors.tertiary,
                            linewidth=1.5, alpha=0.8, label="LOESS")
                except ImportError:
                    pass

                ax.set_xlabel("Year")
                ax.set_ylabel("Residuals")

                if actual_color or actual_shape:
                    ax.legend(
                        bbox_to_anchor=(1.02, 0.5),
                        loc="center left",
                        fontsize=self.style.typography.legend_size
                    )

            fig.suptitle(
                "Global Model Residuals Over Time",
                fontsize=self.style.typography.title_size,
                fontweight=self.style.typography.title_weight,
                y=1.02
            )

            out_path = self.output_dir / "global_residuals_time.png"
            fig.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved global residuals plot to {out_path}")
            return out_path

        except Exception as e:
            logger.error(f"Failed to plot global residuals: {e}")
            plt.close()
            return None

    # =========================================================================
    # Summary Plots
    # =========================================================================

    def plot_polynomial_summary(
        self,
        df_results: pd.DataFrame,
        t_range: Tuple[float, float] = (0, 20)
    ) -> Optional[Path]:
        """Plot mean damage function with uncertainty band."""
        self._apply_style()

        if "alpha" not in df_results.columns or "beta" not in df_results.columns:
            return None

        TT = np.linspace(t_range[0], t_range[1], 100)
        alpha = df_results["alpha"].values[:, None]
        beta = df_results["beta"].values[:, None]
        T_grid = TT[None, :]
        YY = alpha * T_grid + beta * (T_grid ** 2)

        mean_curve = YY.mean(axis=0)
        q25 = np.quantile(YY, 0.25, axis=0)
        q75 = np.quantile(YY, 0.75, axis=0)

        fig, ax = plt.subplots(figsize=(5.0, 3.2))

        ax.fill_between(TT, q25, q75, color=self.style.colors.ci_color, alpha=0.3,
                        label="25-75th percentile")
        ax.plot(TT, mean_curve, color=self.style.colors.primary, lw=1.5,
                label="Mean")
        ax.axhline(0, color=self.style.colors.text_light, linestyle="--",
                   linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Temperature Change (\u00B0C)")
        ax.set_ylabel("Regional Damage Function")
        ax.set_xlim(t_range)

        ax.legend(loc="upper left", frameon=self.style.legend_frameon,
                  fontsize=self.style.typography.legend_size)

        self._add_facet_header(ax, "Mean Regional Damage Function", height=0.08)

        out_path = self.output_dir / "polynomial_summary.png"
        fig.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved polynomial summary to {out_path}")
        return out_path

    def plot_gamma_distribution(
        self,
        global_results: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Plot gamma parameter distribution with CI.

        FIX: Better x-axis formatting for small decimals.
        """
        self._apply_style()

        gamma_mu = global_results.get("gamma", 0.0)
        gamma_se = global_results.get("se", 0.0)

        if gamma_se == 0:
            gamma_se = abs(gamma_mu) * 0.1 if gamma_mu != 0 else 0.001
            logger.warning("Gamma SE not found, using estimate.")

        samples = np.random.normal(gamma_mu, gamma_se, 100000)

        fig, ax = plt.subplots(figsize=(5.0, 3.2))

        sns.kdeplot(samples, fill=True, color=self.style.colors.ci_color,
                    alpha=0.4, ax=ax, linewidth=1.2)

        ax.axvline(gamma_mu, color=self.style.colors.mean_color, linestyle="-",
                   linewidth=1.5, label=f"Mean: {gamma_mu:.4f}")
        ax.axvline(gamma_mu + 1.96 * gamma_se, color=self.style.colors.median_color,
                   linestyle="--", linewidth=1.2, label="95% CI")
        ax.axvline(gamma_mu - 1.96 * gamma_se, color=self.style.colors.median_color,
                   linestyle="--", linewidth=1.2)

        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("Density")

        # FIX: Better x-axis formatting for small decimals
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

        # Rotate labels slightly if needed
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

        ax.legend(loc="upper right", frameon=self.style.legend_frameon,
                  fontsize=self.style.typography.legend_size)

        self._add_facet_header(ax, "Global Gamma Distribution", height=0.08)

        out_path = self.output_dir / "gamma_distribution.png"
        fig.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved gamma distribution to {out_path}")
        return out_path

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def select_worst_regions(
        self,
        df_results: pd.DataFrame,
        criteria: str,
        n: int
    ) -> List[str]:
        """
        Select worst-performing regions by criteria.

        Args:
            df_results: Results DataFrame
            criteria: 'eta' (high noise), 'max_slope', 'rsqr1' (low R2)
            n: Number of regions

        Returns:
            List of region names
        """
        df = df_results.copy()

        if criteria == "max_slope":
            if "alpha" in df.columns and "beta" in df.columns:
                df["_metric"] = (df["alpha"] + 2 * df["beta"] * 5).abs()
                return df.sort_values("_metric", ascending=False).head(n)["region"].tolist()

        elif criteria == "eta":
            if "eta" in df.columns:
                return df.sort_values("eta", ascending=False).head(n)["region"].tolist()

        elif criteria == "rsqr1":
            if "rsqr1" in df.columns:
                return df.sort_values("rsqr1", ascending=True).head(n)["region"].tolist()

        return df["region"].head(n).tolist()

    def _apply_facet_headers(self, g: sns.FacetGrid, col_name: str):
        """Apply clean facet headers to a FacetGrid."""
        for ax in g.axes.flatten():
            if ax.get_title():
                title = ax.get_title().split("=")[-1].strip()
                ax.set_title("")
                self._add_facet_header(ax, title)

    def _apply_facet_headers_combined(self, g: sns.FacetGrid, col_name: str, row_name: str):
        """Apply headers for both col and row faceting."""
        for ax in g.axes.flatten():
            if ax.get_title():
                title = ax.get_title().split("|")[-1].strip()
                title = title.split("=")[-1].strip()
                ax.set_title("")
                self._add_facet_header(ax, title)

    def _add_facet_header(self, ax, title: str, height: float = 0.10):
        """Add a clean header bar above an axis."""
        rect = mpatches.Rectangle(
            (0, 1), 1, height,
            transform=ax.transAxes,
            facecolor=self.style.colors.header_bg,
            edgecolor=self.style.colors.header_border,
            linewidth=0.5,
            clip_on=False
        )
        ax.add_patch(rect)
        ax.text(
            0.5, 1 + height / 2,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=self.style.typography.label_size,
            fontweight="medium",
            color=self.style.colors.text
        )
