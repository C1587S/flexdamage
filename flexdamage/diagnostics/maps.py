"""
Map Visualization Module for FlexDamage.

Generates choropleth maps from F2 tables at different geographic levels:
- Country-level maps (automatic geometry fetching from internet)
- Regional/subnational maps (user-provided shapefiles)
- Comparison maps (two-column diagnostic for sign agreement)

Follows the same style conventions as DiagnosticVisualizer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from typing import Optional, Union, Literal, Tuple, List
import logging
import warnings

from .styles import (
    StyleConfig,
    get_style,
    apply_style,
)

logger = logging.getLogger(__name__)

# Robinson projection for world maps
PROJ_ROBINSON = "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"


class SmartFormatter:
    """Formatter for colorbar ticks that handles various value ranges."""

    def __call__(self, x, pos=None):
        if x == 0:
            return "0"
        elif abs(x) >= 1000:
            return f"{x/1000:.1f}k"
        elif abs(x) >= 1:
            return f"{int(round(x))}"
        elif abs(x) >= 0.01:
            return f"{x:.2f}"
        else:
            return f"{x:.2e}"


class MapVisualizer:
    """
    Professional map visualizer for F2 tables and damage projections.

    Follows the same design principles as DiagnosticVisualizer:
    - Tufte-inspired design
    - Configurable style system
    - Consistent output conventions

    Supports two geographic levels:
    1. Country-level: Automatically fetches geometries from internet (Natural Earth)
    2. Regional/subnational: Requires user-provided shapefile

    Args:
        output_dir: Directory to save map outputs
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

        # Cache for loaded geometries
        self._country_gdf = None
        self._shapefile_cache = {}

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

    # =========================================================================
    # Geometry Loading
    # =========================================================================

    def _get_country_geometries(self, simplify_tolerance: float = 1500) -> "geopandas.GeoDataFrame":
        """
        Fetch country geometries from Natural Earth (via internet).

        Uses geopandas built-in dataset fetching.
        Caches result for subsequent calls.

        Args:
            simplify_tolerance: Geometry simplification tolerance (meters)

        Returns:
            GeoDataFrame with country geometries
        """
        if self._country_gdf is not None:
            return self._country_gdf

        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for map generation. "
                "Install with: pip install geopandas"
            )

        logger.info("Fetching country geometries from Natural Earth...")

        try:
            # Try to get Natural Earth countries dataset
            world = gpd.read_file(
                gpd.datasets.get_path('naturalearth_lowres')
            )
        except Exception:
            # Fallback: fetch from Natural Earth directly
            logger.info("Fetching from Natural Earth CDN...")
            url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            world = gpd.read_file(url)

        # Reproject to Robinson
        world = world.to_crs(PROJ_ROBINSON)

        # Simplify geometries for performance
        if simplify_tolerance > 0:
            world["geometry"] = world.geometry.simplify(simplify_tolerance)

        # Standardize ISO column name
        if "iso_a3" in world.columns:
            world["iso3"] = world["iso_a3"]
        elif "ISO_A3" in world.columns:
            world["iso3"] = world["ISO_A3"]
        elif "ADM0_A3" in world.columns:
            world["iso3"] = world["ADM0_A3"]

        # Filter out Antarctica and invalid entries
        world = world[
            (world["iso3"] != "ATA") &
            (world["iso3"] != "-99") &
            (world["iso3"].notna())
        ]

        self._country_gdf = world
        logger.info(f"Loaded {len(world)} country geometries")
        return world

    def _load_shapefile(
        self,
        shapefile_path: str,
        simplify_tolerance: float = 1500,
        exclude_patterns: Optional[List[str]] = None
    ) -> "geopandas.GeoDataFrame":
        """
        Load a shapefile and reproject to Robinson.

        Args:
            shapefile_path: Path to shapefile
            simplify_tolerance: Geometry simplification tolerance
            exclude_patterns: List of ID patterns to exclude (e.g., ["CA-", "ATA"])

        Returns:
            GeoDataFrame with geometries
        """
        # Check cache
        cache_key = (shapefile_path, simplify_tolerance)
        if cache_key in self._shapefile_cache:
            return self._shapefile_cache[cache_key]

        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for map generation. "
                "Install with: pip install geopandas"
            )

        logger.info(f"Loading shapefile: {shapefile_path}")

        try:
            # Try pyogrio first (faster)
            import pyogrio
            gdf = pyogrio.read_dataframe(shapefile_path)
        except ImportError:
            # Fallback to geopandas
            gdf = gpd.read_file(shapefile_path)

        # Reproject to Robinson
        gdf = gdf.to_crs(PROJ_ROBINSON)

        # Simplify geometries
        if simplify_tolerance > 0:
            gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance)

        # Exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                # Find ID column (common names)
                id_cols = ["hierid", "HIERID", "region", "REGION", "id", "ID"]
                for col in id_cols:
                    if col in gdf.columns:
                        mask = ~gdf[col].astype(str).str.startswith(pattern)
                        gdf = gdf[mask]
                        break

        self._shapefile_cache[cache_key] = gdf
        logger.info(f"Loaded {len(gdf)} geometries from shapefile")
        return gdf

    # =========================================================================
    # Data Loading and Filtering
    # =========================================================================

    def load_data(
        self,
        data: Union[str, Path, pd.DataFrame],
        year: Optional[int] = None,
        year_col: str = "year"
    ) -> pd.DataFrame:
        """
        Load and optionally filter data.

        Args:
            data: Path to data file or DataFrame
            year: Optional year to filter by
            year_col: Column name for year filtering

        Returns:
            Filtered DataFrame
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            path = Path(data)
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".csv":
                df = pd.read_csv(path)
            elif path.suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            else:
                # Try pandas auto-detection
                df = pd.read_csv(path)

        # Year filtering
        if year is not None and year_col in df.columns:
            df = df[df[year_col] == year].copy()
            logger.info(f"Filtered to year {year}: {len(df)} records")

        return df

    # =========================================================================
    # Color Mapping
    # =========================================================================

    def _get_colormap(
        self,
        style: Literal["standard", "degree_days", "comparison", "diverging"] = "standard",
        vmin: float = -1,
        vmax: float = 1
    ) -> Tuple[mcolors.Colormap, mcolors.Normalize]:
        """
        Get colormap and normalization for map plotting.

        Args:
            style: Color scheme style
            vmin: Minimum value for normalization
            vmax: Maximum value for normalization

        Returns:
            Tuple of (colormap, normalization)
        """
        # Use TwoSlopeNorm only if 0 is within the data range (vmin < 0 < vmax)
        # Otherwise use standard Normalize to avoid ValueError
        if vmin < 0 < vmax:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        if style == "degree_days":
            # Blue-white-red for temperature/damage
            colors = [
                (0.0, "#313695"),
                (0.125, "#4575b4"),
                (0.25, "#74add1"),
                (0.375, "#abd9e9"),
                (0.5, "#ffffff"),
                (0.6, "#fee090"),
                (0.7, "#fdae61"),
                (0.8, "#f46d43"),
                (0.9, "#d73027"),
                (1.0, "#a50026")
            ]
            cmap = mcolors.LinearSegmentedColormap.from_list("dd_white_center", colors)
        elif style == "comparison":
            # Red-white-green for agreement
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "comparison", ["#d73027", "#ffffff", "#1a9850"]
            )
        elif style == "diverging":
            # RdBu diverging
            cmap = plt.cm.RdBu_r
        else:
            # Standard blue-white-red
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "standard", ["#0000ff", "#ffffff", "#ff1212"]
            )

        return cmap, norm

    def _set_map_style(self, font_size: int = 10):
        """Apply minimal map styling (no axes, ticks, etc.)."""
        plt.style.use("seaborn-v0_8-white")
        plt.rcParams.update({
            "axes.axisbelow": True,
            "axes.edgecolor": "none",
            "axes.facecolor": "none",
            "axes.grid": False,
            "axes.labelcolor": "none",
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.bottom": False,
            "xtick.labelbottom": False,
            "ytick.left": False,
            "ytick.labelleft": False,
            "font.size": font_size
        })

    # =========================================================================
    # Country-Level Map
    # =========================================================================

    def plot_country_map(
        self,
        data: Union[str, Path, pd.DataFrame],
        value_col: str,
        geo_id_col: str = "iso3",
        year: Optional[int] = None,
        year_col: str = "year",
        title: Optional[str] = None,
        units: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap_style: Literal["standard", "degree_days", "diverging"] = "degree_days",
        filename: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 5),
        simplify_tolerance: float = 1500,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Generate a country-level choropleth map.

        Automatically fetches country geometries from the internet.

        Args:
            data: Data source (path or DataFrame)
            value_col: Column containing values to plot
            geo_id_col: Column containing ISO-3 country codes
            year: Optional year to filter data
            year_col: Column name for year filtering
            title: Map title (defaults to value_col name)
            units: Units for legend label
            vmin: Minimum value for color scale (clips below)
            vmax: Maximum value for color scale (clips above)
            cmap_style: Color scheme style
            filename: Output filename
            figsize: Figure size
            simplify_tolerance: Geometry simplification tolerance
            exclude_patterns: List of patterns to exclude (e.g., ["ATA"] for Antarctica)

        Returns:
            Path to saved map image
        """
        # Load data
        df = self.load_data(data, year=year, year_col=year_col)

        if value_col not in df.columns:
            logger.error(f"Value column '{value_col}' not found in data")
            return None

        if geo_id_col not in df.columns:
            logger.error(f"Geographic ID column '{geo_id_col}' not found in data")
            return None

        # Get country geometries
        try:
            world = self._get_country_geometries(simplify_tolerance)
        except Exception as e:
            logger.error(f"Failed to load country geometries: {e}")
            return None

        # Apply exclusions
        if exclude_patterns:
            for pattern in exclude_patterns:
                world = world[~world["iso3"].str.contains(pattern, na=False)]

        # Merge data with geometries
        gdf = world.merge(
            df[[geo_id_col, value_col]].drop_duplicates(subset=[geo_id_col]),
            left_on="iso3",
            right_on=geo_id_col,
            how="left"
        )

        # Generate the map
        return self._plot_map(
            gdf=gdf,
            value_col=value_col,
            title=title or value_col.replace("_", " ").title(),
            units=units,
            vmin=vmin,
            vmax=vmax,
            cmap_style=cmap_style,
            filename=filename or f"country_map_{value_col}.png",
            figsize=figsize,
        )

    # =========================================================================
    # Regional/Subnational Map
    # =========================================================================

    def plot_regional_map(
        self,
        data: Union[str, Path, pd.DataFrame],
        shapefile_path: str,
        value_col: str,
        geo_id_col: str,
        shape_id_col: str,
        year: Optional[int] = None,
        year_col: str = "year",
        title: Optional[str] = None,
        units: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap_style: Literal["standard", "degree_days", "diverging"] = "degree_days",
        filename: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 5),
        simplify_tolerance: float = 1500,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Generate a regional/subnational choropleth map.

        Requires user-provided shapefile for geometries.

        Args:
            data: Data source (path or DataFrame)
            shapefile_path: Path to shapefile with regional geometries
            value_col: Column containing values to plot
            geo_id_col: Column in data containing region identifiers
            shape_id_col: Column in shapefile containing region identifiers
            year: Optional year to filter data
            year_col: Column name for year filtering
            title: Map title
            units: Units for legend label
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale
            cmap_style: Color scheme style
            filename: Output filename
            figsize: Figure size
            simplify_tolerance: Geometry simplification tolerance
            exclude_patterns: Patterns to exclude from shapefile (e.g., ["CA-", "ATA"])

        Returns:
            Path to saved map image
        """
        # Load data
        df = self.load_data(data, year=year, year_col=year_col)

        if value_col not in df.columns:
            logger.error(f"Value column '{value_col}' not found in data")
            return None

        if geo_id_col not in df.columns:
            logger.error(f"Geographic ID column '{geo_id_col}' not found in data")
            return None

        # Load shapefile
        try:
            shp_gdf = self._load_shapefile(
                shapefile_path,
                simplify_tolerance,
                exclude_patterns
            )
        except Exception as e:
            logger.error(f"Failed to load shapefile: {e}")
            return None

        if shape_id_col not in shp_gdf.columns:
            logger.error(f"Shape ID column '{shape_id_col}' not found in shapefile")
            logger.info(f"Available columns: {list(shp_gdf.columns)}")
            return None

        # Merge data with geometries
        gdf = shp_gdf.merge(
            df[[geo_id_col, value_col]].drop_duplicates(subset=[geo_id_col]),
            left_on=shape_id_col,
            right_on=geo_id_col,
            how="left"
        )

        # Generate the map
        return self._plot_map(
            gdf=gdf,
            value_col=value_col,
            title=title or value_col.replace("_", " ").title(),
            units=units,
            vmin=vmin,
            vmax=vmax,
            cmap_style=cmap_style,
            filename=filename or f"regional_map_{value_col}.png",
            figsize=figsize,
        )

    # =========================================================================
    # Comparison Map (Two-Column Diagnostic)
    # =========================================================================

    def plot_comparison_map(
        self,
        data: Union[str, Path, pd.DataFrame],
        value_col_a: str,
        value_col_b: str,
        geo_id_col: str = "iso3",
        geo_level: Literal["country", "regional"] = "country",
        shapefile_path: Optional[str] = None,
        shape_id_col: Optional[str] = None,
        year: Optional[int] = None,
        year_col: str = "year",
        title: Optional[str] = None,
        filename: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 5),
        simplify_tolerance: float = 1500,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Generate a comparison map showing sign agreement and magnitude difference.

        Compares two value columns to check:
        - Whether predictions match in sign
        - Magnitude of difference between predictions

        Colors:
        - Green: Sign match (both positive or both negative)
        - Red: Sign mismatch (different signs)
        - Intensity: Magnitude of difference

        Args:
            data: Data source (path or DataFrame)
            value_col_a: First value column to compare
            value_col_b: Second value column to compare
            geo_id_col: Column containing geographic identifiers
            geo_level: Geographic level ("country" or "regional")
            shapefile_path: Path to shapefile (required if geo_level="regional")
            shape_id_col: ID column in shapefile (required if geo_level="regional")
            year: Optional year to filter data
            year_col: Column name for year filtering
            title: Map title
            filename: Output filename
            figsize: Figure size
            simplify_tolerance: Geometry simplification tolerance
            exclude_patterns: Patterns to exclude from shapefile

        Returns:
            Path to saved map image
        """
        # Load data
        df = self.load_data(data, year=year, year_col=year_col)

        # Validate columns
        for col in [value_col_a, value_col_b, geo_id_col]:
            if col not in df.columns:
                logger.error(f"Column '{col}' not found in data")
                return None

        # Calculate comparison metrics
        df = df.copy()
        df["_sign_a"] = np.sign(df[value_col_a])
        df["_sign_b"] = np.sign(df[value_col_b])
        df["_sign_match"] = df["_sign_a"] == df["_sign_b"]
        df["_diff_mag"] = (df[value_col_a] - df[value_col_b]).abs()

        # Create visualization value: positive for match, negative for mismatch
        df["_viz_comp"] = np.where(
            df["_sign_match"],
            df["_diff_mag"],
            -df["_diff_mag"]
        )

        # Get geometries
        if geo_level == "country":
            try:
                gdf = self._get_country_geometries(simplify_tolerance)
            except Exception as e:
                logger.error(f"Failed to load country geometries: {e}")
                return None

            gdf = gdf.merge(
                df[[geo_id_col, "_viz_comp", "_sign_match", "_diff_mag"]].drop_duplicates(subset=[geo_id_col]),
                left_on="iso3",
                right_on=geo_id_col,
                how="left"
            )
        else:
            # Regional level
            if not shapefile_path or not shape_id_col:
                logger.error("shapefile_path and shape_id_col required for regional maps")
                return None

            try:
                shp_gdf = self._load_shapefile(
                    shapefile_path,
                    simplify_tolerance,
                    exclude_patterns
                )
            except Exception as e:
                logger.error(f"Failed to load shapefile: {e}")
                return None

            gdf = shp_gdf.merge(
                df[[geo_id_col, "_viz_comp", "_sign_match", "_diff_mag"]].drop_duplicates(subset=[geo_id_col]),
                left_on=shape_id_col,
                right_on=geo_id_col,
                how="left"
            )

        # Generate comparison map
        return self._plot_comparison_map_internal(
            gdf=gdf,
            title=title or f"Comparison: {value_col_a} vs {value_col_b}",
            filename=filename or f"comparison_{value_col_a}_vs_{value_col_b}.png",
            figsize=figsize,
        )

    # =========================================================================
    # Internal Plotting Methods
    # =========================================================================

    def _plot_map(
        self,
        gdf: "geopandas.GeoDataFrame",
        value_col: str,
        title: str,
        units: Optional[str],
        vmin: Optional[float],
        vmax: Optional[float],
        cmap_style: str,
        filename: str,
        figsize: Tuple[float, float],
    ) -> Optional[Path]:
        """Internal method to generate a standard choropleth map."""
        import geopandas as gpd

        # Calculate bounds if not provided
        data_values = gdf[value_col].dropna()
        if len(data_values) == 0:
            logger.warning("No valid data values for mapping")
            return None

        actual_vmin = vmin if vmin is not None else data_values.min()
        actual_vmax = vmax if vmax is not None else data_values.max()

        # Ensure vmin < vmax
        if actual_vmin >= actual_vmax:
            actual_vmax = actual_vmin + 0.0001

        # Get colormap
        cmap, norm = self._get_colormap(cmap_style, actual_vmin, actual_vmax)

        # Set map style
        self._set_map_style()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot choropleth
        gdf.plot(
            column=value_col,
            ax=ax,
            cmap=cmap,
            norm=norm,
            legend=False,
            edgecolor="#d3d3d3",  # Light grey borders for all geometries
            linewidth=0.3,
            missing_kwds={"color": "lightgray", "edgecolor": "#d3d3d3", "linewidth": 0.3}
        )

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.04])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", format=SmartFormatter())

        # Colorbar label
        label = title
        if units:
            label = f"{title} ({units})"
        cbar.set_label(label, labelpad=8)
        cbar.outline.set_visible(False)

        # Remove axes
        ax.set_axis_off()

        # Save
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight", transparent=True)
        plt.close()

        logger.info(f"Saved map to {out_path}")

        # Also save stats
        self._save_map_stats(gdf, value_col, out_path.stem)

        return out_path

    def _plot_comparison_map_internal(
        self,
        gdf: "geopandas.GeoDataFrame",
        title: str,
        filename: str,
        figsize: Tuple[float, float],
    ) -> Optional[Path]:
        """Internal method to generate a comparison map."""
        import geopandas as gpd

        # Get data max for symmetric scale
        data_max = gdf["_diff_mag"].dropna().max()
        if pd.isna(data_max) or data_max == 0:
            data_max = 1

        # Get colormap (comparison style)
        cmap, norm = self._get_colormap("comparison", -data_max, data_max)

        # Set map style
        self._set_map_style()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot choropleth
        gdf.plot(
            column="_viz_comp",
            ax=ax,
            cmap=cmap,
            norm=norm,
            legend=False,
            edgecolor="#d3d3d3",  # Light grey borders for all geometries
            linewidth=0.3,
            missing_kwds={"color": "lightgray", "edgecolor": "#d3d3d3", "linewidth": 0.3}
        )

        # Add legend for sign match/mismatch
        handles = [
            Patch(facecolor="#1a9850", label="Sign match"),
            Patch(facecolor="#d73027", label="Sign mismatch")
        ]
        ax.legend(handles=handles, loc="lower left", frameon=False)

        # Add colorbar for magnitude (grayscale)
        sm = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(0, data_max),
            cmap=plt.cm.Greys
        )
        sm._A = []
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.04])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", format=SmartFormatter())
        cbar.set_label("|Difference| (intensity)", labelpad=8)
        cbar.outline.set_visible(False)

        # Title
        ax.set_title(title, fontsize=self.style.typography.title_size, pad=10)

        # Remove axes
        ax.set_axis_off()

        # Save
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=self.style.save_dpi, bbox_inches="tight", transparent=True)
        plt.close()

        logger.info(f"Saved comparison map to {out_path}")

        # Save comparison stats
        self._save_comparison_stats(gdf, out_path.stem)

        return out_path

    # =========================================================================
    # Statistics Export
    # =========================================================================

    def _save_map_stats(
        self,
        gdf: "geopandas.GeoDataFrame",
        value_col: str,
        base_name: str
    ):
        """Save top 10 and bottom 10 regions by value."""
        # Find region ID column
        id_col = None
        for col in ["iso3", "hierid", "region", "HIERID", "REGION", "ID"]:
            if col in gdf.columns:
                id_col = col
                break

        if id_col is None:
            return

        df = gdf[[id_col, value_col]].dropna().copy()
        if len(df) == 0:
            return

        sorted_df = df.sort_values(by=value_col)

        # Bottom 10 and Top 10
        stats = pd.concat([
            sorted_df.head(10).assign(Category="Bottom 10"),
            sorted_df.tail(10).assign(Category="Top 10")
        ])

        out_path = self.output_dir / f"{base_name}_stats.csv"
        stats.to_csv(out_path, index=False)
        logger.info(f"Saved stats to {out_path}")

    def _save_comparison_stats(
        self,
        gdf: "geopandas.GeoDataFrame",
        base_name: str
    ):
        """Save comparison statistics (best/worst alignment)."""
        # Find region ID column
        id_col = None
        for col in ["iso3", "hierid", "region", "HIERID", "REGION", "ID"]:
            if col in gdf.columns:
                id_col = col
                break

        if id_col is None:
            return

        df = gdf[[id_col, "_sign_match", "_diff_mag"]].dropna().copy()
        if len(df) == 0:
            return

        # Best: Sign match + lowest difference
        best_df = df[df["_sign_match"]].sort_values("_diff_mag", ascending=True).head(10).copy()
        best_df["Category"] = "Best (Match + Low Diff)"

        # Worst: Sign mismatch first, then high difference
        df["_is_mismatch"] = ~df["_sign_match"]
        worst_df = df.sort_values(["_is_mismatch", "_diff_mag"], ascending=[False, False]).head(10).copy()
        worst_df["Category"] = "Worst (Mismatch or High Diff)"

        stats = pd.concat([best_df, worst_df])[[id_col, "_sign_match", "_diff_mag", "Category"]]
        stats.columns = [id_col, "sign_match", "diff_magnitude", "Category"]

        out_path = self.output_dir / f"{base_name}_stats.csv"
        stats.to_csv(out_path, index=False)
        logger.info(f"Saved comparison stats to {out_path}")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def plot_f2_map(
        self,
        f2_data: Union[str, Path, pd.DataFrame],
        value_col: str = "flextotal",
        geo_level: Literal["country", "regional"] = "country",
        geo_id_col: str = "iso3",
        shapefile_path: Optional[str] = None,
        shape_id_col: Optional[str] = None,
        year: Optional[int] = None,
        year_col: str = "year",
        title: Optional[str] = None,
        units: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        **kwargs
    ) -> Optional[Path]:
        """
        Convenience method for plotting F2 table data.

        Automatically selects country or regional map based on geo_level.

        Args:
            f2_data: F2 table data (path or DataFrame)
            value_col: Column to plot (default: "flextotal")
            geo_level: Geographic level ("country" or "regional")
            geo_id_col: Column with geographic identifiers
            shapefile_path: Path to shapefile (required for regional)
            shape_id_col: ID column in shapefile (required for regional)
            year: Optional year filter
            year_col: Year column name
            title: Map title
            units: Units for legend
            vmin: Min value for color scale
            vmax: Max value for color scale
            **kwargs: Additional arguments passed to plot method

        Returns:
            Path to saved map
        """
        if geo_level == "country":
            return self.plot_country_map(
                data=f2_data,
                value_col=value_col,
                geo_id_col=geo_id_col,
                year=year,
                year_col=year_col,
                title=title,
                units=units,
                vmin=vmin,
                vmax=vmax,
                **kwargs
            )
        else:
            if not shapefile_path or not shape_id_col:
                logger.error("shapefile_path and shape_id_col required for regional maps")
                return None

            return self.plot_regional_map(
                data=f2_data,
                shapefile_path=shapefile_path,
                value_col=value_col,
                geo_id_col=geo_id_col,
                shape_id_col=shape_id_col,
                year=year,
                year_col=year_col,
                title=title,
                units=units,
                vmin=vmin,
                vmax=vmax,
                **kwargs
            )

    # =========================================================================
    # GIF Animation Generation
    # =========================================================================

    def create_animated_gif(
        self,
        image_paths: List[Union[str, Path]],
        output_filename: str = "animation.gif",
        duration: int = 1000,
        loop: int = 0
    ) -> Optional[Path]:
        """
        Create an animated GIF from a list of image paths.

        Args:
            image_paths: List of paths to PNG images
            output_filename: Output filename for the GIF
            duration: Duration per frame in milliseconds (default: 1000ms = 1s)
            loop: Number of loops (0 = infinite)

        Returns:
            Path to saved GIF
        """
        try:
            from PIL import Image
        except ImportError:
            logger.error("PIL/Pillow required for GIF creation. Install with: pip install Pillow")
            return None

        if not image_paths:
            logger.warning("No images provided for GIF creation")
            return None

        # Load images
        images = []
        for path in image_paths:
            try:
                img = Image.open(path)
                # Convert to RGBA if needed
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                images.append(img)
            except Exception as e:
                logger.warning(f"Could not load image {path}: {e}")

        if not images:
            logger.error("No valid images loaded for GIF creation")
            return None

        # Save as GIF
        out_path = self.output_dir / output_filename
        images[0].save(
            out_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=False
        )

        logger.info(f"Saved animated GIF to {out_path}")
        return out_path

    def create_grid_gif(
        self,
        data: Union[str, Path, pd.DataFrame],
        time_col: str,
        grid_cols: List[str],
        value_col: str = "flextotal",
        geo_level: Literal["country", "regional"] = "country",
        geo_id_col: str = "iso3",
        shapefile_path: Optional[str] = None,
        shape_id_col: Optional[str] = None,
        title_template: str = "{grid_label} - {time_value}",
        units: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        duration: int = 1000,
        output_filename: str = "grid_animation.gif",
        figsize: Tuple[float, float] = (20, 10),
        exclude_patterns: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Create an animated GIF with a grid layout showing different combinations.

        For each time step, creates a grid with panels for each combination of grid_cols values.
        The animation transitions through time steps.

        Args:
            data: Data source (path or DataFrame)
            time_col: Column to use for animation frames (e.g., "period", "year_center")
            grid_cols: Columns defining grid panels (e.g., ["rcp", "model"])
            value_col: Column with values to plot
            geo_level: "country" or "regional"
            geo_id_col: Column with geographic identifiers
            shapefile_path: Path to shapefile (for regional maps)
            shape_id_col: ID column in shapefile (for regional maps)
            title_template: Template for panel titles. Available placeholders:
                           {grid_label}, {time_value}, {time_col}
            units: Units for legend
            vmin: Min value for color scale (auto if None)
            vmax: Max value for color scale (auto if None)
            duration: Duration per frame in milliseconds
            output_filename: Output filename for the GIF
            figsize: Figure size for the grid
            exclude_patterns: Patterns to exclude from the map (e.g., ["ATA"])

        Returns:
            Path to saved GIF
        """
        try:
            from PIL import Image
            import geopandas as gpd
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            return None

        # Load data
        df = self.load_data(data)
        if df.empty:
            logger.warning("Empty data, cannot create GIF")
            return None

        # Get unique time values
        time_values = sorted(df[time_col].unique())
        logger.info(f"Creating grid GIF with {len(time_values)} time steps: {time_values}")

        # Get unique combinations of grid columns
        if len(grid_cols) == 1:
            combos = [(v,) for v in sorted(df[grid_cols[0]].unique())]
        else:
            combos = df[grid_cols].drop_duplicates().sort_values(grid_cols).values.tolist()
            combos = [tuple(c) for c in combos]

        n_panels = len(combos)
        logger.info(f"Grid has {n_panels} panels: {combos}")

        # Determine grid layout
        n_cols = min(n_panels, 2)
        n_rows = (n_panels + n_cols - 1) // n_cols

        # Calculate global vmin/vmax if not provided
        if vmin is None:
            vmin = df[value_col].min()
        if vmax is None:
            vmax = df[value_col].max()

        logger.info(f"Color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

        # Get geometries
        if geo_level == "country":
            gdf_base = self._get_country_geometries()
        else:
            gdf_base = self._load_shapefile(shapefile_path, simplify_tolerance=0.01)

        if gdf_base is None:
            logger.error("Could not load geometries")
            return None

        # Apply exclusions
        if exclude_patterns and geo_level == "country":
            for pattern in exclude_patterns:
                gdf_base = gdf_base[~gdf_base["iso3"].str.contains(pattern, na=False)]

        # Project to Robinson
        gdf_base = gdf_base.to_crs(PROJ_ROBINSON)

        # Generate frames
        frame_paths = []
        temp_dir = self.output_dir / "_temp_frames"
        temp_dir.mkdir(exist_ok=True)

        for time_val in time_values:
            # Filter data for this time step
            df_time = df[df[time_col] == time_val]

            # Create figure with subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_panels == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            # Get colormap
            cmap, norm = self._get_colormap("standard", vmin, vmax)

            for idx, combo in enumerate(combos):
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row, col]

                # Filter for this combination
                mask = pd.Series([True] * len(df_time))
                for i, grid_col in enumerate(grid_cols):
                    mask = mask & (df_time[grid_col] == combo[i])
                df_combo = df_time[mask]

                # Create label for this combo
                if len(grid_cols) == 1:
                    grid_label = f"{grid_cols[0]}={combo[0]}"
                else:
                    grid_label = ", ".join([f"{grid_cols[i]}={combo[i]}" for i in range(len(grid_cols))])

                # Merge with geometries
                if geo_level == "country":
                    gdf = gdf_base.merge(
                        df_combo[[geo_id_col, value_col]].drop_duplicates(subset=[geo_id_col]),
                        left_on="iso3",
                        right_on=geo_id_col,
                        how="left"
                    )
                else:
                    gdf = gdf_base.merge(
                        df_combo[[geo_id_col, value_col]].drop_duplicates(subset=[geo_id_col]),
                        left_on=shape_id_col,
                        right_on=geo_id_col,
                        how="left"
                    )

                # Plot
                gdf.plot(
                    column=value_col,
                    ax=ax,
                    cmap=cmap,
                    norm=norm,
                    legend=False,
                    edgecolor="#d3d3d3",  # Light grey borders for all geometries
                    linewidth=0.3,
                    missing_kwds={"color": "lightgray", "edgecolor": "#d3d3d3", "linewidth": 0.3}
                )

                # Panel title
                panel_title = title_template.format(
                    grid_label=grid_label,
                    time_value=time_val,
                    time_col=time_col
                )
                ax.set_title(panel_title, fontsize=10)
                ax.set_axis_off()

            # Hide empty subplots
            for idx in range(n_panels, n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                axes[row, col].set_axis_off()

            # Add shared colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", format=SmartFormatter())
            label = value_col.replace("_", " ").title()
            if units:
                label = f"{label} ({units})"
            cbar.set_label(label, labelpad=5)
            cbar.outline.set_visible(False)

            # Main title with time
            fig.suptitle(f"{time_col}: {time_val}", fontsize=14, y=0.98)

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])

            # Save frame
            frame_path = temp_dir / f"frame_{time_val}.png"
            plt.savefig(frame_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            frame_paths.append(frame_path)

        # Create GIF
        gif_path = self.create_animated_gif(frame_paths, output_filename, duration=duration)

        # Clean up temp files
        for fp in frame_paths:
            try:
                fp.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass

        return gif_path

    def create_time_series_gif(
        self,
        data: Union[str, Path, pd.DataFrame],
        time_col: str,
        time_values: Optional[List] = None,
        value_col: str = "flextotal",
        geo_level: Literal["country", "regional"] = "country",
        geo_id_col: str = "iso3",
        shapefile_path: Optional[str] = None,
        shape_id_col: Optional[str] = None,
        title_template: str = "{value_col} - {time_value}",
        units: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        duration: int = 1000,
        output_filename: str = "time_series.gif",
        figsize: Tuple[float, float] = (12, 6),
        exclude_patterns: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Create an animated GIF showing how values change over time.

        Args:
            data: Data source (path or DataFrame)
            time_col: Column to use for animation frames (e.g., "period", "year", "year_center")
            time_values: Specific time values to use (if None, uses all unique values)
            value_col: Column with values to plot
            geo_level: "country" or "regional"
            geo_id_col: Column with geographic identifiers
            shapefile_path: Path to shapefile (for regional maps)
            shape_id_col: ID column in shapefile (for regional maps)
            title_template: Template for titles. Placeholders: {value_col}, {time_value}, {time_col}
            units: Units for legend
            vmin: Min value for color scale (auto if None)
            vmax: Max value for color scale (auto if None)
            duration: Duration per frame in milliseconds
            output_filename: Output filename for the GIF
            figsize: Figure size
            exclude_patterns: Patterns to exclude (e.g., ["ATA"])

        Returns:
            Path to saved GIF
        """
        try:
            from PIL import Image
            import geopandas as gpd
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            return None

        # Load data
        df = self.load_data(data)
        if df.empty:
            logger.warning("Empty data, cannot create GIF")
            return None

        # Get time values
        if time_values is None:
            time_values = sorted(df[time_col].unique())
        else:
            # Filter to requested values
            time_values = [t for t in time_values if t in df[time_col].values]

        logger.info(f"Creating time series GIF with {len(time_values)} frames: {time_values}")

        # Calculate global vmin/vmax if not provided
        if vmin is None:
            vmin = df[value_col].min()
        if vmax is None:
            vmax = df[value_col].max()

        logger.info(f"Color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

        # Get geometries
        if geo_level == "country":
            gdf_base = self._get_country_geometries()
        else:
            gdf_base = self._load_shapefile(shapefile_path, simplify_tolerance=0.01)

        if gdf_base is None:
            logger.error("Could not load geometries")
            return None

        # Apply exclusions
        if exclude_patterns:
            id_col = "iso3" if geo_level == "country" else shape_id_col
            if id_col:
                for pattern in exclude_patterns:
                    gdf_base = gdf_base[~gdf_base[id_col].astype(str).str.contains(pattern, na=False)]

        # Project to Robinson
        gdf_base = gdf_base.to_crs(PROJ_ROBINSON)

        # Generate frames
        frame_paths = []
        temp_dir = self.output_dir / "_temp_frames"
        temp_dir.mkdir(exist_ok=True)

        # Get colormap
        cmap, norm = self._get_colormap("standard", vmin, vmax)

        for time_val in time_values:
            # Filter data for this time step
            df_time = df[df[time_col] == time_val]

            # Merge with geometries
            if geo_level == "country":
                gdf = gdf_base.merge(
                    df_time[[geo_id_col, value_col]].drop_duplicates(subset=[geo_id_col]),
                    left_on="iso3",
                    right_on=geo_id_col,
                    how="left"
                )
            else:
                gdf = gdf_base.merge(
                    df_time[[geo_id_col, value_col]].drop_duplicates(subset=[geo_id_col]),
                    left_on=shape_id_col,
                    right_on=geo_id_col,
                    how="left"
                )

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Plot
            gdf.plot(
                column=value_col,
                ax=ax,
                cmap=cmap,
                norm=norm,
                legend=False,
                edgecolor="#d3d3d3",  # Light grey borders for all geometries
                linewidth=0.3,
                missing_kwds={"color": "lightgray", "edgecolor": "#d3d3d3", "linewidth": 0.3}
            )

            # Title
            title = title_template.format(
                value_col=value_col.replace("_", " ").title(),
                time_value=time_val,
                time_col=time_col
            )
            ax.set_title(title, fontsize=14)
            ax.set_axis_off()

            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cax = fig.add_axes([0.25, 0.05, 0.5, 0.04])
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", format=SmartFormatter())
            label = value_col.replace("_", " ").title()
            if units:
                label = f"{label} ({units})"
            cbar.set_label(label, labelpad=8)
            cbar.outline.set_visible(False)

            # Save frame
            frame_path = temp_dir / f"frame_{time_val}.png"
            plt.savefig(frame_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            frame_paths.append(frame_path)

        # Create GIF
        gif_path = self.create_animated_gif(frame_paths, output_filename, duration=duration)

        # Clean up temp files
        for fp in frame_paths:
            try:
                fp.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass

        return gif_path
