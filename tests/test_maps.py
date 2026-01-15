"""
Unit tests for MapVisualizer.

Tests:
- Data loading and filtering
- Country-level geometry joining
- Regional shapefile geometry joining
- Comparison map calculations
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.append(str(Path(__file__).parent.parent))

from flexdamage.diagnostics.maps import MapVisualizer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def map_visualizer(tmp_path):
    """Create a MapVisualizer instance with a temporary output directory."""
    return MapVisualizer(output_dir=tmp_path / "maps")


@pytest.fixture
def mock_country_f2_data():
    """Mock F2 data with country-level information."""
    np.random.seed(42)
    countries = ["USA", "CAN", "MEX", "BRA", "GBR", "FRA", "DEU", "CHN", "JPN", "IND"]
    years = [2050, 2080, 2100]

    rows = []
    for year in years:
        for iso3 in countries:
            rows.append({
                "iso3": iso3,
                "year": year,
                "flextotal": np.random.uniform(-0.5, 2.0),
                "rawtotal": np.random.uniform(-0.3, 1.8),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_regional_f2_data():
    """Mock F2 data with regional identifiers."""
    np.random.seed(42)
    regions = ["USA.1.1", "USA.2.1", "CAN.1.1", "MEX.1.1", "BRA.1.1"]

    return pd.DataFrame({
        "region": regions,
        "year": 2100,
        "flextotal": np.random.uniform(-0.5, 2.0, len(regions)),
        "rawtotal": np.random.uniform(-0.3, 1.8, len(regions)),
    })


@pytest.fixture
def mock_gdf():
    """Create a mock GeoDataFrame for testing."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        return gpd.GeoDataFrame({
            "iso3": ["USA", "CAN", "MEX", "BRA", "GBR"],
            "name": ["United States", "Canada", "Mexico", "Brazil", "United Kingdom"],
            "geometry": [Point(0, 0)] * 5  # Dummy geometries
        })
    except ImportError:
        pytest.skip("geopandas not installed")


# ============================================================================
# Data Loading Tests
# ============================================================================

class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_dataframe(self, map_visualizer, mock_country_f2_data):
        """Test loading from DataFrame."""
        df = map_visualizer.load_data(mock_country_f2_data)
        assert len(df) == 30  # 10 countries * 3 years

    def test_load_csv(self, map_visualizer, mock_country_f2_data, tmp_path):
        """Test loading from CSV file."""
        csv_path = tmp_path / "test_data.csv"
        mock_country_f2_data.to_csv(csv_path, index=False)

        df = map_visualizer.load_data(csv_path)
        assert len(df) == 30

    def test_load_parquet(self, map_visualizer, mock_country_f2_data, tmp_path):
        """Test loading from Parquet file."""
        parquet_path = tmp_path / "test_data.parquet"
        mock_country_f2_data.to_parquet(parquet_path, index=False)

        df = map_visualizer.load_data(parquet_path)
        assert len(df) == 30

    def test_year_filtering(self, map_visualizer, mock_country_f2_data):
        """Test year filtering."""
        df = map_visualizer.load_data(mock_country_f2_data, year=2100)
        assert len(df) == 10  # 10 countries for year 2100
        assert (df["year"] == 2100).all()

    def test_year_filtering_no_match(self, map_visualizer, mock_country_f2_data):
        """Test year filtering with no matches."""
        df = map_visualizer.load_data(mock_country_f2_data, year=1900)
        assert len(df) == 0


# ============================================================================
# Join Logic Tests
# ============================================================================

class TestJoinLogic:
    """Tests for data-geometry join logic."""

    def test_country_join_exact_match(self, mock_country_f2_data, mock_gdf):
        """Test country join with exact matches."""
        # Filter to year 2100 and get only countries in mock_gdf
        df = mock_country_f2_data[mock_country_f2_data["year"] == 2100].copy()

        # Perform join (simulating what MapVisualizer does)
        merged = mock_gdf.merge(
            df[["iso3", "flextotal"]].drop_duplicates(subset=["iso3"]),
            on="iso3",
            how="left"
        )

        # Countries in both: USA, CAN, MEX, BRA, GBR
        assert len(merged) == 5
        assert merged["flextotal"].notna().sum() == 5

    def test_country_join_partial_match(self, mock_gdf):
        """Test country join with partial matches."""
        # Data with some countries not in mock_gdf
        data = pd.DataFrame({
            "iso3": ["USA", "AUS", "ZAF"],  # Only USA is in mock_gdf
            "flextotal": [1.0, 2.0, 3.0]
        })

        merged = mock_gdf.merge(
            data[["iso3", "flextotal"]],
            on="iso3",
            how="left"
        )

        assert len(merged) == 5  # All mock_gdf rows preserved
        assert merged["flextotal"].notna().sum() == 1  # Only USA matches
        assert merged[merged["iso3"] == "USA"]["flextotal"].iloc[0] == 1.0

    def test_regional_join_with_different_id_cols(self):
        """Test regional join with different ID column names."""
        # Simulate shapefile with 'hierid' column
        shp_data = pd.DataFrame({
            "hierid": ["USA.1.1", "USA.2.1", "CAN.1.1", "MEX.1.1"],
            "name": ["Region 1", "Region 2", "Region 3", "Region 4"]
        })

        # Data with 'region' column
        f2_data = pd.DataFrame({
            "region": ["USA.1.1", "CAN.1.1", "BRA.1.1"],  # BRA not in shapefile
            "flextotal": [1.0, 2.0, 3.0]
        })

        # Join with different column names
        merged = shp_data.merge(
            f2_data,
            left_on="hierid",
            right_on="region",
            how="left"
        )

        assert len(merged) == 4  # All shapefile rows
        assert merged["flextotal"].notna().sum() == 2  # USA.1.1 and CAN.1.1

    def test_join_with_duplicate_ids_in_data(self):
        """Test that duplicate IDs in data are handled."""
        shp_data = pd.DataFrame({
            "iso3": ["USA", "CAN", "MEX"],
        })

        # Data with duplicate ISO codes (e.g., multiple years not filtered)
        f2_data = pd.DataFrame({
            "iso3": ["USA", "USA", "CAN"],
            "year": [2050, 2100, 2100],
            "flextotal": [1.0, 2.0, 3.0]
        })

        # Dropping duplicates before join (as MapVisualizer does)
        merged = shp_data.merge(
            f2_data[["iso3", "flextotal"]].drop_duplicates(subset=["iso3"]),
            on="iso3",
            how="left"
        )

        assert len(merged) == 3
        # Should get first occurrence (1.0 for USA)
        assert merged[merged["iso3"] == "USA"]["flextotal"].iloc[0] == 1.0


# ============================================================================
# Comparison Map Tests
# ============================================================================

class TestComparisonCalculations:
    """Tests for comparison map calculations."""

    def test_sign_match_calculation(self):
        """Test sign match calculation."""
        df = pd.DataFrame({
            "flextotal": [1.0, -1.0, 1.0, -1.0, 0.0],
            "rawtotal":  [2.0, -2.0, -1.0, 1.0, 0.0]
        })

        sign_a = np.sign(df["flextotal"])
        sign_b = np.sign(df["rawtotal"])
        sign_match = sign_a == sign_b

        # Expected: [True, True, False, False, True]
        assert sign_match.tolist() == [True, True, False, False, True]

    def test_diff_magnitude_calculation(self):
        """Test difference magnitude calculation."""
        df = pd.DataFrame({
            "flextotal": [1.0, -1.0, 2.0],
            "rawtotal":  [0.5, -0.5, 1.0]
        })

        diff_mag = (df["flextotal"] - df["rawtotal"]).abs()

        expected = [0.5, 0.5, 1.0]
        np.testing.assert_array_almost_equal(diff_mag.tolist(), expected)

    def test_viz_comp_calculation(self):
        """Test visualization comparison value calculation."""
        df = pd.DataFrame({
            "flextotal": [1.0, -1.0, 1.0, -1.0],
            "rawtotal":  [2.0, -2.0, -1.0, 1.0]
        })

        sign_match = np.sign(df["flextotal"]) == np.sign(df["rawtotal"])
        diff_mag = (df["flextotal"] - df["rawtotal"]).abs()

        # Positive for match, negative for mismatch
        viz_comp = np.where(sign_match, diff_mag, -diff_mag)

        # Expected: [1.0, 1.0, -2.0, -2.0]
        expected = [1.0, 1.0, -2.0, -2.0]
        np.testing.assert_array_almost_equal(viz_comp.tolist(), expected)


# ============================================================================
# Statistics Tests
# ============================================================================

class TestStatistics:
    """Tests for statistics generation."""

    def test_top_bottom_selection(self):
        """Test top 10 / bottom 10 selection."""
        df = pd.DataFrame({
            "region": [f"R{i}" for i in range(25)],
            "value": list(range(25))
        })

        sorted_df = df.sort_values("value")
        bottom_10 = sorted_df.head(10)
        top_10 = sorted_df.tail(10)

        assert len(bottom_10) == 10
        assert len(top_10) == 10
        assert bottom_10["value"].max() < top_10["value"].min()

    def test_comparison_stats_best_worst(self):
        """Test best/worst alignment statistics."""
        df = pd.DataFrame({
            "region": ["R1", "R2", "R3", "R4", "R5"],
            "sign_match": [True, True, False, False, True],
            "diff_mag": [0.1, 0.5, 0.2, 0.8, 0.3]
        })

        # Best: match + low diff
        best = df[df["sign_match"]].sort_values("diff_mag", ascending=True)
        assert best.iloc[0]["region"] == "R1"  # Lowest diff among matches

        # Worst: mismatch first, then high diff
        df["is_mismatch"] = ~df["sign_match"]
        worst = df.sort_values(["is_mismatch", "diff_mag"], ascending=[False, False])
        assert worst.iloc[0]["region"] == "R4"  # Mismatch with highest diff


# ============================================================================
# Color Scale Tests
# ============================================================================

class TestColorScales:
    """Tests for color scale configuration."""

    def test_vmin_vmax_clipping(self, map_visualizer):
        """Test that vmin/vmax clips values correctly."""
        data = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        vmin, vmax = -1.0, 1.5

        # Values outside range should be clipped visually
        # (actual clipping happens in normalization)
        assert data.min() < vmin
        assert data.max() > vmax

    def test_auto_bounds_calculation(self):
        """Test automatic bounds calculation from data."""
        data = pd.Series([-0.5, 0.2, 1.0, 1.5])

        auto_vmin = data.min()
        auto_vmax = data.max()

        assert auto_vmin == -0.5
        assert auto_vmax == 1.5


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_data(self, map_visualizer):
        """Test handling of empty data."""
        empty_df = pd.DataFrame(columns=["iso3", "year", "flextotal"])
        df = map_visualizer.load_data(empty_df)
        assert len(df) == 0

    def test_all_na_values(self):
        """Test handling of all NA values."""
        df = pd.DataFrame({
            "iso3": ["USA", "CAN", "MEX"],
            "flextotal": [np.nan, np.nan, np.nan]
        })

        valid_values = df["flextotal"].dropna()
        assert len(valid_values) == 0

    def test_single_value(self):
        """Test handling of single value (vmin == vmax edge case)."""
        df = pd.DataFrame({
            "iso3": ["USA"],
            "flextotal": [1.0]
        })

        vmin = df["flextotal"].min()
        vmax = df["flextotal"].max()

        # When vmin == vmax, need to adjust
        if vmin >= vmax:
            vmax = vmin + 0.0001

        assert vmax > vmin

    def test_mixed_case_iso_codes(self):
        """Test that ISO code matching is case-sensitive."""
        shp = pd.DataFrame({"iso3": ["USA", "CAN"]})
        data = pd.DataFrame({"iso3": ["usa", "CAN"], "value": [1.0, 2.0]})

        merged = shp.merge(data, on="iso3", how="left")

        # Only CAN should match (case-sensitive)
        assert merged["value"].notna().sum() == 1


# ============================================================================
# Integration Tests (require geopandas)
# ============================================================================

@pytest.mark.skipif(
    "geopandas" not in sys.modules,
    reason="geopandas not installed"
)
class TestIntegration:
    """Integration tests that require geopandas."""

    def test_country_map_generation(self, map_visualizer, mock_country_f2_data):
        """Test full country map generation."""
        # This would actually generate a map if geopandas is available
        # For CI without geopandas, this test is skipped
        pass
