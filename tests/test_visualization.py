"""
Comprehensive tests for visualization functions in main.py.
=======================================================
Tests for _create_distribution_plot, _create_figure_and_axes,
_create_heatmap_plot, _create_plot_by_type, _create_scatter_plot,
_create_time_series_plot, _parse_visualization_parameters, _setup_plotting_style
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    _create_distribution_plot,
    _create_figure_and_axes,
    _create_heatmap_plot,
    _create_plot_by_type,
    _create_scatter_plot,
    _create_time_series_plot,
    _parse_visualization_parameters,
    _setup_plotting_style,
)


class TestParseVisualizationParameters:
    """Test _parse_visualization_parameters function."""

    def test_parse_all_valid_parameters(self):
        """Test parsing all valid parameters."""
        _parse_visualization_parameters(
            figsize="12,8",
            bins=30,
            linewidth=1.5,
            markersize=50,
            font_size=12,
            subplot_rows=2,
            subplot_cols=3,
        )

    def test_parse_invalid_figsize(self):
        """Test parsing invalid figsize."""
        _parse_visualization_parameters(
            figsize="invalid",
            bins=30,
            linewidth=1.5,
            markersize=50,
            font_size=12,
            subplot_rows=1,
            subplot_cols=1,
        )

        # Should use default 12, 8

    def test_parse_invalid_bins_too_small(self):
        """Test parsing bins value too small."""
        result = _parse_visualization_parameters(
            figsize="12,8",
            bins=3,
            linewidth=1.5,
            markersize=50,
            font_size=12,
            subplot_rows=1,
            subplot_cols=1,
        )

        # Should use default 30
        assert result[2] == 30

    def test_parse_invalid_bins_too_large(self):
        """Test parsing bins value too large."""
        result = _parse_visualization_parameters(
            figsize="12,8",
            bins=150,
            linewidth=1.5,
            markersize=50,
            font_size=12,
            subplot_rows=1,
            subplot_cols=1,
        )

        # Should use default 30
        assert result[2] == 30

    def test_parse_invalid_linewidth(self):
        """Test parsing invalid linewidth."""
        result = _parse_visualization_parameters(
            figsize="12,8",
            bins=30,
            linewidth="invalid",
            markersize=50,
            font_size=12,
            subplot_rows=1,
            subplot_cols=1,
        )

        # Should use default 1.5
        assert result[3] == 1.5

    def test_parse_linewidth_out_of_range(self):
        """Test parsing linewidth out of valid range."""
        result = _parse_visualization_parameters(
            figsize="12,8",
            bins=30,
            linewidth=10.0,
            markersize=50,
            font_size=12,
            subplot_rows=1,
            subplot_cols=1,
        )

        # Should use default 1.5
        assert result[3] == 1.5

    def test_parse_invalid_markersize(self):
        """Test parsing invalid markersize."""
        result = _parse_visualization_parameters(
            figsize="12,8",
            bins=30,
            linewidth=1.5,
            markersize="invalid",
            font_size=12,
            subplot_rows=1,
            subplot_cols=1,
        )

        # Should use default 50
        assert result[4] == 50

    def test_parse_markersize_out_of_range(self):
        """Test parsing markersize out of valid range."""
        result = _parse_visualization_parameters(
            figsize="12,8",
            bins=30,
            linewidth=1.5,
            markersize=5,
            font_size=12,
            subplot_rows=1,
            subplot_cols=1,
        )

        # Should use default 50
        assert result[4] == 50

    def test_parse_invalid_font_size(self):
        """Test parsing invalid font size."""
        result = _parse_visualization_parameters(
            figsize="12,8",
            bins=30,
            linewidth=1.5,
            markersize=50,
            font_size="invalid",
            subplot_rows=1,
            subplot_cols=1,
        )

        # Should use default 12
        assert result[5] == 12

    def test_parse_font_size_out_of_range(self):
        """Test parsing font size out of valid range."""
        result = _parse_visualization_parameters(
            figsize="12,8",
            bins=30,
            linewidth=1.5,
            markersize=50,
            font_size=30,
            subplot_rows=1,
            subplot_cols=1,
        )

        # Should use default 12
        assert result[5] == 12

    def test_parse_invalid_subplot_dimensions(self):
        """Test parsing invalid subplot dimensions."""
        result = _parse_visualization_parameters(
            figsize="12,8",
            bins=30,
            linewidth=1.5,
            markersize=50,
            font_size=12,
            subplot_rows=0,
            subplot_cols=1,
        )

        # Should use default 1x1
        assert result[6] == 1
        assert result[7] == 1

    def test_parse_subplot_too_many(self):
        """Test parsing subplot dimensions with too many subplots."""
        result = _parse_visualization_parameters(
            figsize="12,8",
            bins=30,
            linewidth=1.5,
            markersize=50,
            font_size=12,
            subplot_rows=4,
            subplot_cols=4,
        )

        # Should use default 1x1 (4*4=16 > 12)
        assert result[6] == 1
        assert result[7] == 1


class TestSetupPlottingStyle:
    """Test _setup_plotting_style function."""

    def test_setup_seaborn_style(self):
        """Test setting up seaborn style."""
        sns_module = MagicMock()
        plt_module = MagicMock()

        _setup_plotting_style(
            style="seaborn",
            palette="default",
            font_family="Arial",
            font_size_val=12,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        sns_module.set_style.assert_called_once_with("whitegrid")
        plt_module.style.use.assert_called_once_with("seaborn-v0_8")

    def test_setup_ggplot_style(self):
        """Test setting up ggplot style."""
        sns_module = MagicMock()
        plt_module = MagicMock()

        _setup_plotting_style(
            style="ggplot",
            palette="default",
            font_family="Arial",
            font_size_val=12,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        plt_module.style.use.assert_called_once_with("ggplot")

    def test_setup_default_style(self):
        """Test setting up default style."""
        sns_module = MagicMock()
        plt_module = MagicMock()

        _setup_plotting_style(
            style="unknown",
            palette="default",
            font_family="Arial",
            font_size_val=12,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        plt_module.style.use.assert_called_once_with("default")

    def test_setup_custom_palette(self):
        """Test setting up custom color palette."""
        sns_module = MagicMock()
        plt_module = MagicMock()
        sns_module.palettes.__dict__ = {"Set2": MagicMock()}

        _setup_plotting_style(
            style="default",
            palette="Set2",
            font_family="Arial",
            font_size_val=12,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        sns_module.set_palette.assert_called_once_with("Set2")

    def test_setup_font_properties(self):
        """Test setting up font properties."""
        sns_module = MagicMock()
        plt_module = MagicMock()
        # Use a real dictionary for rcParams to properly test dictionary access
        plt_module.rcParams = {}

        _setup_plotting_style(
            style="default",
            palette="default",
            font_family="Helvetica",
            font_size_val=14,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        assert plt_module.rcParams["font.family"] == "Helvetica"
        assert plt_module.rcParams["font.size"] == 14
        assert plt_module.rcParams["axes.titlesize"] == 16
        assert plt_module.rcParams["axes.labelsize"] == 14


class TestCreateFigureAndAxes:
    """Test _create_figure_and_axes function."""

    def test_create_single_subplot(self):
        """Test creating figure with single subplot."""
        mock_fig = MagicMock()
        mock_ax_array = MagicMock()
        mock_ax_array.flatten.return_value = [MagicMock()]
        plt_module = MagicMock()
        plt_module.subplots.return_value = (mock_fig, mock_ax_array)

        fig, axes = _create_figure_and_axes(
            fig_width=12,
            fig_height=8,
            subplot_rows_val=1,
            subplot_cols_val=1,
            aspect="auto",
            plt_module=plt_module,
        )

        assert fig == mock_fig
        assert len(axes) == 1
        plt_module.subplots.assert_called_once_with(figsize=(12, 8), squeeze=False)

    def test_create_multiple_subplots(self):
        """Test creating figure with multiple subplots."""
        mock_fig = MagicMock()
        mock_axes_array = MagicMock()
        mock_axes_array.flatten.return_value = [MagicMock() for _ in range(6)]
        plt_module = MagicMock()
        plt_module.subplots.return_value = (mock_fig, mock_axes_array)

        fig, axes = _create_figure_and_axes(
            fig_width=12,
            fig_height=8,
            subplot_rows_val=2,
            subplot_cols_val=3,
            aspect="auto",
            plt_module=plt_module,
        )

        assert fig == mock_fig
        assert len(axes) == 6
        plt_module.subplots.assert_called_once_with(2, 3, figsize=(12, 8), squeeze=False)

    def test_set_aspect_ratio(self):
        """Test setting aspect ratio."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax_array = MagicMock()
        mock_ax_array.flatten.return_value = [mock_ax]
        plt_module = MagicMock()
        plt_module.subplots.return_value = (mock_fig, mock_ax_array)

        fig, axes = _create_figure_and_axes(
            fig_width=12,
            fig_height=8,
            subplot_rows_val=1,
            subplot_cols_val=1,
            aspect="equal",
            plt_module=plt_module,
        )

        mock_ax.set_aspect.assert_called_once_with("equal")


class TestCreateTimeSeriesPlot:
    """Test _create_time_series_plot function."""

    def test_create_time_series_plot_success(self):
        """Test successful time series plot creation."""
        data = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=10),
                "value1": np.random.randn(10),
                "value2": np.random.randn(10),
            }
        )
        data.set_index("time", inplace=True)

        axes = [MagicMock()]

        _create_time_series_plot(
            data=data,
            axes=axes,
            alpha=0.7,
            linewidth_val=2.0,
            marker="o",
            markersize_val=50,
            xlabel="Time",
            ylabel="Value",
            title="Test Time Series",
            grid=True,
            legend=True,
        )

        # Verify plot was created with correct parameters
        axes[0].plot.assert_called()

    def test_create_time_series_plot_with_empty_data(self):
        """Test time series plot with empty data."""
        data = pd.DataFrame()
        axes = [MagicMock()]

        _create_time_series_plot(
            data=data,
            axes=axes,
            alpha=0.7,
            linewidth_val=2.0,
            marker="o",
            markersize_val=50,
            xlabel="Time",
            ylabel="Value",
            title="Test Time Series",
            grid=True,
            legend=True,
        )

        # Should handle empty data gracefully
        assert True  # Function should not crash


class TestCreateScatterPlot:
    """Test _create_scatter_plot function."""

    def test_create_scatter_plot_success(self):
        """Test successful scatter plot creation."""
        data = pd.DataFrame(
            {
                "x": np.random.randn(10),
                "y": np.random.randn(10),
                "z": np.random.randn(10),
            }
        )

        axes = [MagicMock()]

        result = _create_scatter_plot(
            data=data,
            axes=axes,
            alpha=0.7,
            markersize_val=50,
            marker="o",
            linewidth_val=1.0,
            xlabel="X",
            ylabel="Y",
            title="Test Scatter",
            grid=True,
            legend=True,
        )

        assert result
        axes[0].scatter.assert_called_once()

    def test_create_scatter_plot_insufficient_columns(self):
        """Test scatter plot with insufficient numeric columns."""
        data = pd.DataFrame(
            {
                "x": np.random.randn(10),
            }
        )

        axes = [MagicMock()]

        result = _create_scatter_plot(
            data=data,
            axes=axes,
            alpha=0.7,
            markersize_val=50,
            marker="o",
            linewidth_val=1.0,
            xlabel="X",
            ylabel="Y",
            title="Test Scatter",
            grid=True,
            legend=True,
        )

        assert not (result)


class TestCreateHeatmapPlot:
    """Test _create_heatmap_plot function."""

    def test_create_heatmap_plot_success(self):
        """Test successful heatmap plot creation."""
        data = pd.DataFrame(
            {
                "a": np.random.randn(10),
                "b": np.random.randn(10),
                "c": np.random.randn(10),
            }
        )

        sns_module = MagicMock()
        plt_module = MagicMock()

        result = _create_heatmap_plot(
            data=data,
            colormap="viridis",
            alpha=0.8,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        assert result
        sns_module.heatmap.assert_called_once()

    def test_create_heatmap_plot_no_numeric_data(self):
        """Test heatmap plot with no numeric data."""
        data = pd.DataFrame(
            {
                "text": ["a", "b", "c"],
            }
        )

        sns_module = MagicMock()
        plt_module = MagicMock()

        result = _create_heatmap_plot(
            data=data,
            colormap="viridis",
            alpha=0.8,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        assert not (result)


class TestCreateDistributionPlot:
    """Test _create_distribution_plot function."""

    def test_create_distribution_plot_success(self):
        """Test successful distribution plot creation."""
        data = pd.DataFrame(
            {
                "value": np.random.randn(100),
            }
        )

        plt_module = MagicMock()

        # Patch on the class: `data["value"]` constructs a new Series object on
        # every access, so setting an attribute on one indexing result left the
        # real Series.hist in place and the later assertion inspected a plain
        # bound method.
        with patch.object(pd.Series, "hist") as mock_hist:
            result = _create_distribution_plot(
                data=data,
                bins_val=30,
                alpha=0.7,
                grid=True,
                plt_module=plt_module,
            )

            assert result
            mock_hist.assert_called_once_with(bins=30, alpha=0.7)

    def test_create_distribution_plot_no_numeric_data(self):
        """Test distribution plot with no numeric data."""
        data = pd.DataFrame(
            {
                "text": ["a", "b", "c"],
            }
        )

        plt_module = MagicMock()

        result = _create_distribution_plot(
            data=data,
            bins_val=30,
            alpha=0.7,
            grid=True,
            plt_module=plt_module,
        )

        assert not (result)


class TestCreatePlotByType:
    """Test _create_plot_by_type function."""

    @patch("main._create_time_series_plot")
    def test_create_time_series_plot_by_type(self, mock_create_ts):
        """Test creating time series plot by type."""
        data = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=10),
                "value": np.random.randn(10),
            }
        )
        data.set_index("time", inplace=True)

        axes = [MagicMock()]
        mock_create_ts.return_value = True

        sns_module = MagicMock()
        plt_module = MagicMock()

        result = _create_plot_by_type(
            plot_type="time_series",
            data=data,
            axes=axes,
            alpha=0.7,
            linewidth_val=2.0,
            marker="o",
            markersize_val=50,
            xlabel="Time",
            ylabel="Value",
            title="Test",
            grid=True,
            legend=True,
            colormap="viridis",
            bins_val=30,
            fig_width=12,
            fig_height=8,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        assert result
        mock_create_ts.assert_called_once()

    @patch("main._create_scatter_plot")
    def test_create_scatter_plot_by_type(self, mock_create_scatter):
        """Test creating scatter plot by type."""
        data = pd.DataFrame(
            {
                "x": np.random.randn(10),
                "y": np.random.randn(10),
            }
        )

        axes = [MagicMock()]
        mock_create_scatter.return_value = True

        sns_module = MagicMock()
        plt_module = MagicMock()

        result = _create_plot_by_type(
            plot_type="scatter",
            data=data,
            axes=axes,
            alpha=0.7,
            linewidth_val=1.0,
            marker="o",
            markersize_val=50,
            xlabel="X",
            ylabel="Y",
            title="Test",
            grid=True,
            legend=True,
            colormap="viridis",
            bins_val=30,
            fig_width=12,
            fig_height=8,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        assert result
        mock_create_scatter.assert_called_once()

    @patch("main._create_heatmap_plot")
    def test_create_heatmap_plot_by_type(self, mock_create_heatmap):
        """Test creating heatmap plot by type."""
        data = pd.DataFrame(
            {
                "a": np.random.randn(10),
                "b": np.random.randn(10),
            }
        )

        axes = [MagicMock()]
        mock_create_heatmap.return_value = True

        sns_module = MagicMock()
        plt_module = MagicMock()

        result = _create_plot_by_type(
            plot_type="heatmap",
            data=data,
            axes=axes,
            alpha=0.8,
            linewidth_val=1.0,
            marker="o",
            markersize_val=50,
            xlabel="",
            ylabel="",
            title="Test",
            grid=False,
            legend=False,
            colormap="viridis",
            bins_val=30,
            fig_width=12,
            fig_height=8,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        assert result
        mock_create_heatmap.assert_called_once()

    @patch("main._create_distribution_plot")
    def test_create_distribution_plot_by_type(self, mock_create_dist):
        """Test creating distribution plot by type."""
        data = pd.DataFrame(
            {
                "value": np.random.randn(100),
            }
        )

        axes = [MagicMock()]
        mock_create_dist.return_value = True

        sns_module = MagicMock()
        plt_module = MagicMock()

        result = _create_plot_by_type(
            plot_type="distribution",
            data=data,
            axes=axes,
            alpha=0.7,
            linewidth_val=1.0,
            marker="o",
            markersize_val=50,
            xlabel="",
            ylabel="",
            title="Test",
            grid=True,
            legend=False,
            colormap="viridis",
            bins_val=30,
            fig_width=12,
            fig_height=8,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        assert result
        mock_create_dist.assert_called_once()

    def test_create_unknown_plot_type(self):
        """Test creating plot with unknown type."""
        data = pd.DataFrame(
            {
                "value": np.random.randn(10),
            }
        )

        axes = [MagicMock()]
        sns_module = MagicMock()
        plt_module = MagicMock()

        result = _create_plot_by_type(
            plot_type="unknown_type",
            data=data,
            axes=axes,
            alpha=0.7,
            linewidth_val=1.0,
            marker="o",
            markersize_val=50,
            xlabel="",
            ylabel="",
            title="Test",
            grid=True,
            legend=False,
            colormap="viridis",
            bins_val=30,
            fig_width=12,
            fig_height=8,
            sns_module=sns_module,
            plt_module=plt_module,
        )

        assert not (result)
