"""
Tests for untested visualization functions in main.py
====================================================
Comprehensive tests for 9 visualization functions.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    _load_visualization_data,
    _parse_visualization_parameters,
    _setup_plotting_style,
)


class TestLoadVisualizationData:
    """Test _load_visualization_data function."""

    @patch("main.pd.read_csv")
    def test_load_visualization_data_success(self, mock_read_csv, tmp_path):
        """Test successful data loading."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\n1,2\n3,4\n")

        mock_read_csv.return_value = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})

        result = _load_visualization_data(str(test_file))

        assert result is not None

    @patch("main.pd.read_csv")
    def test_load_visualization_data_file_not_found(self, mock_read_csv, tmp_path):
        """Test data loading with non-existent file."""
        test_file = tmp_path / "nonexistent.csv"
        mock_read_csv.side_effect = FileNotFoundError()

        result = _load_visualization_data(str(test_file))

        assert result is None

    @patch("main.pd.read_csv")
    def test_load_visualization_data_parser_error(self, mock_read_csv, tmp_path):
        """Test data loading with parser error."""
        test_file = tmp_path / "invalid.csv"
        mock_read_csv.side_effect = pd.errors.ParserError("Invalid CSV")

        result = _load_visualization_data(str(test_file))

        assert result is None


class TestParseVisualizationParameters:
    """Test _parse_visualization_parameters function."""

    def test_parse_visualization_parameters_valid(self):
        """Test parsing valid parameters."""
        _parse_visualization_parameters("12,8", "30", "1.5", "50", "12", "1", "1")

    def test_parse_visualization_parameters_invalid_figsize(self):
        """Test parsing with invalid figsize."""
        result = _parse_visualization_parameters("invalid", "30", "1.5", "50", "12", "1", "1")

        assert result[0] == 12  # Default width
        assert result[1] == 8  # Default height

    def test_parse_visualization_parameters_invalid_bins(self):
        """Test parsing with invalid bins."""
        result = _parse_visualization_parameters("12,8", "200", "1.5", "50", "12", "1", "1")

        assert result[2] == 30  # Default bins

    def test_parse_visualization_parameters_invalid_linewidth(self):
        """Test parsing with invalid linewidth."""
        result = _parse_visualization_parameters("12,8", "30", "10", "50", "12", "1", "1")

        assert result[3] == 1.5  # Default linewidth

    def test_parse_visualization_parameters_invalid_markersize(self):
        """Test parsing with invalid markersize."""
        result = _parse_visualization_parameters("12,8", "30", "1.5", "500", "12", "1", "1")

        assert result[4] == 50.0  # Default markersize

    def test_parse_visualization_parameters_invalid_font_size(self):
        """Test parsing with invalid font size."""
        result = _parse_visualization_parameters("12,8", "30", "1.5", "50", "30", "1", "1")

        assert result[5] == 12  # Default font size

    def test_parse_visualization_parameters_invalid_subplots(self):
        """Test parsing with invalid subplot dimensions."""
        result = _parse_visualization_parameters("12,8", "30", "1.5", "50", "12", "5", "5")

        assert result[6] == 1  # Default rows
        assert result[7] == 1  # Default cols


class TestSetupPlottingStyle:
    """Test _setup_plotting_style function."""

    def test_setup_plotting_style_seaborn(self):
        """Test setting up seaborn style."""
        mock_sns = MagicMock()
        mock_plt = MagicMock()
        _setup_plotting_style("seaborn", "Set2", "Arial", 12, mock_sns, mock_plt)

        mock_sns.set_style.assert_called_once_with("whitegrid")
        mock_plt.style.use.assert_called()

    def test_setup_plotting_style_ggplot(self):
        """Test setting up ggplot style."""
        mock_sns = MagicMock()
        mock_plt = MagicMock()
        _setup_plotting_style("ggplot", "default", "Arial", 12, mock_sns, mock_plt)

        mock_plt.style.use.assert_called_with("ggplot")

    def test_setup_plotting_style_default(self):
        """Test setting up default style."""
        mock_sns = MagicMock()
        mock_plt = MagicMock()
        _setup_plotting_style("invalid", "default", "Arial", 12, mock_sns, mock_plt)

        mock_plt.style.use.assert_called_with("default")


class TestCreateFigureAndAxes:
    """Test _create_figure_and_axes function."""

    def test_create_figure_and_axes_single(self):
        """Test creating single figure and axes."""
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_ax_flat = [MagicMock()]
        mock_ax = MagicMock()
        mock_ax.flatten.return_value = mock_ax_flat
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig, axes = _create_figure_and_axes(12, 8, 1, 1, "auto", mock_plt)

        assert fig == mock_fig
        assert list(axes) == mock_ax_flat

    def test_create_figure_and_axes_multiple(self):
        """Test creating multiple subplots."""
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_axes_flat = [MagicMock() for _ in range(4)]
        mock_axes_array = MagicMock()
        mock_axes_array.flatten.return_value = mock_axes_flat
        mock_plt.subplots.return_value = (mock_fig, mock_axes_array)

        fig, axes = _create_figure_and_axes(12, 8, 2, 2, "auto", mock_plt)

        assert fig == mock_fig
        assert len(axes) == 4


class TestCreateTimeSeriesPlot:
    """Test _create_time_series_plot function."""

    def test_create_time_series_plot_success(self):
        """Test creating time series plot successfully."""
        data = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [2, 4, 6, 8, 10]})
        axes = [MagicMock()]

        _create_time_series_plot(data, axes, 0.5, 1.5, "o", 50, "Time", "Value", "Test Plot", True, True)

        assert axes[0].plot.called

    def test_create_time_series_plot_no_numeric(self):
        """Test creating time series plot with no numeric data."""
        data = pd.DataFrame({"col1": ["a", "b", "c"]})
        axes = [MagicMock()]

        _create_time_series_plot(data, axes, 0.5, 1.5, "o", 50, "Time", "Value", "Test Plot", True, True)

        # Should not crash, just not plot anything


class TestCreateScatterPlot:
    """Test _create_scatter_plot function."""

    @patch("main.console")
    def test_create_scatter_plot_success(self, mock_console):
        """Test creating scatter plot successfully."""
        data = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [2, 4, 6, 8, 10]})
        axes = [MagicMock()]

        result = _create_scatter_plot(data, axes, 0.5, 50, "o", 1.5, "X", "Y", "Test Plot", True, True)

        assert result
        assert axes[0].scatter.called

    @patch("main.console")
    def test_create_scatter_plot_insufficient_columns(self, mock_console):
        """Test creating scatter plot with insufficient columns."""
        data = pd.DataFrame({"col1": [1, 2, 3]})
        axes = [MagicMock()]

        result = _create_scatter_plot(data, axes, 0.5, 50, "o", 1.5, "X", "Y", "Test Plot", True, True)

        assert not (result)


class TestCreateHeatmapPlot:
    """Test _create_heatmap_plot function."""

    def test_create_heatmap_plot_success(self):
        """Test creating heatmap plot successfully."""
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [2, 4, 6], "col3": [3, 6, 9]})
        mock_sns = MagicMock()
        mock_plt = MagicMock()

        result = _create_heatmap_plot(data, "viridis", 0.8, mock_sns, mock_plt)

        assert result
        assert mock_sns.heatmap.called

    def test_create_heatmap_plot_no_numeric(self):
        """Test creating heatmap plot with no numeric data."""
        data = pd.DataFrame({"col1": ["a", "b", "c"]})
        mock_sns = MagicMock()
        mock_plt = MagicMock()

        result = _create_heatmap_plot(data, "viridis", 0.8, mock_sns, mock_plt)

        assert not (result)


class TestCreateDistributionPlot:
    """Test _create_distribution_plot function."""

    def test_create_distribution_plot_success(self):
        """Test creating distribution plot successfully."""
        data = pd.DataFrame({"col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        mock_plt = MagicMock()

        result = _create_distribution_plot(data, 30, 0.8, True, mock_plt)

        assert result

    def test_create_distribution_plot_no_numeric(self):
        """Test creating distribution plot with no numeric data."""
        data = pd.DataFrame({"col1": ["a", "b", "c"]})
        mock_plt = MagicMock()

        result = _create_distribution_plot(data, 30, 0.8, True, mock_plt)

        assert not (result)


class TestCreatePlotByType:
    """Test _create_plot_by_type function."""

    @patch("main._create_time_series_plot")
    def test_create_plot_by_type_time_series(self, mock_time_series):
        """Test creating time series plot by type."""
        data = pd.DataFrame({"col1": [1, 2, 3]})
        axes = [MagicMock()]
        mock_sns = MagicMock()
        mock_plt = MagicMock()
        mock_time_series.return_value = True

        result = _create_plot_by_type(
            "time_series",
            data,
            axes,
            0.5,
            1.5,
            "o",
            50,
            "X",
            "Y",
            "Test",
            True,
            True,
            "viridis",
            30,
            12,
            8,
            mock_sns,
            mock_plt,
        )

        assert result
        mock_time_series.assert_called_once()

    @patch("main._create_scatter_plot")
    def test_create_plot_by_type_scatter(self, mock_scatter):
        """Test creating scatter plot by type."""
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [2, 4, 6]})
        axes = [MagicMock()]
        mock_sns = MagicMock()
        mock_plt = MagicMock()
        mock_scatter.return_value = True

        result = _create_plot_by_type(
            "scatter",
            data,
            axes,
            0.5,
            1.5,
            "o",
            50,
            "X",
            "Y",
            "Test",
            True,
            True,
            "viridis",
            30,
            12,
            8,
            mock_sns,
            mock_plt,
        )

        assert result
        mock_scatter.assert_called_once()

    @patch("main._create_heatmap_plot")
    def test_create_plot_by_type_heatmap(self, mock_heatmap):
        """Test creating heatmap plot by type."""
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [2, 4, 6]})
        axes = [MagicMock()]
        mock_sns = MagicMock()
        mock_plt = MagicMock()
        mock_heatmap.return_value = True

        result = _create_plot_by_type(
            "heatmap",
            data,
            axes,
            0.5,
            1.5,
            "o",
            50,
            "X",
            "Y",
            "Test",
            True,
            True,
            "viridis",
            30,
            12,
            8,
            mock_sns,
            mock_plt,
        )

        assert result
        mock_heatmap.assert_called_once()

    @patch("main._create_distribution_plot")
    def test_create_plot_by_type_distribution(self, mock_distribution):
        """Test creating distribution plot by type."""
        data = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        axes = [MagicMock()]
        mock_sns = MagicMock()
        mock_plt = MagicMock()
        mock_distribution.return_value = True

        result = _create_plot_by_type(
            "distribution",
            data,
            axes,
            0.5,
            1.5,
            "o",
            50,
            "X",
            "Y",
            "Test",
            True,
            True,
            "viridis",
            30,
            12,
            8,
            mock_sns,
            mock_plt,
        )

        assert result
        mock_distribution.assert_called_once()

    def test_create_plot_by_type_invalid(self):
        """Test creating plot with invalid type."""
        data = pd.DataFrame({"col1": [1, 2, 3]})
        axes = [MagicMock()]
        mock_sns = MagicMock()
        mock_plt = MagicMock()

        result = _create_plot_by_type(
            "invalid",
            data,
            axes,
            0.5,
            1.5,
            "o",
            50,
            "X",
            "Y",
            "Test",
            True,
            True,
            "viridis",
            30,
            12,
            8,
            mock_sns,
            mock_plt,
        )

        assert not (result)
