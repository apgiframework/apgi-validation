"""Test suite for Theory/APGI_Computational_Benchmarking.py

Rewritten against the module's actual API. The previous version exercised six
methods that have never existed on ComputationalBenchmarker
(register_algorithm, run_comparative_benchmark, calculate_performance_metrics,
test_statistical_significance, run_legacy_benchmark, analyze_scalability), so
every test raised AttributeError and the file provided no coverage at all.

The real surface is:
    ComputationalBenchmarker.run_benchmark_suite(paradigms) -> {name: [BenchmarkResult]}
    ComputationalBenchmarker.generate_comparison_table(results) -> str
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Theory"))
from APGI_Computational_Benchmarking import BenchmarkResult, ComputationalBenchmarker  # noqa: E402

EXPECTED_FRAMEWORKS = {"FEP", "GNW", "IIT", "APGI"}


def paradigm(name="test_paradigm", n=60, seed=0):
    """A minimal, valid benchmark paradigm: a 1-D time series."""
    rng = np.random.default_rng(seed)
    return {"name": name, "inputs": rng.standard_normal(n)}


class TestInitialization:
    def test_benchmarker_constructs(self):
        assert ComputationalBenchmarker() is not None

    def test_exposes_public_api(self):
        benchmarker = ComputationalBenchmarker()
        assert hasattr(benchmarker, "run_benchmark_suite")
        assert hasattr(benchmarker, "generate_comparison_table")

    def test_registers_all_competing_frameworks(self):
        """APGI is benchmarked against FEP, GNW and IIT."""
        assert set(ComputationalBenchmarker().frameworks) == EXPECTED_FRAMEWORKS


class TestBenchmarkSuite:
    @pytest.fixture(scope="class")
    def results(self):
        return ComputationalBenchmarker().run_benchmark_suite([paradigm()])

    def test_every_framework_produces_a_result(self, results):
        assert set(results) == EXPECTED_FRAMEWORKS
        for name, entries in results.items():
            assert len(entries) == 1, f"{name} produced {len(entries)} results for one paradigm"

    def test_results_are_benchmark_result_instances(self, results):
        for entries in results.values():
            assert all(isinstance(e, BenchmarkResult) for e in entries)

    def test_result_fields_are_populated(self, results):
        for name, entries in results.items():
            result = entries[0]
            assert result.framework_name == name
            assert isinstance(result.parameters_used, dict)
            assert result.parameter_count == len(result.parameters_used)
            assert isinstance(result.fit_metrics, dict)
            assert isinstance(result.simulation_data, dict)

    def test_computational_cost_is_non_negative_seconds(self, results):
        for entries in results.values():
            assert isinstance(entries[0].computational_cost, float)
            assert entries[0].computational_cost >= 0.0

    def test_parameter_count_is_positive(self, results):
        for name, entries in results.items():
            assert entries[0].parameter_count > 0, f"{name} reported zero free parameters"

    def test_simulation_data_names_its_framework(self, results):
        for name, entries in results.items():
            assert entries[0].simulation_data.get("framework")

    def test_multiple_paradigms_accumulate(self):
        results = ComputationalBenchmarker().run_benchmark_suite([paradigm("a", seed=1), paradigm("b", seed=2)])
        for name, entries in results.items():
            assert len(entries) == 2, f"{name} did not record both paradigms"

    def test_empty_paradigm_list_yields_empty_results(self):
        results = ComputationalBenchmarker().run_benchmark_suite([])
        assert set(results) == EXPECTED_FRAMEWORKS
        assert all(entries == [] for entries in results.values())


class TestInputValidation:
    """Malformed paradigms must fail clearly at the entry point."""

    def test_two_dimensional_inputs_rejected(self):
        """Frameworks index inputs[t] as a scalar.

        A 2-D array previously failed deep inside FEPFramework.simulate with
        "setting an array element with a sequence".
        """
        bad = {"name": "bad", "inputs": np.random.default_rng(0).standard_normal((20, 3))}
        with pytest.raises(ValueError, match="1-D"):
            ComputationalBenchmarker().run_benchmark_suite([bad])

    def test_empty_inputs_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            ComputationalBenchmarker().run_benchmark_suite([{"name": "bad", "inputs": np.array([])}])

    @pytest.mark.parametrize("missing", ["name", "inputs"])
    def test_missing_required_key_rejected(self, missing):
        entry = paradigm()
        del entry[missing]
        with pytest.raises(KeyError, match=missing):
            ComputationalBenchmarker().run_benchmark_suite([entry])

    def test_list_inputs_accepted(self):
        """A plain Python list is a valid 1-D series."""
        results = ComputationalBenchmarker().run_benchmark_suite(
            [{"name": "list_inputs", "inputs": [0.1, 0.2, 0.3, 0.4, 0.5]}]
        )
        assert all(len(entries) == 1 for entries in results.values())


class TestComparisonTable:
    @pytest.fixture(scope="class")
    def results(self):
        return ComputationalBenchmarker().run_benchmark_suite([paradigm()])

    def test_table_is_non_empty_string(self, results):
        table = ComputationalBenchmarker().generate_comparison_table(results)
        assert isinstance(table, str)
        assert table.strip()

    def test_table_names_every_framework(self, results):
        table = ComputationalBenchmarker().generate_comparison_table(results)
        for name in EXPECTED_FRAMEWORKS:
            assert name in table, f"{name} missing from comparison table"

    def test_empty_results_do_not_crash(self):
        benchmarker = ComputationalBenchmarker()
        table = benchmarker.generate_comparison_table({name: [] for name in EXPECTED_FRAMEWORKS})
        assert isinstance(table, str)


class TestDeterminism:
    def test_same_seed_gives_same_parameter_count(self):
        """Structural properties must not vary run to run."""
        first = ComputationalBenchmarker().run_benchmark_suite([paradigm(seed=7)])
        second = ComputationalBenchmarker().run_benchmark_suite([paradigm(seed=7)])
        for name in EXPECTED_FRAMEWORKS:
            assert first[name][0].parameter_count == second[name][0].parameter_count


class TestPerformance:
    @pytest.mark.performance
    def test_small_paradigm_completes_quickly(self):
        import time

        start = time.time()
        ComputationalBenchmarker().run_benchmark_suite([paradigm(n=100)])
        assert time.time() - start < 60.0, "Benchmark suite unexpectedly slow on a 100-sample series"
