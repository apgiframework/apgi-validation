"""Specification 1.1.0 data contract: loader fields, verdicts and registry paths."""

from pathlib import Path

import pytest

from utils.protocol_loader import (
    SPECIFICATION_VERSION,
    Verdict,
    classify_prediction,
    load_all_protocols,
    load_protocol,
    load_protocol_template,
    spec_prediction_results,
)
from utils.protocol_registry import PROTOCOL_REGISTRY
from utils.protocol_schema import PredictionResult, PredictionStatus

ROOT = Path(__file__).resolve().parent.parent


class TestSpecificationRelease:
    def test_every_protocol_is_release_1_1_0(self):
        specs = load_all_protocols()
        assert sorted(specs) == [f"APGI-P0{n}" for n in range(8)]
        for spec in specs.values():
            assert spec.version == SPECIFICATION_VERSION
            assert spec.is_current_specification

    def test_study_status_and_blocking_issues_are_mapped(self):
        spec = load_protocol("APGI-P07")
        assert spec.validation_status == "pre_collection"
        assert spec.caveats == spec.to_dict()["submission_blocking_issues"]

    def test_kappa_is_uncalibrated(self):
        for spec in load_all_protocols().values():
            kappa = spec.get_parameter("kappa")
            assert kappa["value"] is None
            assert kappa["status"] == "uncalibrated"

    def test_sub_prediction_fields(self):
        spec = load_protocol("APGI-P01")
        pred = spec.get_prediction("Pred 1.A")
        assert pred.status == "confirmatory"
        assert pred.is_confirmatory
        assert "not evidence of falsification" in pred.indeterminate_rule
        assert pred.confirming_thresholds["alpha"] == pytest.approx(0.0125)
        assert not spec.get_prediction("Pred 1.E").is_confirmatory

    def test_cohort_sizes(self):
        people = load_protocol("APGI-P01").participants
        assert (people["n_target"], people["n_analyzable_target"], people["n_minimum"]) == (62, 59, 48)


class TestVerdicts:
    @pytest.mark.parametrize(
        "confirmed, falsified, expected",
        [
            (True, False, Verdict.CONFIRMED),
            (True, True, Verdict.CONFIRMED),
            (False, True, Verdict.FALSIFIED),
            (False, False, Verdict.INDETERMINATE),
        ],
    )
    def test_failure_to_confirm_is_not_falsification(self, confirmed, falsified, expected):
        assert classify_prediction(confirmed, falsified) is expected

    def test_spec_prediction_results_links_and_defaults(self):
        spec = load_protocol("APGI-P06")
        named = {
            "V20.1": PredictionResult(passed=True, value=0.8, threshold=0.5, status=PredictionStatus.PASSED),
            "V20.2": PredictionResult(passed=False, value=0.1, threshold=0.5, status=PredictionStatus.FAILED),
        }
        results = spec_prediction_results(spec, named, {"Pred 6.A": "V20.1", "Pred 6.C": "V20.2"}, ["test"])

        assert set(results) == {f"Pred 6.{c}" for c in "ABCDE"}
        assert results["Pred 6.A"].status is PredictionStatus.PASSED
        assert results["Pred 6.A"].metadata["verdict"] == "confirmed"
        assert results["Pred 6.A"].value == 0.8
        # A failed simulation check and an unlinked prediction are both indeterminate.
        for pid in ("Pred 6.B", "Pred 6.C", "Pred 6.D", "Pred 6.E"):
            assert results[pid].status is PredictionStatus.INDETERMINATE
            assert results[pid].metadata["verdict"] == "indeterminate"
            assert not results[pid].passed
        assert results["Pred 6.C"].sources == ["APGI-P06", "test"]

    def test_spec_prediction_results_without_spec(self):
        assert spec_prediction_results(None, {}, {}, []) == {}


class TestRegistryPaths:
    def test_experimental_protocol_files_exist(self):
        infos = PROTOCOL_REGISTRY.list_protocols("ExperimentalProtocol")
        assert {i.canonical_id for i in infos} >= {f"APGI-P0{n}" for n in range(8)}
        for info in infos:
            assert (ROOT / info.filename).is_file(), info.filename

    def test_registry_matches_protocol_ids(self):
        for pid, spec in load_all_protocols().items():
            info = PROTOCOL_REGISTRY.get_protocol(pid)
            assert info is not None
            assert load_protocol(pid).to_dict() == spec.to_dict()
            assert Path(info.filename).name.startswith(f"protocol_{pid[-1]}_")

    def test_legacy_template_moved_out_of_release(self):
        assert not (ROOT / "protocols" / "apgi_protocol_template.json").exists()
        assert load_protocol_template()["apgi_validation_metadata"]["protocols"]
