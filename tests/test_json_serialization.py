"""Tests for canonical JSON sanitisation of protocol results.

Historical defect: protocol results carried NumPy scalars into
``json.dumps``. Under NumPy 2.x ``np.True_`` reports ``__class__.__name__ ==
"bool"`` without being a Python ``bool``, so serialisation failed with
"Object of type bool is not JSON serializable" — and results written to
``*_results.json`` could not be produced at all.
"""

import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from utils.json_serialization import APGIJSONEncoder, dump_json, dumps_json, to_json_safe


class TestNumpyScalars:
    def test_numpy_bool_becomes_python_bool(self):
        """The exact type that broke serialisation under NumPy 2.x."""
        result = to_json_safe(np.True_)
        assert result is True
        assert isinstance(result, bool)

    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint8])
    def test_numpy_integers(self, dtype):
        value = to_json_safe(dtype(7))
        assert value == 7
        assert isinstance(value, int)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_numpy_floats(self, dtype):
        value = to_json_safe(dtype(1.5))
        assert value == pytest.approx(1.5)
        assert isinstance(value, float)

    def test_numpy_complex(self):
        assert to_json_safe(np.complex128(1 + 2j)) == {"real": 1.0, "imag": 2.0}

    def test_numpy_array_becomes_list(self):
        assert to_json_safe(np.array([1, 2, 3])) == [1, 2, 3]

    def test_nested_numpy_array(self):
        assert to_json_safe(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]

    def test_bool_array(self):
        result = to_json_safe(np.array([True, False]))
        assert result == [True, False]
        assert all(isinstance(item, bool) for item in result)


class TestNonFiniteValues:
    """NaN and Infinity are not valid JSON."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_python_non_finite_becomes_null(self, value):
        assert to_json_safe(value) is None

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_numpy_non_finite_becomes_null(self, value):
        assert to_json_safe(np.float64(value)) is None

    def test_output_is_strict_json(self):
        payload = dumps_json({"x": float("nan")})
        assert "NaN" not in payload
        json.loads(payload)  # strict parse


class TestContainers:
    def test_nested_dict(self):
        data = {"a": {"b": {"significant": np.True_, "p": np.float64(0.01)}}}
        assert to_json_safe(data) == {"a": {"b": {"significant": True, "p": 0.01}}}

    def test_tuple_and_set_become_lists(self):
        assert to_json_safe((1, 2)) == [1, 2]
        assert sorted(to_json_safe({1, 2})) == [1, 2]

    def test_non_string_keys_become_strings(self):
        assert to_json_safe({1: "a", None: "b"}) == {"1": "a", "None": "b"}

    def test_list_of_numpy_scalars(self):
        assert to_json_safe([np.int64(1), np.True_]) == [1, True]

    def test_deeply_nested_structure_round_trips(self):
        data = {"results": {"V2.3": {"comparison": {"significant": np.bool_(False)}}}}
        assert json.loads(dumps_json(data))["results"]["V2.3"]["comparison"]["significant"] is False


class TestOtherTypes:
    def test_datetime_isoformat(self):
        moment = datetime(2026, 7, 30, 12, 0, 0)
        assert to_json_safe(moment) == moment.isoformat()

    def test_path_becomes_string(self):
        assert to_json_safe(Path("/tmp/x.json")) == "/tmp/x.json"

    def test_decimal_becomes_float(self):
        assert to_json_safe(Decimal("1.25")) == 1.25

    def test_enum_uses_value(self):
        class Status(Enum):
            OK = "ok"

        assert to_json_safe(Status.OK) == "ok"

    def test_bytes_decoded(self):
        assert to_json_safe(b"hello") == "hello"

    def test_object_uses_public_attributes(self):
        class Result:
            def __init__(self):
                self.passed = np.True_
                self._private = "hidden"

        converted = to_json_safe(Result())
        assert converted == {"passed": True}

    def test_unknown_type_falls_back_to_string(self):
        assert isinstance(to_json_safe(object()), str)

    def test_plain_types_pass_through(self):
        for value in ("s", 1, 1.5, True, None):
            assert to_json_safe(value) == value


class TestEncoderAndHelpers:
    def test_encoder_handles_numpy(self):
        assert json.loads(json.dumps({"x": np.True_}, cls=APGIJSONEncoder)) == {"x": True}

    def test_dumps_json_is_valid(self):
        assert json.loads(dumps_json({"x": np.float64(2.5)})) == {"x": 2.5}

    def test_dump_json_writes_file(self, tmp_path):
        target = tmp_path / "nested" / "out.json"
        dump_json({"significant": np.True_}, target)
        assert json.loads(target.read_text(encoding="utf-8")) == {"significant": True}


class TestProtocolResultContract:
    """Every protocol result must be plain-JSON serialisable."""

    def test_model_dump_is_serialisable_with_numpy_metadata(self):
        from utils.protocol_schema import PredictionResult, ProtocolResult

        result = ProtocolResult(
            protocol_id="TEST_01",
            named_predictions={"V1.1": PredictionResult(passed=True, value=1.0, threshold=0.5)},
            completion_percentage=100,
            status="success",
            metadata={"results": {"comparison": {"significant": np.True_, "p": np.float64(0.004)}}},
        )

        payload = json.dumps(result.model_dump())
        assert json.loads(payload)["metadata"]["results"]["comparison"]["significant"] is True

    def test_to_dict_matches_model_dump(self):
        from utils.protocol_schema import ProtocolResult

        result = ProtocolResult(
            protocol_id="TEST_02",
            named_predictions={},
            completion_percentage=0,
            status="error",
            metadata={"flag": np.False_},
        )
        assert result.to_dict() == result.model_dump()
        assert result.to_dict()["metadata"]["flag"] is False
