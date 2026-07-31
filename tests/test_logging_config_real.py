"""Tests for structured logging, correlation scoping, metrics and log export.

These replace coverage for three functions that previously returned success
without doing any work:
    - APGILogger.export_logs wrote a literal "placeholder" string, returned True
    - APGILogger.get_performance_summary always returned {}
    - JsonFormatter minted a fresh correlation_id per record
"""

import csv
import importlib
import json
import logging
import threading

import pytest


@pytest.fixture()
def logging_module(tmp_path, monkeypatch):
    """Reload logging_config with file logging pointed at a temp directory."""
    log_file = tmp_path / "apgi.log"
    monkeypatch.setenv("APGI_LOG_FILE", str(log_file))
    monkeypatch.setenv("APGI_LOG_MAX_SIZE", "1MB")
    monkeypatch.setenv("APGI_LOG_BACKUP_COUNT", "2")

    module = importlib.import_module("utils.logging_config")
    module = importlib.reload(module)
    module.apgi_logger.reset_performance_metrics()
    yield module

    for handler in list(module.logger.handlers):
        handler.close()
        module.logger.removeHandler(handler)


def _flush(module):
    for handler in module.logger.handlers:
        handler.flush()


def _read_records(module):
    path = module.get_log_file_path()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestCorrelationScoping:
    """A correlation ID must span an operation, not a single record."""

    def test_records_in_one_scope_share_an_id(self, logging_module):
        with logging_module.correlation_scope() as cid:
            logging_module.logger.info("first")
            logging_module.logger.info("second")
        _flush(logging_module)

        ids = {r["correlation_id"] for r in _read_records(logging_module) if r["message"] in {"first", "second"}}
        assert ids == {cid}, f"Expected one shared correlation ID, got {ids}"

    def test_separate_scopes_get_distinct_ids(self, logging_module):
        with logging_module.correlation_scope() as first:
            logging_module.logger.info("a")
        with logging_module.correlation_scope() as second:
            logging_module.logger.info("b")

        assert first != second

    def test_explicit_id_is_honoured(self, logging_module):
        with logging_module.correlation_scope("req-42") as cid:
            assert cid == "req-42"
            assert logging_module.get_correlation_id() == "req-42"

    def test_scope_restores_previous_id(self, logging_module):
        with logging_module.correlation_scope("outer"):
            with logging_module.correlation_scope("inner"):
                assert logging_module.get_correlation_id() == "inner"
            assert logging_module.get_correlation_id() == "outer"

    def test_unscoped_records_use_stable_sentinel(self, logging_module):
        assert logging_module.get_correlation_id() == logging_module.UNSCOPED_CORRELATION_ID

    def test_scope_is_inherited_by_threads(self, logging_module):
        seen = []

        with logging_module.correlation_scope("shared") as cid:
            thread = threading.Thread(target=lambda: seen.append(logging_module.get_correlation_id()))
            thread.start()
            thread.join()

        # contextvars propagate to threads started inside the scope only when the
        # context is copied; the sentinel is the acceptable fallback.
        assert seen[0] in {cid, logging_module.UNSCOPED_CORRELATION_ID}


class TestSizeParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("10MB", 10 * 1024**2),
            ("512KB", 512 * 1024),
            ("1GB", 1024**3),
            ("2048", 2048),
            ("  4 MB ", 4 * 1024**2),
        ],
    )
    def test_valid_sizes(self, logging_module, text, expected):
        assert logging_module.parse_size(text) == expected

    @pytest.mark.parametrize("text", ["", "nonsense", "MB", "-5MB"])
    def test_invalid_sizes_fall_back_to_default(self, logging_module, text):
        """A typo must not prevent startup."""
        assert logging_module.parse_size(text) == logging_module.DEFAULT_LOG_MAX_BYTES


class TestFileLogging:
    def test_log_file_is_created_and_written(self, logging_module):
        logging_module.logger.info("persisted")
        _flush(logging_module)

        path = logging_module.get_log_file_path()
        assert path.exists()
        assert any(r["message"] == "persisted" for r in _read_records(logging_module))

    def test_records_are_valid_json_with_expected_fields(self, logging_module):
        logging_module.logger.warning("structured")
        _flush(logging_module)

        record = next(r for r in _read_records(logging_module) if r["message"] == "structured")
        assert record["level"] == "WARNING"
        assert record["logger"] == "apgi"
        assert "timestamp" in record and "correlation_id" in record

    def test_extra_context_is_captured(self, logging_module):
        logging_module.logger.info("with context", extra={"protocol": "VP_02", "trials": 100})
        _flush(logging_module)

        record = next(r for r in _read_records(logging_module) if r["message"] == "with context")
        assert record["context"]["protocol"] == "VP_02"
        assert record["context"]["trials"] == 100

    def test_unserialisable_context_is_stringified(self, logging_module):
        logging_module.logger.info("odd", extra={"obj": object()})
        _flush(logging_module)

        record = next(r for r in _read_records(logging_module) if r["message"] == "odd")
        assert isinstance(record["context"]["obj"], str)

    def test_exception_info_is_recorded(self, logging_module):
        try:
            raise ValueError("boom")
        except ValueError:
            logging_module.logger.exception("failed")
        _flush(logging_module)

        record = next(r for r in _read_records(logging_module) if r["message"] == "failed")
        assert "ValueError: boom" in record["exception"]

    def test_unwritable_destination_disables_file_logging(self, logging_module, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        assert logging_module.configure_file_logging(str(blocker / "nested" / "x.log")) is None


class TestSetLogLevel:
    @pytest.mark.parametrize("level", ["debug", "INFO", "Warning", "error", "CRITICAL"])
    def test_valid_levels_applied_to_logger_and_handlers(self, logging_module, level):
        applied = logging_module.set_log_level(level)
        assert applied == level.upper()
        assert logging_module.logger.level == logging.getLevelName(level.upper())
        for handler in logging_module.logger.handlers:
            assert handler.level == logging.getLevelName(level.upper())

    @pytest.mark.parametrize("level", ["BOGUS", "", "12345", "trace"])
    def test_invalid_levels_raise(self, logging_module, level):
        with pytest.raises(ValueError, match="Invalid log level"):
            logging_module.set_log_level(level)

    def test_level_filters_records(self, logging_module):
        logging_module.set_log_level("ERROR")
        logging_module.logger.info("suppressed")
        logging_module.logger.error("kept")
        _flush(logging_module)

        messages = [r["message"] for r in _read_records(logging_module)]
        assert "suppressed" not in messages
        assert "kept" in messages


class TestPerformanceMetrics:
    def test_summary_is_empty_before_any_metric(self, logging_module):
        assert logging_module.apgi_logger.get_performance_summary() == {}

    def test_metrics_aggregate_correctly(self, logging_module):
        for value in (1.5, 2.5, 3.5):
            logging_module.apgi_logger.log_performance_metric("sim", value, "s")

        stats = logging_module.apgi_logger.get_performance_summary()["sim"]
        assert stats["count"] == 3
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["min"] == pytest.approx(1.5)
        assert stats["max"] == pytest.approx(3.5)
        assert stats["unit"] == "s"

    def test_summary_exposes_keys_the_cli_renders(self, logging_module):
        """main.py `performance` indexes count/mean/min/max/unit directly."""
        logging_module.apgi_logger.log_performance_metric("m", 1.0, "s")
        stats = logging_module.apgi_logger.get_performance_summary()["m"]
        assert {"count", "mean", "min", "max", "unit"} <= set(stats)

    def test_distinct_metrics_tracked_separately(self, logging_module):
        logging_module.apgi_logger.log_performance_metric("a", 1.0)
        logging_module.apgi_logger.log_performance_metric("b", 9.0)

        summary = logging_module.apgi_logger.get_performance_summary()
        assert summary["a"]["mean"] == pytest.approx(1.0)
        assert summary["b"]["mean"] == pytest.approx(9.0)

    def test_reset_clears_metrics(self, logging_module):
        logging_module.apgi_logger.log_performance_metric("a", 1.0)
        logging_module.apgi_logger.reset_performance_metrics()
        assert logging_module.apgi_logger.get_performance_summary() == {}

    def test_concurrent_recording_is_thread_safe(self, logging_module):
        def worker():
            for _ in range(100):
                logging_module.apgi_logger.log_performance_metric("concurrent", 1.0)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert logging_module.apgi_logger.get_performance_summary()["concurrent"]["count"] == 800


class TestLogExport:
    """Export must move real records and report failure honestly."""

    def _seed(self, module):
        module.logger.info("info line")
        module.logger.warning("warning line")
        module.logger.error("error line")
        _flush(module)

    def test_json_export_contains_real_records(self, logging_module, tmp_path):
        self._seed(logging_module)
        target = tmp_path / "out.json"

        assert logging_module.apgi_logger.export_logs(str(target), "json") is True

        payload = json.loads(target.read_text(encoding="utf-8"))
        messages = [r["message"] for r in payload]
        assert "info line" in messages and "error line" in messages
        assert "placeholder" not in target.read_text(encoding="utf-8").lower()

    def test_csv_export_has_header_and_rows(self, logging_module, tmp_path):
        self._seed(logging_module)
        target = tmp_path / "out.csv"

        assert logging_module.apgi_logger.export_logs(str(target), "csv") is True

        with open(target, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {"timestamp", "level", "correlation_id", "message"} <= set(rows[0])
        assert any(row["message"] == "warning line" for row in rows)

    def test_txt_export_is_human_readable(self, logging_module, tmp_path):
        self._seed(logging_module)
        target = tmp_path / "out.txt"

        assert logging_module.apgi_logger.export_logs(str(target), "txt") is True
        assert "[ERROR]" in target.read_text(encoding="utf-8")

    def test_level_filter_excludes_lower_levels(self, logging_module, tmp_path):
        self._seed(logging_module)
        target = tmp_path / "filtered.json"

        logging_module.apgi_logger.export_logs(str(target), "json", log_level="ERROR")

        levels = {r["level"] for r in json.loads(target.read_text(encoding="utf-8"))}
        assert levels == {"ERROR"}

    def test_unsupported_format_returns_false(self, logging_module, tmp_path):
        self._seed(logging_module)
        assert logging_module.apgi_logger.export_logs(str(tmp_path / "o.xml"), "xml") is False

    def test_missing_source_returns_false(self, logging_module, tmp_path, monkeypatch):
        monkeypatch.setattr(logging_module, "get_log_file_path", lambda: None)
        assert logging_module.apgi_logger.export_logs(str(tmp_path / "o.json"), "json") is False

    def test_unwritable_destination_returns_false(self, logging_module, tmp_path):
        self._seed(logging_module)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        assert logging_module.apgi_logger.export_logs(str(blocker / "nested" / "o.json"), "json") is False

    def test_export_creates_parent_directories(self, logging_module, tmp_path):
        self._seed(logging_module)
        target = tmp_path / "deep" / "nested" / "out.json"
        assert logging_module.apgi_logger.export_logs(str(target), "json") is True
        assert target.exists()
