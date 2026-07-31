"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1

Structured logging for the APGI framework.

Provides:
    - JSON-formatted records with *operation-scoped* correlation IDs
    - Optional rotating file output driven by APGI_LOG_FILE
    - A thread-safe in-process metrics registry
    - Real log export (json / csv / txt)

Correlation IDs are stored in a :mod:`contextvars` variable so every record
emitted inside one logical operation shares an ID. A fresh ID per record would
make correlation impossible, which is the defect this module previously had.
"""

import csv
import json
import logging
import logging.handlers
import os
import re
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# ---------------------------------------------------------------------------
# Correlation context
# ---------------------------------------------------------------------------

_correlation_id: ContextVar[Optional[str]] = ContextVar("apgi_correlation_id", default=None)

#: Correlation ID reported when no operation scope is active. Constant (rather
#: than random) so records emitted outside a scope remain groupable.
UNSCOPED_CORRELATION_ID = "unscoped"


def get_correlation_id() -> str:
    """Return the correlation ID bound to the current context."""
    return _correlation_id.get() or UNSCOPED_CORRELATION_ID


def set_correlation_id(correlation_id: str) -> None:
    """Bind ``correlation_id`` to the current context."""
    _correlation_id.set(correlation_id)


@contextmanager
def correlation_scope(correlation_id: Optional[str] = None) -> Iterator[str]:
    """Bind one correlation ID for the duration of a logical operation.

    Every record emitted inside the block — including from nested calls and
    from threads that inherit this context — shares the returned ID.

    Args:
        correlation_id: Explicit ID; generated when omitted.

    Yields:
        The active correlation ID.
    """
    cid = correlation_id or str(uuid.uuid4())
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Emit each record as a single JSON object."""

    #: Attributes present on every LogRecord; anything else is treated as
    #: caller-supplied context and included in the payload.
    _RESERVED = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            # Explicit per-record override wins; otherwise use the active scope.
            "correlation_id": getattr(record, "correlation_id", None) or get_correlation_id(),
        }

        extras = {k: v for k, v in record.__dict__.items() if k not in self._RESERVED and k != "correlation_id"}
        if extras:
            log_record["context"] = {k: _jsonable(v) for k, v in extras.items()}

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, default=str)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary context values to JSON-safe types."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Handler configuration
# ---------------------------------------------------------------------------

_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
DEFAULT_LOG_MAX_BYTES = 10 * 1024**2
DEFAULT_LOG_BACKUP_COUNT = 5


def parse_size(value: str, default: int = DEFAULT_LOG_MAX_BYTES) -> int:
    """Parse a human-readable size such as ``10MB`` into bytes.

    Falls back to ``default`` for unparseable input rather than raising, so a
    typo in the environment cannot prevent the process from starting.
    """
    if not value:
        return default
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMG]?B)?\s*", value.strip(), re.IGNORECASE)
    if not match:
        return default
    amount = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    return int(amount * _SIZE_UNITS.get(unit, 1))


logger = logging.getLogger("apgi")
logger.propagate = False

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(JsonFormatter())
logger.addHandler(_stream_handler)
logger.setLevel(os.environ.get("APGI_LOG_LEVEL", "INFO").upper())

_file_handler: Optional[logging.handlers.RotatingFileHandler] = None


def configure_file_logging(
    log_file: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> Optional[logging.handlers.RotatingFileHandler]:
    """Attach a rotating file handler to the APGI logger.

    Reads ``APGI_LOG_FILE``, ``APGI_LOG_MAX_SIZE`` and ``APGI_LOG_BACKUP_COUNT``
    when arguments are omitted. Returns ``None`` when no log file is configured
    or the destination is not writable — logging must never be fatal.
    """
    global _file_handler

    target = log_file or os.environ.get("APGI_LOG_FILE")
    if not target:
        return None

    if max_bytes is None:
        max_bytes = parse_size(os.environ.get("APGI_LOG_MAX_SIZE", ""), DEFAULT_LOG_MAX_BYTES)
    if backup_count is None:
        try:
            backup_count = int(os.environ.get("APGI_LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT))
        except ValueError:
            backup_count = DEFAULT_LOG_BACKUP_COUNT

    try:
        path = Path(target).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
    except OSError as exc:
        logger.warning(f"File logging disabled; cannot open {target}: {exc}")
        return None

    if _file_handler is not None:
        logger.removeHandler(_file_handler)
        _file_handler.close()

    handler.setLevel(logger.level)
    logger.addHandler(handler)
    _file_handler = handler
    return handler


def get_log_file_path() -> Optional[Path]:
    """Return the active log file path, if file logging is configured."""
    if _file_handler is not None:
        return Path(_file_handler.baseFilename)
    target = os.environ.get("APGI_LOG_FILE")
    return Path(target).expanduser() if target else None


def set_log_level(level: str) -> str:
    """Set the level on the APGI logger and all of its handlers.

    Args:
        level: Level name, case-insensitive (e.g. ``"debug"``).

    Returns:
        The normalised level name that was applied.

    Raises:
        ValueError: If ``level`` is not a recognised logging level.
    """
    normalised = str(level).strip().upper()
    numeric = logging.getLevelName(normalised)
    if not isinstance(numeric, int):
        valid = ", ".join(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        raise ValueError(f"Invalid log level '{level}'. Expected one of: {valid}")

    logger.setLevel(numeric)
    for handler in logger.handlers:
        handler.setLevel(numeric)
    return normalised


# Honour APGI_LOG_FILE at import time so operators get file logs without an
# explicit call. Safe: returns None when unset or unwritable.
configure_file_logging()


# ---------------------------------------------------------------------------
# Metrics registry
# ---------------------------------------------------------------------------


class _MetricsRegistry:
    """Thread-safe accumulator for performance metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: Dict[str, Dict[str, Any]] = {}

    def record(self, name: str, value: float, unit: str = "") -> None:
        numeric = float(value)
        with self._lock:
            entry = self._metrics.get(name)
            if entry is None:
                self._metrics[name] = {
                    "count": 1,
                    "total": numeric,
                    "min": numeric,
                    "max": numeric,
                    "unit": unit,
                }
                return
            entry["count"] += 1
            entry["total"] += numeric
            entry["min"] = min(entry["min"], numeric)
            entry["max"] = max(entry["max"], numeric)
            if unit and not entry["unit"]:
                entry["unit"] = unit

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Return ``{name: {count, mean, min, max, unit}}``."""
        with self._lock:
            return {
                name: {
                    "count": entry["count"],
                    "mean": entry["total"] / entry["count"],
                    "min": entry["min"],
                    "max": entry["max"],
                    "unit": entry["unit"],
                }
                for name, entry in self._metrics.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()


_metrics = _MetricsRegistry()


# ---------------------------------------------------------------------------
# Log export
# ---------------------------------------------------------------------------

_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def _read_log_records(source: Path, min_level: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse JSON log lines from ``source``, filtering by minimum level."""
    threshold = _LEVEL_ORDER.get((min_level or "").upper(), 0)
    records: List[Dict[str, Any]] = []

    with open(source, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate non-JSON lines written by third-party handlers.
                record = {"level": "INFO", "message": line, "timestamp": "", "correlation_id": ""}
            if _LEVEL_ORDER.get(str(record.get("level", "")).upper(), 0) >= threshold:
                records.append(record)

    return records


# ---------------------------------------------------------------------------
# Public logger wrapper
# ---------------------------------------------------------------------------


class APGILogger:
    """Wrapper exposing structured helpers on top of a stdlib logger."""

    def __init__(self, logger_instance: logging.Logger):
        self.logger = logger_instance

    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to the underlying logger."""
        return getattr(self.logger, name)

    def log_simulation_start(self, name: str, params: Any) -> None:
        """Log the start of a simulation."""
        self.logger.info(f"Simulation started: {name}", extra={"params": params})

    def log_simulation_end(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Log the end of a simulation."""
        self.logger.info(f"Simulation ended: {name}")

    def log_performance_metric(self, name: str, value: float, unit: str = "") -> None:
        """Record a performance metric and log it."""
        _metrics.record(name, value, unit)
        self.logger.info(
            f"Metric: {name} = {value}{unit}",
            extra={"metric": name, "value": value, "unit": unit},
        )

    def log_simulation_error(self, name: str, error: Exception) -> None:
        """Log a simulation error."""
        self.logger.error(f"Simulation error in {name}: {str(error)}")

    def export_logs(self, export_path: str, format_type: str = "json", log_level: Optional[str] = None) -> bool:
        """Export captured log records to a file.

        Reads the configured rotating log file (``APGI_LOG_FILE``), optionally
        filtering to records at or above ``log_level``, and writes them in the
        requested format.

        Args:
            export_path: Destination file path.
            format_type: One of ``json``, ``csv``, ``txt``.
            log_level: Optional minimum level filter (e.g. ``"WARNING"``).

        Returns:
            True on success. False if no log source exists, the format is
            unsupported, or the write fails — callers surface this to the user.
        """
        source = get_log_file_path()
        if source is None or not source.exists():
            self.logger.error(
                "Cannot export logs: no log file configured or file missing. Set APGI_LOG_FILE to enable."
            )
            return False

        fmt = format_type.strip().lower()
        if fmt not in {"json", "csv", "txt"}:
            self.logger.error(f"Cannot export logs: unsupported format '{format_type}' (expected json, csv or txt)")
            return False

        try:
            records = _read_log_records(source, log_level)
            destination = Path(export_path)
            destination.parent.mkdir(parents=True, exist_ok=True)

            if fmt == "json":
                with open(destination, "w", encoding="utf-8") as handle:
                    json.dump(records, handle, indent=2, default=str)
            elif fmt == "csv":
                columns = ["timestamp", "level", "logger", "correlation_id", "message"]
                with open(destination, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                    writer.writeheader()
                    for record in records:
                        writer.writerow({column: record.get(column, "") for column in columns})
            else:
                with open(destination, "w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(
                            f"{record.get('timestamp', '')} "
                            f"[{record.get('level', '')}] "
                            f"({record.get('correlation_id', '')}) "
                            f"{record.get('message', '')}\n"
                        )
        except OSError as exc:
            self.logger.error(f"Log export to {export_path} failed: {exc}")
            return False

        self.logger.info(f"Exported {len(records)} log records to {export_path}")
        return True

    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return aggregated performance metrics.

        Returns:
            ``{metric_name: {count, mean, min, max, unit}}``. Empty when no
            metrics have been recorded in this process.
        """
        return _metrics.summary()

    def reset_performance_metrics(self) -> None:
        """Clear all recorded metrics. Intended for tests and long-lived runs."""
        _metrics.reset()

    def log_error_with_context(self, error: Exception, context: dict) -> None:
        """Log an error with context information."""
        self.logger.error(f"Error: {str(error)}", extra={"error_context": context})


# Export aliases for compatibility
apgi_logger = APGILogger(logger)


def log_error(message: str) -> None:
    """Log an error message."""
    logger.error(message)


def log_error_with_context(error: Exception, context: dict) -> None:
    """Log an error with context information."""
    logger.error(f"Error: {str(error)}", extra={"error_context": context})
