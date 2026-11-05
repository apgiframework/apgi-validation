#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from utils.logging_config import apgi_logger
except ImportError:
    apgi_logger = None


class DataCollector:
    """Collects and stores historical APGI framework data."""

    def __init__(
        self,
        db_path: str = "data_repository/historical_data.db",
        collection_interval: int = 60,
    ):
        """
        Initialize the data collector.

        Args:
            db_path: Path to SQLite database
            collection_interval: Collection interval in seconds
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.collection_interval = collection_interval

        # Threading controls
        self._collection_active = False
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Initialize database
        self._init_database()

        if apgi_logger:
            apgi_logger.logger.info(f"Data collector initialized with {collection_interval}s interval")

    def _init_database(self):
        """Initialize database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # System metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        cpu_percent REAL,
                        memory_percent REAL,
                        memory_used_gb REAL,
                        disk_usage_percent REAL,
                        network_connections INTEGER,
                        load_average REAL
                    )
                """)

                # Validation results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS validation_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        protocol_number INTEGER,
                        protocol_name TEXT,
                        status TEXT,
                        execution_time REAL,
                        tests_passed INTEGER,
                        tests_failed INTEGER,
                        success_rate REAL,
                        error_message TEXT
                    )
                """)

                # Performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metric_category TEXT,
                        metric_name TEXT,
                        metric_value REAL,
                        unit TEXT,
                        metadata TEXT
                    )
                """)

                conn.commit()

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Failed to initialize database: {e}")
            raise

    def start_collection(self):
        """Start background data collection."""
        if self._collection_active:
            if apgi_logger:
                apgi_logger.logger.warning("Data collection already active")
            return

        self._collection_active = True
        self._stop_event.clear()
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()

        if apgi_logger:
            apgi_logger.logger.info("Started data collection")

    def stop_collection(self):
        """Stop background data collection."""
        if not self._collection_active:
            return

        self._collection_active = False
        self._stop_event.set()

        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5)

        if apgi_logger:
            apgi_logger.logger.info("Stopped data collection")

    def _collection_loop(self):
        """Main collection loop running in background thread."""
        while self._collection_active and not self._stop_event.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Collect performance metrics from logs
                self._collect_performance_metrics()

                # Process validation results
                self._process_validation_results()

                # Wait for next collection
                self._stop_event.wait(self.collection_interval)

            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Error in collection loop: {e}")
                # Continue collecting even if there's an error

    def _collect_system_metrics(self):
        """Collect current system metrics."""
        if not PSUTIL_AVAILABLE:
            return

        try:
            metrics: Dict[str, Union[str, float, int]] = {
                "timestamp": datetime.now().isoformat(),
            }

            # Collect CPU metrics with fallback
            try:
                metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
            except Exception:
                metrics["cpu_percent"] = 0.0

            # Collect memory metrics with fallback
            try:
                memory = psutil.virtual_memory()
                metrics["memory_percent"] = memory.percent
                metrics["memory_used_gb"] = memory.used / (1024**3)
            except Exception:
                metrics["memory_percent"] = 0.0
                metrics["memory_used_gb"] = 0.0

            # Collect disk usage with fallback
            try:
                metrics["disk_usage_percent"] = psutil.disk_usage("/").percent
            except Exception:
                metrics["disk_usage_percent"] = 0.0

            # Collect network connections with fallback
            try:
                metrics["network_connections"] = len(psutil.net_connections())
            except Exception:
                metrics["network_connections"] = 0

            # Add load average for Unix systems
            try:
                import os

                load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
                if load_avg is not None:
                    metrics["load_average"] = load_avg
            except (AttributeError, OSError):
                metrics["load_average"] = None

            # Store in database
            self._store_system_metrics(metrics)

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error collecting system metrics: {e}")
            # Don't re-raise, just continue with default values
            try:
                fallback_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "memory_used_gb": 0.0,
                    "disk_usage_percent": 0.0,
                    "network_connections": 0,
                    "load_average": None,
                }
                self._store_system_metrics(fallback_metrics)
            except (ImportError, RuntimeError, OSError):
                # Log the error but don't crash the monitoring system
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to collect system metrics: {type(e).__name__}")
                self._store_system_metrics(fallback_metrics)

    def _store_system_metrics(self, metrics: Dict[str, Any]):
        """Store system metrics in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, memory_used_gb, 
                     disk_usage_percent, network_connections, load_average)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metrics["timestamp"],
                        metrics["cpu_percent"],
                        metrics["memory_percent"],
                        metrics["memory_used_gb"],
                        metrics["disk_usage_percent"],
                        metrics["network_connections"],
                        metrics.get("load_average"),
                    ),
                )
                conn.commit()

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error storing system metrics: {e}")

    def _collect_performance_metrics(self):
        """Collect performance metrics from log files."""
        try:
            # Look for recent performance log files
            log_dir = Path("logs")
            if not log_dir.exists():
                return

            # Find recent log files (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)

            for log_file in log_dir.glob("*.log"):
                try:
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        continue

                    # Parse log file for performance metrics
                    self._parse_log_file(log_file)

                except Exception as e:
                    if apgi_logger:
                        apgi_logger.logger.warning(f"Error parsing log file {log_file}: {e}")

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error collecting performance metrics: {e}")

    def _parse_log_file(self, log_file: Path):
        """Parse a log file for performance metrics."""
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    # Look for performance-related log entries
                    if any(keyword in line.lower() for keyword in ["performance", "execution", "timing"]):
                        self._extract_performance_metric(line)

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.warning(f"Error parsing log file content: {e}")

    def _extract_performance_metric(self, log_line: str):
        """Extract performance metric from log line."""
        try:
            # Simple pattern matching for performance metrics
            # This is a basic implementation - could be enhanced with regex
            timestamp = datetime.now().isoformat()

            if "execution time" in log_line.lower():
                # Extract execution time
                parts = log_line.split()
                for i, part in enumerate(parts):
                    if "time" in part.lower() and i + 1 < len(parts):
                        try:
                            time_value = float(parts[i + 1].replace("s", ""))
                            self._store_performance_metric(
                                timestamp,
                                "execution",
                                "execution_time",
                                time_value,
                                "seconds",
                            )
                        except ValueError:
                            pass

            elif "memory" in log_line.lower():
                # Extract memory usage
                parts = log_line.split()
                for i, part in enumerate(parts):
                    if "memory" in part.lower() and i + 1 < len(parts):
                        try:
                            mem_value = float(parts[i + 1].replace("MB", "").replace("GB", ""))
                            if "GB" in parts[i + 1]:
                                mem_value *= 1024  # Convert to MB
                            self._store_performance_metric(timestamp, "memory", "usage", mem_value, "MB")
                        except ValueError:
                            pass

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.debug(f"Error extracting performance metric: {e}")

    def _store_performance_metric(self, timestamp: str, category: str, name: str, value: float, unit: str):
        """Store performance metric in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO performance_metrics 
                    (timestamp, metric_category, metric_name, metric_value, unit)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (timestamp, category, name, value, unit),
                )
                conn.commit()

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error storing performance metric: {e}")

    def _process_validation_results(self):
        """Process validation results from output files."""
        try:
            # Look for validation result files
            results_dir = Path("data_repository/processed")
            if not results_dir.exists():
                return

            # Find recent result files
            for result_file in results_dir.glob("*.json"):
                try:
                    # Check if this file has already been processed
                    if self._is_result_processed(result_file):
                        continue

                    # Parse validation result
                    self._parse_validation_result(result_file)

                except Exception as e:
                    if apgi_logger:
                        apgi_logger.logger.warning(f"Error processing result file {result_file}: {e}")

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error processing validation results: {e}")

    def _is_result_processed(self, result_file: Path) -> bool:
        """Check if a result file has already been processed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM validation_results 
                    WHERE metadata LIKE ?
                """,
                    (f"%{result_file.name}%",),
                )
                return cursor.fetchone()[0] > 0

        except Exception:
            return False  # Assume not processed if there's an error

    def _parse_validation_result(self, result_file: Path):
        """Parse a validation result JSON file."""
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract validation information
            timestamp = datetime.now().isoformat()

            # Try to determine protocol from filename or content
            protocol_number = self._extract_protocol_number(result_file, data)
            protocol_name = f"Protocol {protocol_number}"

            # Extract results
            status = data.get("status", "unknown")
            execution_time = data.get("execution_time", 0.0)
            tests_passed = data.get("tests_passed", 0)
            tests_failed = data.get("tests_failed", 0)
            success_rate = (
                (tests_passed / (tests_passed + tests_failed) * 100) if (tests_passed + tests_failed) > 0 else 0
            )
            error_message = data.get("error_message", "")

            # Store in database
            self._store_validation_result(
                timestamp,
                protocol_number,
                protocol_name,
                status,
                execution_time,
                tests_passed,
                tests_failed,
                success_rate,
                error_message,
            )

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error parsing validation result: {e}")

    def _extract_protocol_number(self, result_file: Path, data: Dict) -> int:
        """Extract protocol number from filename or data."""
        # Try to extract from filename
        filename = result_file.name.lower()
        for i in range(1, 13):
            if f"protocol_{i}" in filename or f"fp-{i}" in filename:
                return i

        # Try to extract from data
        if "protocol_number" in data:
            return int(data["protocol_number"])
        elif "protocol" in data:
            protocol_str = str(data["protocol"])
            for i in range(1, 13):
                if str(i) in protocol_str:
                    return i

        # Default to 1 if can't determine
        return 1

    def _store_validation_result(
        self,
        timestamp: str,
        protocol_number: int,
        protocol_name: str,
        status: str,
        execution_time: float,
        tests_passed: int,
        tests_failed: int,
        success_rate: float,
        error_message: str,
    ):
        """Store validation result in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO validation_results 
                    (timestamp, protocol_number, protocol_name, status, execution_time,
                     tests_passed, tests_failed, success_rate, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        protocol_number,
                        protocol_name,
                        status,
                        execution_time,
                        tests_passed,
                        tests_failed,
                        success_rate,
                        error_message,
                    ),
                )
                conn.commit()

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error storing validation result: {e}")

    def get_recent_data(self, table: str, hours: int = 24) -> List[Dict]:
        """Get recent data from specified table."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Validate table name to prevent SQL injection
                allowed_tables = ["validation_results", "benchmark_results", "simulation_data"]
                if table not in allowed_tables:
                    raise ValueError(f"Invalid table name: {table}. Allowed: {allowed_tables}")

                # Semgrep sqlalchemy-execute-raw-query reviewed: table name validated against allowed_tables; values are bound parameters.
                cursor.execute(  # nosemgrep: sqlalchemy-execute-raw-query
                    f"SELECT * FROM {table} WHERE timestamp >= ? ORDER BY timestamp DESC",  # nosec B608
                    (cutoff_time.isoformat(),),
                )

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error getting recent data from {table}: {e}")
            return []

    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to prevent database from growing too large."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Clean up old data from each table
                tables = ["system_metrics", "validation_results", "performance_metrics"]
                for table in tables:
                    # Validate table name to prevent SQL injection
                    if not table.replace("_", "").isalnum():
                        raise ValueError(f"Invalid table name: {table}")

                    cursor.execute(
                        """
                        DELETE FROM {table} WHERE timestamp < ?
                        """,
                        (cutoff_time.isoformat(),),
                    )

                conn.commit()

                if apgi_logger:
                    apgi_logger.logger.info(f"Cleaned up data older than {days_to_keep} days")

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error cleaning up old data: {e}")


# Global data collector instance
_data_collector: Optional[DataCollector] = None


def get_data_collector() -> DataCollector:
    """Get or create the global data collector instance."""
    global _data_collector
    if _data_collector is None:
        _data_collector = DataCollector()
    return _data_collector


def start_data_collection():
    """Start the global data collection."""
    collector = get_data_collector()
    collector.start_collection()


def stop_data_collection():
    """Stop the global data collection."""
    collector = get_data_collector()
    collector.stop_collection()


if __name__ == "__main__":
    # Run data collector when executed directly
    collector = DataCollector()

    # Check if running in production mode (daemon/service mode)
    # Default to test mode for validation - set APGI_PRODUCTION_MODE=1 for daemon mode
    production_mode = os.environ.get("APGI_PRODUCTION_MODE", "0") == "1"

    try:
        print("Starting APGI data collection...")
        collector.start_collection()

        if production_mode:
            # Keep running until interrupted (production mode)
            print("Running in production mode (Ctrl+C to stop)...")
            while True:
                time.sleep(60)
        else:
            # In test mode, run for a short time then exit
            print("Running in test mode (5 seconds)...")
            time.sleep(5)
            print("Test collection complete.")

    except KeyboardInterrupt:
        print("\nStopping data collection...")
    finally:
        collector.stop_collection()
        print("Data collection stopped.")
