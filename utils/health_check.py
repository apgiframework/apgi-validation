"""Operational health checks with structured alerts (APGI-IMP-003).

Each check returns ``ok``/``warn``/``fail`` with a measured value. Any non-ok
result is logged once as an ``apgi_alert`` record (JSON) at WARNING/ERROR so
log-based alerting can route it. ``run_health_checks`` accepts injected
probes so tests can induce resource, integrity and audit faults and prove the
signals fire.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("apgi.health")
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class HealthResult:
    check: str
    status: str  # ok | warn | fail
    value: object
    detail: str = ""


def _disk(path: Path = ROOT, warn_gb: float = 5.0, fail_gb: float = 1.0) -> HealthResult:
    free_gb = shutil.disk_usage(path).free / 1e9
    status = "fail" if free_gb < fail_gb else "warn" if free_gb < warn_gb else "ok"
    return HealthResult("disk_free", status, round(free_gb, 2), f"thresholds warn<{warn_gb} fail<{fail_gb} GB")


def _protocol_lock() -> HealthResult:
    from utils.protocol_lock import ProtocolLockError, verify_protocol_lock

    try:
        lock = verify_protocol_lock()
        return HealthResult("protocol_lock", "ok", lock.release)
    except ProtocolLockError as exc:
        return HealthResult("protocol_lock", "fail", None, str(exc))


def _keys() -> HealthResult:
    try:
        from utils.secure_key_manager import get_secure_key_manager

        mgr = get_secure_key_manager()
        mgr.get_backup_hmac_key()
        mgr.get_jwt_secret_key()
        return HealthResult("key_material", "ok", True)
    except Exception as exc:  # report, never crash the health endpoint
        return HealthResult("key_material", "fail", False, type(exc).__name__)


def _audit_writable(log_dir: Optional[Path] = None) -> HealthResult:
    log_dir = Path(log_dir or os.environ.get("APGI_LOG_DIR", ROOT / "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=log_dir, delete=True) as fh:
            fh.write(b"health")
        return HealthResult("audit_log_writable", "ok", str(log_dir))
    except OSError as exc:
        return HealthResult("audit_log_writable", "fail", str(log_dir), str(exc))


DEFAULT_CHECKS: Dict[str, Callable[[], HealthResult]] = {
    "disk_free": _disk,
    "protocol_lock": _protocol_lock,
    "key_material": _keys,
    "audit_log_writable": _audit_writable,
}


def run_health_checks(checks: Optional[Dict[str, Callable[[], HealthResult]]] = None) -> Dict[str, object]:
    results: List[HealthResult] = []
    for name, probe in (checks or DEFAULT_CHECKS).items():
        try:
            res = probe()
        except Exception as exc:
            res = HealthResult(name, "fail", None, f"probe raised {type(exc).__name__}: {exc}")
        results.append(res)
        if res.status != "ok":
            alert = {"apgi_alert": True, "time": datetime.now(timezone.utc).isoformat(), **asdict(res)}
            logger.log(logging.ERROR if res.status == "fail" else logging.WARNING, json.dumps(alert, default=str))
    overall = (
        "fail"
        if any(r.status == "fail" for r in results)
        else "warn" if any(r.status == "warn" for r in results) else "ok"
    )
    return {"status": overall, "checks": [asdict(r) for r in results]}


if __name__ == "__main__":  # pragma: no cover
    report = run_health_checks()
    print(json.dumps(report, indent=2, default=str))
    raise SystemExit(0 if report["status"] != "fail" else 1)
