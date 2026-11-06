"""Per-data-class retention enforcement (APGI-DAT-003).

Reads ``config/data_retention.yaml`` and reports or deletes files that have
exceeded their class's retention period. Safety rules:

* a class whose ``retention_days`` is ``PENDING`` is never touched;
* ``lineage_withdrawal`` classes are never deleted by age — they are removed
  only through ``utils.consent_gate.WithdrawalOrchestrator`` so derivatives,
  caches and stores are handled together;
* deletion only happens with ``dry_run=False`` and every action is returned
  as an auditable record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "config" / "data_retention.yaml"


@dataclass
class RetentionAction:
    data_class: str
    path: str
    age_days: float
    action: str  # "would_delete" | "deleted" | "skipped_pending" | "skipped_lineage"
    reason: str


def load_policy(path: Optional[Path] = None) -> Dict[str, Any]:
    return yaml.safe_load(Path(path or POLICY_PATH).read_text(encoding="utf-8"))


def enforce(
    policy: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
    dry_run: bool = True,
    now: Optional[float] = None,
) -> List[RetentionAction]:
    policy = policy or load_policy()
    root = Path(root or ROOT)
    now = now if now is not None else time.time()
    actions: List[RetentionAction] = []
    for name, cls in policy.get("classes", {}).items():
        days = cls.get("retention_days")
        for rel in cls.get("paths", []):
            base = root / rel
            if not base.exists():
                continue
            files = [p for p in base.rglob("*") if p.is_file()]
            if days == "PENDING" or days is None:
                actions.extend(
                    RetentionAction(name, str(p), 0.0, "skipped_pending", "retention period not approved")
                    for p in files
                )
                continue
            if cls.get("deletion") == "lineage_withdrawal":
                actions.extend(
                    RetentionAction(name, str(p), 0.0, "skipped_lineage", "deleted only via withdrawal orchestration")
                    for p in files
                )
                continue
            for p in files:
                age = (now - p.stat().st_mtime) / 86400.0
                if age > float(days):
                    if not dry_run:
                        p.unlink()
                    actions.append(
                        RetentionAction(
                            name,
                            str(p),
                            round(age, 2),
                            "would_delete" if dry_run else "deleted",
                            f"older than {days} d",
                        )
                    )
    return actions
