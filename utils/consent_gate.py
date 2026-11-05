"""Participant consent enforcement and lineage-wide withdrawal (APGI-CON-001, APGI-CON-002).

Scope: Section 5.2 of the governing specification marks CON-001 as "–" for
A05 while Section 4.9 covers any component that processes research data.
This module implements the gate so the scope decision (recorded in
``docs/validation/application-validation-profile.md``) can be enforced
either way; it does not itself decide applicability.

:class:`ConsentGate` must be called before ingest, processing and export of
any record that names participants. An absent, withdrawn, expired or
out-of-scope consent blocks the operation *before* anything is persisted and
the refusal is logged.

:class:`WithdrawalOrchestrator` deletes a participant's records and every
derived descendant across all registered stores, verifies absence in each
store and returns a deletion record. Backups are handled by an explicit
policy hook; the backup policy itself awaits approval (AUD-SPEC-07).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Set

from utils.data_provenance import ProvenanceRecord, ProvenanceRegistry

logger = logging.getLogger(__name__)


class Operation(str, Enum):
    INGEST = "ingest"
    PROCESS = "process"
    EXPORT = "export"


class ConsentStatus(str, Enum):
    ACTIVE = "active"
    WITHDRAWN = "withdrawn"


class ConsentError(PermissionError):
    """An operation was refused because consent does not cover it."""


@dataclass
class ConsentRecord:
    participant_id: str
    status: ConsentStatus
    scopes: Set[Operation]
    purposes: Set[str]
    expires_at: Optional[datetime] = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConsentRegistry:
    def __init__(self) -> None:
        self._records: Dict[str, ConsentRecord] = {}

    def record(self, consent: ConsentRecord) -> None:
        self._records[consent.participant_id] = consent

    def withdraw(self, participant_id: str) -> None:
        if participant_id in self._records:
            self._records[participant_id].status = ConsentStatus.WITHDRAWN

    def get(self, participant_id: str) -> Optional[ConsentRecord]:
        return self._records.get(participant_id)


def _pseudonym(participant_id: str) -> str:
    """Log-safe participant reference (never log raw identifiers)."""
    return hashlib.sha256(participant_id.encode("utf-8")).hexdigest()[:12]


class ConsentGate:
    def __init__(self, registry: ConsentRegistry) -> None:
        self.registry = registry
        self.audit: List[Dict[str, str]] = []

    def _log(self, outcome: str, operation: Operation, participant_id: str, purpose: str, reason: str = "") -> None:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "outcome": outcome,
            "operation": operation.value,
            "participant": _pseudonym(participant_id),
            "purpose": purpose,
            "reason": reason,
        }
        self.audit.append(entry)
        logger.log(logging.INFO if outcome == "permitted" else logging.WARNING, "consent_gate %s", entry)

    def check(self, operation: Operation, participant_ids: Iterable[str], purpose: str) -> None:
        """Raise :class:`ConsentError` unless every participant consents to ``operation`` for ``purpose``."""
        ids = list(participant_ids)
        now = datetime.now(timezone.utc)
        for pid in ids:
            consent = self.registry.get(pid)
            reason = ""
            if consent is None:
                reason = "no consent record"
            elif consent.status is ConsentStatus.WITHDRAWN:
                reason = "consent withdrawn"
            elif consent.expires_at is not None and consent.expires_at <= now:
                reason = "consent expired"
            elif operation not in consent.scopes:
                reason = f"operation {operation.value} outside consented scope"
            elif purpose not in consent.purposes:
                reason = f"purpose {purpose!r} outside consented purposes"
            if reason:
                self._log("blocked", operation, pid, purpose, reason)
                raise ConsentError(f"{operation.value} blocked for participant {_pseudonym(pid)}: {reason}")
        for pid in ids:
            self._log("permitted", operation, pid, purpose)

    def guarded(
        self, operation: Operation, record: ProvenanceRecord, purpose: str, action: Callable[[], object]
    ) -> object:
        """Run ``action`` only after the consent check passes (nothing persists otherwise)."""
        self.check(operation, record.participant_ids, purpose)
        return action()


class Store(Protocol):
    name: str

    def delete(self, record_id: str) -> None: ...

    def contains(self, record_id: str) -> bool: ...


class InMemoryStore:
    """Minimal store used by tests and as a reference adapter."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.items: Dict[str, object] = {}

    def put(self, record_id: str, value: object) -> None:
        self.items[record_id] = value

    def delete(self, record_id: str) -> None:
        self.items.pop(record_id, None)

    def contains(self, record_id: str) -> bool:
        return record_id in self.items


@dataclass
class DeletionRecord:
    participant: str
    requested_at: str
    completed_at: str
    deleted_record_ids: List[str]
    stores_verified: List[str]
    backup_disposition: str


class WithdrawalIncompleteError(RuntimeError):
    """A record survived withdrawal in at least one store."""


class WithdrawalOrchestrator:
    def __init__(
        self,
        registry: ProvenanceRegistry,
        stores: Iterable[Store],
        consents: ConsentRegistry,
        backup_policy: Callable[[List[str]], str] = lambda ids: "PENDING_APPROVED_BACKUP_POLICY",
    ) -> None:
        self.registry = registry
        self.stores = list(stores)
        self.consents = consents
        self.backup_policy = backup_policy

    def affected_records(self, participant_id: str) -> List[str]:
        direct = [r.record_id for r in self.registry.records() if participant_id in r.participant_ids]
        affected: List[str] = []
        for rid in direct:
            for x in [rid, *self.registry.descendants(rid)]:
                if x not in affected:
                    affected.append(x)
        return affected

    def withdraw(self, participant_id: str) -> DeletionRecord:
        requested = datetime.now(timezone.utc).isoformat()
        self.consents.withdraw(participant_id)
        ids = self.affected_records(participant_id)
        for store in self.stores:
            for rid in ids:
                store.delete(rid)
        survivors = [(s.name, rid) for s in self.stores for rid in ids if s.contains(rid)]
        if survivors:
            raise WithdrawalIncompleteError(f"records survived withdrawal: {survivors}")
        self.registry.remove(ids)
        disposition = self.backup_policy(ids)
        return DeletionRecord(
            participant=_pseudonym(participant_id),
            requested_at=requested,
            completed_at=datetime.now(timezone.utc).isoformat(),
            deleted_record_ids=ids,
            stores_verified=[s.name for s in self.stores],
            backup_disposition=disposition,
        )
