"""Quarantine and every-record reconciliation for incoming data (APGI-DAT-002).

Each incoming record receives exactly one named outcome:

``accepted``   valid and admitted
``malformed``  failed structural/schema validation
``duplicate``  record_id (or content digest) already seen
``delayed``    arrived after the declared acquisition window closed
``unmatched``  references a session/participant/parent that is not known

Non-accepted records are segregated into the quarantine with the reason, a
notification hook is called, and :meth:`QuarantineLedger.reconcile` proves
``received == accepted + Σ quarantined`` so no record silently disappears.
Loader exceptions are captured as ``malformed`` outcomes instead of being
suppressed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set


class Outcome(str, Enum):
    ACCEPTED = "accepted"
    MALFORMED = "malformed"
    DUPLICATE = "duplicate"
    DELAYED = "delayed"
    UNMATCHED = "unmatched"


class ReconciliationError(RuntimeError):
    """Counts do not reconcile: at least one record is unaccounted for."""


@dataclass
class Disposition:
    index: int
    record_id: Optional[str]
    outcome: Outcome
    reason: str = ""


@dataclass
class QuarantineLedger:
    required_fields: Iterable[str] = ("record_id",)
    known_keys: Optional[Set[str]] = None
    match_field: Optional[str] = None
    window_end: Optional[datetime] = None
    timestamp_field: str = "received_at"
    validator: Optional[Callable[[Mapping[str, Any]], None]] = None
    notify: Callable[[Disposition], None] = lambda d: None
    accepted: List[Mapping[str, Any]] = field(default_factory=list)
    quarantine: List[Dict[str, Any]] = field(default_factory=list)
    dispositions: List[Disposition] = field(default_factory=list)
    received: int = 0
    _seen_ids: Set[str] = field(default_factory=set)
    _seen_digests: Set[str] = field(default_factory=set)

    def _classify(self, record: Any) -> tuple:
        if not isinstance(record, Mapping):
            return Outcome.MALFORMED, "record is not an object"
        missing = [f for f in self.required_fields if record.get(f) in (None, "")]
        if missing:
            return Outcome.MALFORMED, f"missing required fields {missing}"
        if self.validator is not None:
            try:
                self.validator(record)
            except Exception as exc:  # captured, never suppressed
                return Outcome.MALFORMED, f"validation error: {exc}"
        digest = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
        rid = str(record["record_id"])
        if rid in self._seen_ids or digest in self._seen_digests:
            return Outcome.DUPLICATE, f"record_id {rid} or identical content already received"
        if self.window_end is not None and record.get(self.timestamp_field) is not None:
            ts = record[self.timestamp_field]
            try:
                ts = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
                late = ts > self.window_end
            except (TypeError, ValueError) as exc:
                # Unparseable or naive/aware-mismatched timestamps are a named
                # outcome, never an exception that leaves the record unaccounted.
                return Outcome.MALFORMED, f"invalid {self.timestamp_field}: {exc}"
            if late:
                return Outcome.DELAYED, f"{self.timestamp_field} {ts.isoformat()} after window end"
        if self.match_field and self.known_keys is not None and record.get(self.match_field) not in self.known_keys:
            return Outcome.UNMATCHED, f"{self.match_field}={record.get(self.match_field)!r} not known"
        self._seen_ids.add(rid)
        self._seen_digests.add(digest)
        return Outcome.ACCEPTED, ""

    def ingest(self, records: Iterable[Any]) -> List[Disposition]:
        out: List[Disposition] = []
        for record in records:
            idx = self.received
            self.received += 1
            outcome, reason = self._classify(record)
            rid = str(record.get("record_id")) if isinstance(record, Mapping) and record.get("record_id") else None
            disp = Disposition(idx, rid, outcome, reason)
            self.dispositions.append(disp)
            if outcome is Outcome.ACCEPTED:
                self.accepted.append(record)
            else:
                self.quarantine.append({"index": idx, "outcome": outcome.value, "reason": reason, "record": record})
                self.notify(disp)
            out.append(disp)
        return out

    def counts(self) -> Dict[str, int]:
        c = Counter(d.outcome.value for d in self.dispositions)
        return {o.value: c.get(o.value, 0) for o in Outcome}

    def reconcile(self) -> Dict[str, int]:
        counts = self.counts()
        total = sum(counts.values())
        if total != self.received or len(self.accepted) + len(self.quarantine) != self.received:
            raise ReconciliationError(f"received={self.received} but accounted={total}")
        return {"received": self.received, **counts}
