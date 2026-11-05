"""Record-level provenance and inferential labels (S4-VAL-07, S4-VAL-13.x, APGI-DAT-004).

Every data record that enters validation carries:

* ``data_origin`` — one of exactly five registered values (S4-VAL-13.1);
* ``processing_stage`` — ``raw`` or ``derived``; a derived record names its
  parents and its origin is *inherited* from them (S4-VAL-13.2);
* ``inferential_category`` — confirmatory | exploratory | descriptive |
  method_validation (S4-VAL-13.3);
* for ``imported_empirical`` — source, licence, version, SHA-256, acquisition
  metadata and a complete transform history (S4-VAL-13.4).

Origin inheritance rule (implementation proposal, pending owner approval
under AUD-SPEC-07): a derived record's origin is the *least empirical* origin
among its parents, using the precedence

    demonstration > synthetic > replay > imported_empirical > human_observed

so a single simulated parent taints every descendant. A derived record that
declares a different origin from the inherited one is rejected.

:class:`ResearchStoreGuard` is the ingestion boundary for the research store:
non-empirical origins, Appendix simulator output and pre-research Explorer
profiles are refused (S4-VAL-11.5, S4-VAL-13.6).
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DataOrigin(str, Enum):
    HUMAN_OBSERVED = "human_observed"
    SYNTHETIC = "synthetic"
    REPLAY = "replay"
    IMPORTED_EMPIRICAL = "imported_empirical"
    DEMONSTRATION = "demonstration"


class ProcessingStage(str, Enum):
    RAW = "raw"
    DERIVED = "derived"


class InferentialCategory(str, Enum):
    CONFIRMATORY = "confirmatory"
    EXPLORATORY = "exploratory"
    DESCRIPTIVE = "descriptive"
    METHOD_VALIDATION = "method_validation"


#: Origins that may support empirical confirmation.
EMPIRICAL_ORIGINS = frozenset({DataOrigin.HUMAN_OBSERVED, DataOrigin.IMPORTED_EMPIRICAL})

#: Lower rank = more empirical; inheritance takes the maximum rank.
_ORIGIN_RANK: Dict[DataOrigin, int] = {
    DataOrigin.HUMAN_OBSERVED: 0,
    DataOrigin.IMPORTED_EMPIRICAL: 1,
    DataOrigin.REPLAY: 2,
    DataOrigin.SYNTHETIC: 3,
    DataOrigin.DEMONSTRATION: 4,
}

#: Producing systems whose output is barred from the research store.
BARRED_SOURCE_SYSTEMS = frozenset({"appendix_simulator", "explorer_pre_research"})


class ProvenanceError(ValueError):
    """A record's provenance labels are missing, invalid or inconsistent."""


class TransformStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ProvenanceRecord(BaseModel):
    """Provenance and inferential labels required on every record."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    record_id: str = Field(min_length=1)
    data_origin: DataOrigin
    processing_stage: ProcessingStage
    inferential_category: InferentialCategory
    parents: List[str] = Field(default_factory=list)
    source_system: Optional[str] = None
    source: Optional[str] = None
    license: Optional[str] = None
    version: Optional[str] = None
    sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    acquisition_metadata: Dict[str, Any] = Field(default_factory=dict)
    transform_history: List[TransformStep] = Field(default_factory=list)
    participant_ids: List[str] = Field(default_factory=list)

    @field_validator("parents")
    @classmethod
    def _unique_parents(cls, value: List[str]) -> List[str]:
        if len(set(value)) != len(value):
            raise ValueError("parents must be unique")
        return value

    @model_validator(mode="after")
    def _stage_and_import_rules(self) -> "ProvenanceRecord":
        if self.processing_stage is ProcessingStage.RAW and self.parents:
            raise ValueError("raw records cannot declare parents")
        if self.processing_stage is ProcessingStage.DERIVED:
            if not self.parents:
                raise ValueError("derived records must declare parent lineage")
            if not self.transform_history:
                raise ValueError("derived records must carry transform_history")
        if self.data_origin is DataOrigin.IMPORTED_EMPIRICAL:
            missing = [name for name in ("source", "license", "version", "sha256") if not getattr(self, name)]
            if not self.acquisition_metadata:
                missing.append("acquisition_metadata")
            if self.processing_stage is ProcessingStage.DERIVED and not self.transform_history:
                missing.append("transform_history")
            if missing:
                raise ValueError(f"imported_empirical record requires: {', '.join(missing)}")
        return self

    @property
    def is_empirical(self) -> bool:
        return self.data_origin in EMPIRICAL_ORIGINS


def inherited_origin(parent_origins: Iterable[DataOrigin]) -> DataOrigin:
    """Deterministic origin of a derived record from its parents' origins."""
    origins = [DataOrigin(o) for o in parent_origins]
    if not origins:
        raise ProvenanceError("cannot inherit origin from an empty parent set")
    return max(origins, key=lambda o: _ORIGIN_RANK[o])


class ProvenanceRegistry:
    """Holds records and enforces lineage consistency on registration."""

    def __init__(self) -> None:
        self._records: Dict[str, ProvenanceRecord] = {}

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._records

    def get(self, record_id: str) -> ProvenanceRecord:
        return self._records[record_id]

    def records(self) -> List[ProvenanceRecord]:
        return list(self._records.values())

    def register(self, record: ProvenanceRecord | Mapping[str, Any]) -> ProvenanceRecord:
        rec = record if isinstance(record, ProvenanceRecord) else ProvenanceRecord.model_validate(dict(record))
        if rec.record_id in self._records:
            raise ProvenanceError(f"duplicate record_id {rec.record_id}")
        if rec.processing_stage is ProcessingStage.DERIVED:
            unknown = [p for p in rec.parents if p not in self._records]
            if unknown:
                raise ProvenanceError(f"{rec.record_id}: unknown parent(s) {unknown}")
            expected = inherited_origin(self._records[p].data_origin for p in rec.parents)
            if rec.data_origin is not expected:
                raise ProvenanceError(
                    f"{rec.record_id}: declared origin {rec.data_origin.value} differs from inherited "
                    f"origin {expected.value}"
                )
        self._records[rec.record_id] = rec
        return rec

    def lineage(self, record_id: str) -> List[str]:
        """All ancestors (depth-first, de-duplicated) of ``record_id``."""
        seen: List[str] = []
        stack = list(self._records[record_id].parents)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.append(current)
            stack.extend(self._records[current].parents)
        return seen

    def descendants(self, record_id: str) -> List[str]:
        out: List[str] = []
        frontier = [record_id]
        while frontier:
            current = frontier.pop()
            for rec in self._records.values():
                if current in rec.parents and rec.record_id not in out:
                    out.append(rec.record_id)
                    frontier.append(rec.record_id)
        return out

    def remove(self, record_ids: Iterable[str]) -> None:
        for rid in record_ids:
            self._records.pop(rid, None)


class ResearchStoreRejection(ProvenanceError):
    """A record was refused at the research-store boundary."""


class ResearchStoreGuard:
    """Ingestion boundary: only empirical, confirmatory-eligible records are admitted."""

    def __init__(self) -> None:
        self.admitted: List[str] = []
        self.rejections: List[Dict[str, str]] = []

    def check(self, record: ProvenanceRecord) -> None:
        if record.source_system and record.source_system in BARRED_SOURCE_SYSTEMS:
            raise ResearchStoreRejection(
                f"{record.record_id}: records from {record.source_system} are barred from the research store"
            )
        if not record.is_empirical:
            raise ResearchStoreRejection(
                f"{record.record_id}: data_origin={record.data_origin.value} cannot enter the research store"
            )

    def admit(self, record: ProvenanceRecord) -> ProvenanceRecord:
        try:
            self.check(record)
        except ResearchStoreRejection as exc:
            self.rejections.append({"record_id": record.record_id, "reason": str(exc)})
            raise
        self.admitted.append(record.record_id)
        return record


def can_support_empirical_confirmation(records: Iterable[ProvenanceRecord]) -> bool:
    """True only if every record is empirical and labelled confirmatory."""
    recs = list(records)
    return bool(recs) and all(
        r.is_empirical and r.inferential_category is InferentialCategory.CONFIRMATORY for r in recs
    )


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_record_bytes(record: ProvenanceRecord, path: Path | str) -> None:
    """Raise unless the file at ``path`` matches the record's SHA-256."""
    if not record.sha256:
        raise ProvenanceError(f"{record.record_id}: no sha256 recorded")
    actual = file_sha256(path)
    if actual != record.sha256:
        raise ProvenanceError(f"{record.record_id}: checksum mismatch ({actual} != {record.sha256})")
