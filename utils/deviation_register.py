"""Registered-plan deviations and exploratory labelling (APGI-REP-004, AUD-GOV-04).

Any analysis that departs from its registered plan — an unregistered
parameter value, a substituted method, an added or omitted procedural step —
must be recorded as a deviation, and every output it touches becomes
*exploratory*: it can be reported, but it cannot carry a registered
framework vote (confirmation or falsification).

:func:`compare_to_plan` derives deviations mechanically from the registered
and executed plans, so an absent step cannot go unnoticed.
:func:`flag_result` propagates the flags into a ``ProtocolResult`` (and hence
into every export path that serialises it), and ``utils.kgate.classify_gate``
refuses a PASS/FAIL decision when deviations are supplied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class DeviationKind(str, Enum):
    PARAMETER = "parameter"
    METHOD = "method"
    OMITTED_STEP = "omitted_step"
    ADDED_STEP = "added_step"
    PROCEDURE = "procedure"


@dataclass(frozen=True)
class Deviation:
    kind: DeviationKind
    item: str
    registered: Any
    actual: Any
    rationale: str = "UNRECORDED"
    approved_by: Optional[str] = None
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


def compare_to_plan(
    registered_steps: Sequence[str],
    executed_steps: Sequence[str],
    registered_parameters: Optional[Mapping[str, Any]] = None,
    actual_parameters: Optional[Mapping[str, Any]] = None,
    registered_method: Optional[str] = None,
    actual_method: Optional[str] = None,
) -> List[Deviation]:
    """Return every difference between the registered and executed plan."""
    out: List[Deviation] = []
    executed = list(executed_steps)
    for step in registered_steps:
        if step not in executed:
            out.append(Deviation(DeviationKind.OMITTED_STEP, step, step, None))
    for step in executed:
        if step not in registered_steps:
            out.append(Deviation(DeviationKind.ADDED_STEP, step, None, step))
    order = [s for s in executed if s in registered_steps]
    expected_order = [s for s in registered_steps if s in executed]
    if order != expected_order:
        out.append(Deviation(DeviationKind.PROCEDURE, "step_order", expected_order, order))
    reg = dict(registered_parameters or {})
    act = dict(actual_parameters or {})
    for name in sorted(set(reg) | set(act)):
        if reg.get(name, _MISSING) != act.get(name, _MISSING):
            out.append(Deviation(DeviationKind.PARAMETER, name, reg.get(name), act.get(name)))
    if registered_method is not None and actual_method != registered_method:
        out.append(Deviation(DeviationKind.METHOD, "method", registered_method, actual_method))
    return out


_MISSING = object()


def flag_result(result: Any, deviations: Iterable[Deviation]) -> Any:
    """Mark a ProtocolResult (and each prediction) exploratory when deviations exist.

    Returns the same object for chaining. No-op when there are no deviations.
    """
    devs = [d.to_dict() for d in deviations]
    if not devs:
        return result
    result.metadata["deviations"] = devs
    result.metadata["inferential_category"] = "exploratory"
    result.metadata["framework_vote_eligible"] = False
    for pred in getattr(result, "named_predictions", {}).values():
        pred.metadata["inferential_category"] = "exploratory"
        pred.metadata["framework_vote_eligible"] = False
        pred.metadata["deviation_count"] = len(devs)
    return result
