"""Named scientific failure types (APGI-INT-003, APGI-REP-003).

A missing dependency, unsupported method or failed numerical routine must
surface as one of these errors. Code must never substitute a plausible
constant (e.g. H=0.5, power=0.8, n=50) and report it as a computed result.
"""


class ScientificComputationError(RuntimeError):
    """A scientific quantity could not be computed; no default was substituted."""


class UnsupportedMethodError(ScientificComputationError, ValueError):
    """The requested statistical/scientific method is not implemented or validated."""


class MissingScientificDependencyError(ScientificComputationError, ImportError):
    """A dependency required for a validated computation is not installed."""
