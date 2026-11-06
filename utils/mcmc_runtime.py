"""Safe core count for PyMC sampling (CI-MEMORY follow-up).

PyMC runs parallel chains by forking the current process. Forking a process
that already runs other threads (BLAS, torch, GUI or test-runner threads) is
unsafe: chain workers can hang and are never reaped. In the test suite one
FP_10 run left 255 orphaned workers, which later aborted the session and
killed CI runners.

``mcmc_cores(n)`` returns the number of worker processes to use:

* ``APGI_MCMC_CORES`` caps it (``1`` = sequential chains). The test suite
  sets it to 1; operators may set it in any environment.
* From a daemon thread (GUI workers) it is always 1.

Chains are seeded individually, so sequential and parallel sampling draw the
same samples; only wall-clock time changes.
"""

from __future__ import annotations

import os
import threading


def mcmc_cores(requested: int) -> int:
    requested = max(1, int(requested))
    if threading.current_thread().daemon:
        return 1
    raw = os.environ.get("APGI_MCMC_CORES", "").strip()
    if raw:
        try:
            cap = int(raw)
        except ValueError as exc:
            raise ValueError(f"APGI_MCMC_CORES must be a positive integer, got {raw!r}") from exc
        if cap < 1:
            raise ValueError(f"APGI_MCMC_CORES must be a positive integer, got {raw!r}")
        return min(requested, cap)
    return requested
