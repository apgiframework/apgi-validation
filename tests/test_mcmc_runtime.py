"""Tests for utils.mcmc_runtime (CI-MEMORY follow-up: no forked PyMC workers under pytest)."""

import threading

import pytest

from utils.mcmc_runtime import mcmc_cores


def test_suite_runs_chains_sequentially():
    # Set session-wide by the sequential_mcmc_chains fixture in conftest.py.
    assert mcmc_cores(4) == 1


def test_cap_and_passthrough(monkeypatch):
    monkeypatch.setenv("APGI_MCMC_CORES", "2")
    assert mcmc_cores(4) == 2
    assert mcmc_cores(1) == 1
    monkeypatch.delenv("APGI_MCMC_CORES")
    assert mcmc_cores(4) == 4


@pytest.mark.parametrize("bad", ["0", "-1", "two"])
def test_invalid_cap_rejected(monkeypatch, bad):
    monkeypatch.setenv("APGI_MCMC_CORES", bad)
    with pytest.raises(ValueError):
        mcmc_cores(4)


def test_daemon_thread_is_sequential(monkeypatch):
    monkeypatch.delenv("APGI_MCMC_CORES", raising=False)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("n", mcmc_cores(4)), daemon=True)
    t.start()
    t.join()
    assert out["n"] == 1
