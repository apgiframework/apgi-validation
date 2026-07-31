"""Keep .env.example in step with the variables the code actually reads.

Historical defect: .env.example documented 55 variables — database URLs, AWS
and Azure credentials, SSL paths, worker counts, cache sizes — of which 52 were
never read by any module. An operator configuring APGI_LOG_FILE or
APGI_MAX_WORKERS got silence, with no way to tell working settings from
decorative ones.

This test enforces the contract in both directions:
    - every variable documented in .env.example is read somewhere in the code
    - every APGI_* variable read by the code is documented in .env.example
"""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

# Fixture variables that exist only to exercise config loading.
EXEMPT = {
    "MISSING_VARIABLE",
    "TESTVAR",
    "TEST_VAR",
    "TEST_CONFIG_VAR",
}

# Variables consumed by third-party libraries rather than by APGI code. They
# belong in .env.example because operators may need to set them, but no APGI
# module reads them, so the "documented implies read" rule cannot apply.
EXTERNAL_TOOL_VARS = {
    "PYTENSOR_FLAGS",
}

SEARCH_DIRS = [
    "utils",
    "apgi_core",
    "gui",
    "protocols",
    "Validation",
    "Falsification",
    "Theory",
    "tests",  # APGI_REAL_TKINTER is read by conftest.py
]
SEARCH_FILES = [
    "main.py",
    "Validation_GUI.py",
    "Protocols_GUI.py",
    "Falsification_GUI.py",
    "OSF_GUI.py",
    "Utils_GUI.py",
    "Tests_GUI.py",
    "Theory_GUI.py",
]

READ_PATTERN = re.compile(r"""(?:os\.environ\.get|os\.getenv)\(\s*["']([A-Z][A-Z0-9_]*)["']""")
DECLARE_PATTERN = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)


def _python_sources():
    for name in SEARCH_FILES:
        path = PROJECT_ROOT / name
        if path.exists():
            yield path
    for directory in SEARCH_DIRS:
        base = PROJECT_ROOT / directory
        if base.exists():
            yield from base.rglob("*.py")


def variables_read_by_code():
    found = set()
    for path in _python_sources():
        try:
            found.update(READ_PATTERN.findall(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return {name for name in found if name.startswith("APGI_") or name in {"PICKLE_SECRET_KEY"}} - EXEMPT


def variables_documented():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(DECLARE_PATTERN.findall(text)) - EXEMPT - EXTERNAL_TOOL_VARS


def test_env_example_exists():
    assert ENV_EXAMPLE.exists(), ".env.example is missing"


@pytest.mark.parametrize("variable", sorted(variables_documented()))
def test_documented_variable_is_read_by_code(variable):
    """No decorative configuration: documenting a variable implies wiring it."""
    assert variable in variables_read_by_code(), (
        f"{variable} is documented in .env.example but never read by any module. "
        f"Either wire it up or remove it from .env.example."
    )


@pytest.mark.parametrize("variable", sorted(variables_read_by_code()))
def test_read_variable_is_documented(variable):
    """Operators must be able to discover every setting that has an effect."""
    assert variable in variables_documented(), (
        f"{variable} is read by the code but absent from .env.example. " f"Add it so operators can discover it."
    )


def test_required_secrets_are_documented():
    documented = variables_documented()
    for required in ("APGI_MASTER_KEY", "APGI_ENFORCE_AUTH", "APGI_LOG_LEVEL"):
        assert required in documented


def test_auth_is_documented_as_enforced_by_default():
    """The dev default must be visible in the template, not just in code."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert re.search(
        r"^APGI_ENFORCE_AUTH=0\s*$", text, re.MULTILINE
    ), "APGI_ENFORCE_AUTH should be shown as 0 (dev mode) in .env.example"


def test_no_placeholder_secret_values():
    """Placeholders like 'your_..._here' become live config if left unedited."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    offenders = re.findall(r"^\s*[A-Z_]+=.*your_.*_here.*$", text, re.MULTILINE)
    assert not offenders, f"Placeholder secret values present: {offenders}"
