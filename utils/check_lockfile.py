#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1

Verify requirements.lock is consistent with requirements.txt.

A byte-for-byte comparison against a fresh ``uv pip compile`` is the obvious
check, but it fails every time *any* transitive dependency publishes a new
release — noise unrelated to this repository. The property that actually
matters is:

    1. every direct requirement appears in the lock, and
    2. every locked version satisfies the declared version specifiers.

Run:  python3 -m utils.check_lockfile
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover - packaging ships with pip
    print("ERROR: the 'packaging' library is required to verify the lockfile")
    sys.exit(2)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIN_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(.+?)\s*$")


def normalize(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirements(path: Path, seen=None) -> List[Requirement]:
    """Parse a requirements file, following ``-r`` includes."""
    seen = seen if seen is not None else set()
    if path in seen:
        return []
    seen.add(path)

    requirements: List[Requirement] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            included = path.parent / line.split(None, 1)[1].strip()
            requirements.extend(read_requirements(included, seen))
            continue
        if line.startswith("-"):
            continue
        try:
            requirements.append(Requirement(line))
        except Exception:
            print(f"  WARNING: unparseable requirement in {path.name}: {line}")
    return requirements


def read_lock(path: Path) -> Dict[str, str]:
    """Return ``{normalised_name: version}`` from a pinned lockfile."""
    pins: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_PATTERN.match(line)
        if match:
            pins[normalize(match.group(1))] = match.group(2)
    return pins


def check() -> Tuple[List[str], int]:
    """Return (problems, number_of_pins_checked)."""
    requirements_file = PROJECT_ROOT / "requirements.txt"
    lock_file = PROJECT_ROOT / "requirements.lock"

    problems: List[str] = []
    if not lock_file.exists():
        return ([f"{lock_file.name} is missing. Run: make lock"], 0)

    requirements = read_requirements(requirements_file)
    pins = read_lock(lock_file)

    for requirement in requirements:
        if requirement.marker and not requirement.marker.evaluate():
            continue

        name = normalize(requirement.name)
        if name not in pins:
            problems.append(f"{requirement.name} is required but absent from the lock. Run: make lock")
            continue

        try:
            locked = Version(pins[name])
        except InvalidVersion:
            problems.append(f"{requirement.name}: unparseable locked version {pins[name]!r}")
            continue

        if requirement.specifier and not requirement.specifier.contains(locked, prereleases=True):
            problems.append(
                f"{requirement.name}: locked {locked} does not satisfy '{requirement.specifier}'. Run: make lock"
            )

    return problems, len(pins)


def main() -> int:
    problems, pin_count = check()
    if problems:
        print("Lockfile is inconsistent with requirements.txt:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"Lockfile is consistent: {pin_count} pinned packages satisfy all declared requirements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
