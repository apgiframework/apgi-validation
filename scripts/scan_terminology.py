#!/usr/bin/env python3
"""APGI-SCI-006 terminology scan: APGI must expand only to "Allostatic Precision-Gated Ignition".

Scans every tracked text file for known forbidden expansions and for any
"A... P... G... I..." four-word expansion adjacent to "(APGI)" that differs
from the canonical one. Exit 1 on any finding.
"""

import re
import subprocess  # nosec B404
import sys

CANONICAL = "allostatic precision-gated ignition"
FORBIDDEN = [
    "adaptive predictive global integration",
    "active phase-transition global ignition",
    "active posterior global integration",
    "active predictive global ignition",
    "active predictive generative inference",
]
EXPANSION = re.compile(r"\b(A[\w-]+)\s+(P[\w-]+)[\s-]+(G[\w-]+)\s+(I[\w-]+)\s*\(APGI\)", re.IGNORECASE)
SELF = {"scripts/scan_terminology.py", "tests/test_release_and_terminology.py"}


def main() -> int:
    files = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True).stdout.split()  # nosec
    findings = []
    for path in files:
        if path in SELF:
            continue
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        low = text.lower()
        for bad in FORBIDDEN:
            if bad in low:
                findings.append(f"{path}: forbidden expansion '{bad}'")
        for m in EXPANSION.finditer(text):
            words = " ".join(m.groups()).lower().replace("precision gated", "precision-gated")
            if words != CANONICAL:
                findings.append(f"{path}: non-canonical expansion '{' '.join(m.groups())}'")
    for f in findings:
        print(f"FAIL {f}")
    print(f"Terminology scan: {len(files)} files, {len(findings)} finding(s).")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
