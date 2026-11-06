#!/usr/bin/env python3
"""Notation inventory scan (APGI-SCI-004).

Lists identifiers whose names are bare members of symbol families that the
specification says must carry disambiguating subscripts (alpha, beta, gamma,
pi/Pi, theta, tau). Writes a machine-readable inventory so the approved
Notation Appendix, once supplied, can be applied and a zero-findings scan
retained.

    python scripts/scan_notation.py            # summary
    python scripts/scan_notation.py --json out.json
    python scripts/scan_notation.py --fail     # exit 1 if any bare symbol remains

``--fail`` is not used in CI yet: renaming before the appendix exists would
only introduce a third convention (docs/validation/notation-and-claims.md).
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess  # nosec B404
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BARE = {"alpha", "beta", "gamma", "pi", "Pi", "theta", "tau"}


def _names(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id, node.lineno
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            yield node.target.id, node.lineno


def scan():
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, check=True, capture_output=True, text=True)  # nosec
    findings = defaultdict(list)
    for rel in files.stdout.split():
        if rel.startswith("tests/"):
            continue
        try:
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for name, line in _names(tree):
            if name in BARE:
                findings[name].append(f"{rel}:{line}")
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args(argv)
    findings = scan()
    counts = Counter({k: len(v) for k, v in findings.items()})
    for name, n in counts.most_common():
        print(f"{name:6s} {n:5d} bare uses")
    print(f"total bare symbol bindings: {sum(counts.values())}")
    if args.json:
        args.json.write_text(json.dumps({k: sorted(set(v)) for k, v in findings.items()}, indent=2) + "\n")
    return 1 if (args.fail and findings) else 0


if __name__ == "__main__":
    sys.exit(main())
