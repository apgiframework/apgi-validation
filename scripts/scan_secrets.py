#!/usr/bin/env python3
"""Content-based secret scan of tracked files and git history (APGI-API-003).

The filename check in .github/workflows/security.yml catches key files; this
script inspects *content*: private-key blocks, cloud/provider tokens and
long hex/base64 values assigned to secret-looking names.

Usage:
    python scripts/scan_secrets.py            # tracked files in the working tree
    python scripts/scan_secrets.py --history  # also every blob reachable from HEAD

Exit status 1 if any finding is reported. A line can be allow-listed with the
marker ``secret-scan: allow`` (use only for documented test fixtures).
"""

from __future__ import annotations

import argparse
import re
import subprocess  # nosec B404 - fixed git invocations only
import sys
from typing import Iterable, List, Tuple

PATTERNS = [
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9]{40,}\b")),
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b(?:[A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_KEY|MASTER_KEY|HMAC_KEY))\b\s*[:=]\s*"
            r"[\"']?([0-9a-f]{32,}|[A-Za-z0-9+/]{40,}={0,2})[\"']?"
        ),
    ),
]
ALLOW_MARKER = "secret-scan: allow"
SKIP_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".npz",
    ".npy",
    ".pkl",
    ".zip",
    ".gz",
    ".ico",
    ".woff",
    ".ttf",
)
SKIP_PATHS = ("requirements.lock",)


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout  # nosec B603 B607


def scan_text(path: str, text: str) -> List[Tuple[str, int, str]]:
    findings = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARKER in line:
            continue
        for name, pattern in PATTERNS:
            if pattern.search(line):
                findings.append((path, lineno, name))
    return findings


def _skip(path: str) -> bool:
    return path.endswith(SKIP_SUFFIXES) or path in SKIP_PATHS


def tracked_files() -> Iterable[Tuple[str, str]]:
    for path in _git("ls-files", "-z").decode().split("\0"):
        if not path or _skip(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                yield path, fh.read()
        except (UnicodeDecodeError, OSError):
            continue


def history_blobs() -> Iterable[Tuple[str, str]]:
    seen = set()
    for line in _git("rev-list", "--objects", "--all").decode().splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2 or parts[0] in seen or _skip(parts[1]):
            continue
        seen.add(parts[0])
        if _git("cat-file", "-t", parts[0]).strip() != b"blob":
            continue
        try:
            yield f"{parts[1]}@{parts[0][:10]}", _git("cat-file", "-p", parts[0]).decode("utf-8")
        except UnicodeDecodeError:
            continue


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="also scan every blob in git history")
    args = parser.parse_args(argv)
    sources = list(tracked_files())
    if args.history:
        sources.extend(history_blobs())
    findings = [f for path, text in sources for f in scan_text(path, text)]
    for path, lineno, name in findings:
        # Never print the matched value itself.
        print(f"::error file={path.split('@')[0]},line={lineno}::possible {name}")
    print(f"Scanned {len(sources)} text sources; {len(findings)} finding(s).")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
