"""
Protocol manifest integrity verification.

Loads config/protocol_manifest.json and exposes verify_protocol_file() so
GUI loaders can confirm a file's SHA-256 hash before exec_module() is called.

CLI usage (run from the project root)
--------------------------------------
Refresh all hashes (run after any protocol edit):

    python -m utils.protocol_manifest --refresh

Check for drift without writing:

    python -m utils.protocol_manifest --check

Both modes exit 0 on success, 1 if any mismatch or missing file is found.
The pre-commit hook calls ``--refresh`` automatically so developers never
need to run this manually.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent.parent / "config" / "protocol_manifest.json"
_manifest: Optional[dict] = None


def _load_manifest() -> dict:
    global _manifest
    if _manifest is None:
        try:
            _manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not load protocol manifest: {e}")
            _manifest = {}
    return _manifest


def _manifest_path() -> Path:
    return _MANIFEST_PATH


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_protocol_file(file_path: Path, directory_key: str) -> bool:
    """Return True if file_path matches its recorded SHA-256 hash.

    Args:
        file_path: Absolute path to the protocol .py file.
        directory_key: One of "Theory", "Validation", or "Falsification".

    Returns:
        True if the file is in the manifest and its hash matches.
        False (with a warning) if missing from manifest or hash mismatch.
        True unconditionally when APGI_SKIP_MANIFEST_CHECK=1 is set (dev override).
    """
    if os.environ.get("APGI_SKIP_MANIFEST_CHECK", "0") == "1":
        if os.environ.get("APGI_ENV", "").strip().lower() in {"prod", "production"}:
            logger.error("SECURITY: APGI_SKIP_MANIFEST_CHECK=1 is refused when APGI_ENV=production")
            return False
        logger.debug(f"Manifest check skipped for {file_path.name} (APGI_SKIP_MANIFEST_CHECK=1)")
        return True

    manifest = _load_manifest()
    dir_entries: dict = manifest.get("directories", {}).get(directory_key, {})

    if file_path.name not in dir_entries:
        logger.warning(
            f"SECURITY: {file_path.name} is not in the protocol manifest for {directory_key}/. "
            "File will not be loaded. Add it to config/protocol_manifest.json or set "
            "APGI_SKIP_MANIFEST_CHECK=1 to allow unregistered files during development."
        )
        return False

    expected = dir_entries[file_path.name]
    actual = _sha256(file_path)
    if actual != expected:
        logger.error(
            f"SECURITY: Hash mismatch for {file_path.name} in {directory_key}/. "
            f"Expected {expected[:16]}…, got {actual[:16]}…. "
            "File may have been modified externally and will not be loaded."
        )
        return False

    return True


def update_manifest_entry(file_path: Path, directory_key: str) -> None:
    """Recompute and persist the SHA-256 hash for one file.

    Useful after intentional protocol edits — call this from a dev CLI command
    rather than bypassing the manifest entirely.
    """
    manifest_file = _manifest_path()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception:
        manifest = {"version": "1.0", "directories": {}}

    manifest.setdefault("directories", {}).setdefault(directory_key, {})[file_path.name] = _sha256(file_path)
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    global _manifest
    _manifest = manifest
    logger.info(f"Manifest updated for {directory_key}/{file_path.name}")


# ---------------------------------------------------------------------------
# Bulk refresh / check helpers
# ---------------------------------------------------------------------------

#: Protocol file prefixes that belong in the manifest.
#: __init__.py, conftest.py and similar helpers are intentionally excluded.
_PROTOCOL_PREFIXES = ("VP_", "FP_", "APGI_", "Master_")

#: Directories that contain protocol files, relative to the project root.
_PROTOCOL_DIRS = ("Theory", "Validation", "Falsification")


def _project_root() -> Path:
    """Return the project root (parent of this file's package directory)."""
    return Path(__file__).parent.parent


def refresh_all(*, write: bool = True, verbose: bool = True) -> tuple[int, int]:
    """Recompute SHA-256 hashes for every tracked protocol file.

    Also **registers** any protocol file that is on disk but missing from the
    manifest, so new files are picked up automatically.

    Args:
        write:   If True (default), persist changes to protocol_manifest.json.
                 Pass False to perform a dry-run check without writing.
        verbose: Print a line for each refreshed / new / missing entry.

    Returns:
        ``(refreshed, missing)`` — counts of updated entries and files that
        are in the manifest but no longer exist on disk.
    """
    root = _project_root()
    manifest_file = _manifest_path()

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[manifest] Could not read manifest: {exc}", file=sys.stderr)
        manifest = {"version": "1.0", "directories": {}}

    refreshed = 0
    missing = 0

    for dir_key in _PROTOCOL_DIRS:
        dir_path = root / dir_key
        entries: dict = manifest.setdefault("directories", {}).setdefault(dir_key, {})

        # ── Update / register every protocol .py on disk ──────────────────
        for fpath in sorted(dir_path.glob("*.py")):
            if not any(fpath.name.startswith(p) for p in _PROTOCOL_PREFIXES):
                continue  # skip __init__.py and similar helpers

            new_hash = _sha256(fpath)
            old_hash = entries.get(fpath.name)

            if old_hash is None:
                if verbose:
                    print(f"  [+] NEW      {dir_key}/{fpath.name}")
                entries[fpath.name] = new_hash
                refreshed += 1
            elif old_hash != new_hash:
                if verbose:
                    print(f"  [~] REFRESH  {dir_key}/{fpath.name}")
                entries[fpath.name] = new_hash
                refreshed += 1

        # ── Warn about entries whose file no longer exists ─────────────────
        for fname in list(entries.keys()):
            if not (dir_path / fname).exists():
                if verbose:
                    print(f"  [!] MISSING  {dir_key}/{fname}  (deleted? remove from manifest manually)")
                missing += 1

        # Sort keys alphabetically for stable diffs
        manifest["directories"][dir_key] = dict(sorted(entries.items()))

    if write and refreshed:
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        global _manifest
        _manifest = None  # invalidate in-process cache so next verify reads fresh data

    return refreshed, missing


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m utils.protocol_manifest",
        description="Refresh or check protocol manifest hashes.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute all hashes and write to config/protocol_manifest.json.",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Check for drift without writing. Exits 1 if any mismatch found.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file output.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.refresh:
        refreshed, missing = refresh_all(write=True, verbose=not args.quiet)
        if not args.quiet:
            if refreshed:
                print(f"\n✓ Manifest refreshed — {refreshed} entries updated.")
            else:
                print("\n✓ Manifest already current — no changes.")
            if missing:
                print(f"  ⚠ {missing} manifest entries have no matching file on disk.")
        sys.exit(1 if missing else 0)

    if args.check:
        refreshed, missing = refresh_all(write=False, verbose=not args.quiet)
        if refreshed or missing:
            if not args.quiet:
                print(f"\n✗ Manifest is STALE — {refreshed} hashes differ, {missing} files missing.")
            sys.exit(1)
        if not args.quiet:
            print("\n✓ Manifest is current.")
        sys.exit(0)


if __name__ == "__main__":
    _main()
