"""At-rest encryption for sensitive datasets and backups (APGI-API-003).

Backups were plain ZIP/TAR archives with an HMAC integrity signature — that
detects tampering but leaves content readable. This module encrypts a file
with AES-256-GCM (authenticated encryption).

Key: derived with HKDF-SHA256 from the managed backup key
(``utils.secure_key_manager``) using a distinct ``info`` label, so the
encryption key is never the HMAC key itself and no new secret has to be
provisioned. Pass ``key=`` to use a dedicated 32-byte key instead.

File format: ``b"APGIENC1" | 12-byte nonce | ciphertext+tag``; the original
file name is bound as associated data so a ciphertext cannot be silently
renamed onto another dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

MAGIC = b"APGIENC1"
NONCE_BYTES = 12
SUFFIX = ".apgienc"


class PayloadEncryptionError(ValueError):
    """Encryption or authenticated decryption failed."""


def derive_key(master_hex: Optional[str] = None) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    if master_hex is None:
        from utils.secure_key_manager import get_backup_hmac_key

        master_hex = get_backup_hmac_key()
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"apgi-backup-encryption-v1").derive(
        bytes.fromhex(master_hex)
    )


def encrypt_bytes(data: bytes, aad: bytes, key: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(NONCE_BYTES)
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, data, aad)


def decrypt_bytes(blob: bytes, aad: bytes, key: bytes) -> bytes:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not blob.startswith(MAGIC):
        raise PayloadEncryptionError("not an APGI-encrypted payload")
    nonce = blob[len(MAGIC) : len(MAGIC) + NONCE_BYTES]
    try:
        return AESGCM(key).decrypt(nonce, blob[len(MAGIC) + NONCE_BYTES :], aad)
    except InvalidTag as exc:
        raise PayloadEncryptionError("authentication failed: wrong key, modified data or renamed file") from exc


def encrypt_file(path: Path | str, key: Optional[bytes] = None, remove_plaintext: bool = True) -> Path:
    """Encrypt ``path`` to ``path + .apgienc``; by default the plaintext is removed."""
    src = Path(path)
    key = key or derive_key()
    out = src.with_name(src.name + SUFFIX)
    out.write_bytes(encrypt_bytes(src.read_bytes(), src.name.encode("utf-8"), key))
    os.chmod(out, 0o600)
    if remove_plaintext:
        src.unlink()
    return out


def decrypt_file(path: Path | str, key: Optional[bytes] = None, out: Optional[Path | str] = None) -> Path:
    """Decrypt an ``.apgienc`` file; raises on any authentication failure."""
    src = Path(path)
    if not src.name.endswith(SUFFIX):
        raise PayloadEncryptionError(f"expected a {SUFFIX} file: {src}")
    original_name = src.name[: -len(SUFFIX)]
    key = key or derive_key()
    plaintext = decrypt_bytes(src.read_bytes(), original_name.encode("utf-8"), key)
    target = Path(out) if out is not None else src.with_name(original_name)
    target.write_bytes(plaintext)
    os.chmod(target, 0o600)
    return target
