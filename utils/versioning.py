"""Version domains of this application (APGI-REL-004, APGI-INT-001).

These are *different* version domains; equality across them is not implied.
The tested relationships between them are recorded in
``docs/validation/compatibility-matrix.md``.
"""

#: Application (package) version — must equal [project].version in pyproject.toml.
APP_VERSION = "1.3.0"
#: Protocol definition bundle verified by utils.protocol_lock.
PROTOCOL_BUNDLE_VERSION = "1.1.0"
#: Local inter-protocol contract version (utils/protocol_contracts.py).
LOCAL_CONTRACT_VERSION = "1.0"
#: ProtocolResult exchange schema version (utils/protocol_schema.py).
PROTOCOL_RESULT_SCHEMA_VERSION = "2.0.0"
#: Governing software specification this release is assessed against.
GOVERNING_SPECIFICATION = "APGI Protocol v1.2 Software Validation Specification (draft)"
