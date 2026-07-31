"""Protocol naming migration tests — Phase 4 (hard deprecation).

This suite previously targeted Phase 3 "soft deprecation", importing
``_emit_legacy_protocol_warning`` and ``_is_legacy_protocol``. Both were
removed when utils/protocol_loader.py advanced to Phase 4, so the module
failed at collection and the whole suite errored out.

Phase 4 is now the contract: legacy VP-##/FP-## identifiers are rejected with
a ValueError carrying migration guidance, and only canonical APGI-P## IDs
resolve. These tests assert that contract.
"""

import pytest

from utils.protocol_loader import load_all_protocols, load_protocol

LEGACY_IDS = ["VP-01", "VP-02", "VP-14", "FP-01", "FP-02", "FP-15"]
CANONICAL_IDS = ["APGI-P00", "APGI-P01", "APGI-P02", "APGI-P07"]


class TestLegacyProtocolIdsRejected:
    """Legacy identifiers must fail loudly, not silently resolve."""

    @pytest.mark.parametrize("legacy_id", LEGACY_IDS)
    def test_legacy_id_raises_value_error(self, legacy_id):
        with pytest.raises(ValueError):
            load_protocol(legacy_id)

    @pytest.mark.parametrize("legacy_id", LEGACY_IDS)
    def test_error_message_carries_migration_guidance(self, legacy_id):
        """An operator hitting this must learn what to do next."""
        with pytest.raises(ValueError) as excinfo:
            load_protocol(legacy_id)

        message = str(excinfo.value)
        assert "PROTOCOL MIGRATION" in message
        assert "PHASE 4" in message
        assert legacy_id in message
        assert "APGI-P" in message, "Error should name the canonical replacement scheme"
        assert "PROTOCOL-MIGRATION-USER-GUIDE.md" in message

    def test_legacy_prefix_rejected_regardless_of_suffix(self):
        """The guard keys on the prefix, so arbitrary suffixes are still rejected."""
        for bogus in ("VP-999", "FP-abc", "VP-"):
            with pytest.raises(ValueError):
                load_protocol(bogus)

    def test_whitespace_padded_legacy_id_still_rejected(self):
        """IDs are normalised before the check; padding must not bypass it."""
        with pytest.raises(ValueError):
            load_protocol("  VP-01  ")


class TestCanonicalProtocolIdsResolve:
    """Canonical naming is the supported path."""

    @pytest.mark.parametrize("protocol_id", CANONICAL_IDS)
    def test_canonical_id_does_not_raise(self, protocol_id):
        load_protocol(protocol_id)

    @pytest.mark.parametrize("protocol_id", CANONICAL_IDS)
    def test_canonical_id_returns_spec_or_none(self, protocol_id):
        """Unknown-but-canonical IDs return None rather than raising."""
        result = load_protocol(protocol_id)
        assert result is None or hasattr(result, "protocol_id")

    def test_unknown_canonical_id_returns_none(self):
        assert load_protocol("APGI-P99") is None


class TestProtocolRegistry:
    def test_load_all_protocols_returns_mapping(self):
        protocols = load_all_protocols()
        assert isinstance(protocols, dict)

    def test_registry_contains_no_legacy_ids(self):
        """Phase 4 removed protocols/legacy/ entirely."""
        offenders = [pid for pid in load_all_protocols() if pid.startswith(("VP-", "FP-"))]
        assert offenders == [], f"Legacy protocol IDs still registered: {offenders}"

    def test_registry_ids_are_canonical(self):
        protocols = load_all_protocols()
        if protocols:
            assert all(pid.startswith("APGI-P") for pid in protocols), f"Non-canonical IDs present: {sorted(protocols)}"
