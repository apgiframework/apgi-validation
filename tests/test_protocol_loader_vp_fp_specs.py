import pytest

from utils.protocol_loader import load_all_protocols, load_protocol


class TestCanonicalProtocolLoading:
    """Tests for canonical APGI-P## protocol naming (Phase 4)."""

    def test_load_protocol_resolves_canonical_apgi_p_ids(self):
        """Test that canonical APGI-P## IDs load successfully."""
        # Load canonical protocols
        apgi_p01 = load_protocol("APGI-P01")
        apgi_p02 = load_protocol("APGI-P02")

        # Both should load successfully
        assert apgi_p01 is not None
        assert apgi_p02 is not None
        assert apgi_p01.protocol_id == "APGI-P01"
        assert apgi_p02.protocol_id == "APGI-P02"

    def test_load_all_canonical_protocols(self):
        """Test loading all canonical APGI-P## protocols."""
        specs = load_all_protocols()

        # Should have only canonical protocols
        assert len(specs) > 0
        # All protocols should start with APGI-P
        for protocol_id in specs.keys():
            assert protocol_id.startswith("APGI-P"), f"Unexpected protocol ID: {protocol_id}"

    def test_canonical_protocol_structure(self):
        """Test that canonical protocols have expected structure."""
        spec = load_protocol("APGI-P01")
        assert spec is not None
        assert spec.protocol_id == "APGI-P01"
        assert hasattr(spec, "title")
        assert hasattr(spec, "version")
        assert hasattr(spec, "apgi_parameters")
        assert hasattr(spec, "sub_predictions")


class TestLegacyProtocolRejection:
    """Tests for legacy VP-##/FP-## protocol rejection (Phase 4)."""

    def test_load_protocol_vp_legacy_raises_error(self):
        """Test that legacy VP-## IDs raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_protocol("VP-01")

        # Error message should indicate legacy support removed
        assert "PHASE 4" in str(exc_info.value)
        assert "no longer supported" in str(exc_info.value)
        assert "APGI-P" in str(exc_info.value)

    def test_load_protocol_fp_legacy_raises_error(self):
        """Test that legacy FP-## IDs raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_protocol("FP-01")

        # Error message should indicate legacy support removed
        assert "PHASE 4" in str(exc_info.value)
        assert "no longer supported" in str(exc_info.value)

    def test_load_protocol_vp_variants_raise_errors(self):
        """Test that various legacy VP-## IDs all raise errors."""
        legacy_ids = ["VP-00", "VP-07", "VP-12", "VP-20", "VP-22"]
        for legacy_id in legacy_ids:
            with pytest.raises(ValueError) as exc_info:
                load_protocol(legacy_id)
            assert "PHASE 4" in str(exc_info.value)

    def test_load_protocol_fp_variants_raise_errors(self):
        """Test that various legacy FP-## IDs all raise errors."""
        legacy_ids = ["FP-03", "FP-04", "FP-12", "FP-13", "FP-14", "FP-15"]
        for legacy_id in legacy_ids:
            with pytest.raises(ValueError) as exc_info:
                load_protocol(legacy_id)
            assert "PHASE 4" in str(exc_info.value)

    def test_legacy_aliases_also_raise_errors(self):
        """Test that runtime aliases for legacy protocols also raise errors."""
        legacy_aliases = ["VP_07_TMSCausalInterventions", "FP_01_ActiveInference"]
        for alias in legacy_aliases:
            # These should not find the file via alias since legacy search is removed
            result = load_protocol(alias)
            # Either raises error or returns None (no match found)
            assert result is None

    def test_load_all_protocols_contains_only_canonical(self):
        """Verify load_all_protocols returns only canonical APGI-P## protocols."""
        specs = load_all_protocols()

        # Should NOT have any VP-## or FP-## protocols
        for protocol_id in specs.keys():
            assert not protocol_id.startswith("VP-"), f"Found legacy VP-## protocol: {protocol_id}"
            assert not protocol_id.startswith("FP-"), f"Found legacy FP-## protocol: {protocol_id}"
            assert protocol_id.startswith("APGI-P"), f"Unexpected protocol ID: {protocol_id}"

    def test_canonical_protocol_count(self):
        """Verify only canonical protocols are loaded (8 total)."""
        specs = load_all_protocols()

        # 8 canonical protocols: APGI-P00 through APGI-P07
        assert len(specs) == 8, f"Expected 8 canonical protocols, got {len(specs)}: {list(specs.keys())}"
