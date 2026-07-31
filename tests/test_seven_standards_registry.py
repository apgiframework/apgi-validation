"""Tests for Seven Standards Registry module - comprehensive coverage."""

from utils.seven_standards_registry import (
    SEVEN_STANDARDS,
    ComplianceLevel,
    Standard,
    StandardCategory,
    StandardsRegistry,
    check_compliance,
    get_standard,
    list_standards,
    register_standard,
)


class TestStandard:
    """Test Standard dataclass."""

    def test_standard_creation(self):
        """Test creating a standard."""
        from utils.seven_standards_registry import StandardCategory, ValidationCriterion

        standard = Standard(
            standard_id="TEST-001",
            name="Test Standard",
            category=StandardCategory.MATHEMATICAL,
            description="A test standard",
            rationale="Test rationale",
            criteria=[
                ValidationCriterion(
                    name="Test Criterion",
                    description="Test description",
                    falsification_threshold="Test threshold",
                    test_method="Test method",
                    effect_size_requirement="Test effect",
                    alternative_criterion="Test alternative",
                    priority_level="Medium",
                    current_status="Proposed",
                )
            ],
            overall_status="Proposed",
        )
        assert standard.standard_id == "TEST-001"
        assert standard.name == "Test Standard"


class TestComplianceLevel:
    """Test Compliance Level enum."""

    def test_compliance_levels(self):
        """Test compliance level values."""
        assert ComplianceLevel.NONE.value == 0
        assert ComplianceLevel.PARTIAL.value == 1
        assert ComplianceLevel.FULL.value == 2


class TestStandardsRegistry:
    """Test Standards Registry."""

    def test_init(self):
        """Test registry initialization."""
        registry = StandardsRegistry()
        assert registry is not None

    def test_register_standard(self):
        """Test registering a standard."""
        from utils.seven_standards_registry import StandardCategory, ValidationCriterion

        registry = StandardsRegistry()
        standard = Standard(
            standard_id="REG-001",
            name="Registration Test",
            category=StandardCategory.EMPIRICAL,
            description="Test registration",
            rationale="Test rationale",
            criteria=[
                ValidationCriterion(
                    name="Test Criterion",
                    description="Test description",
                    falsification_threshold="Test threshold",
                    test_method="Test method",
                    effect_size_requirement="Test effect",
                    alternative_criterion="Test alternative",
                    priority_level="Medium",
                    current_status="Proposed",
                )
            ],
            overall_status="Proposed",
        )
        registry.register_standard(standard)
        assert "REG-001" in registry._standards

    def test_get_standard(self):
        """Test retrieving a standard."""
        from utils.seven_standards_registry import StandardCategory, ValidationCriterion

        registry = StandardsRegistry()
        standard = Standard(
            standard_id="GET-001",
            name="Get Test",
            category=StandardCategory.COMPUTATIONAL,
            description="Test get",
            rationale="Test rationale",
            criteria=[
                ValidationCriterion(
                    name="Test Criterion",
                    description="Test description",
                    falsification_threshold="Test threshold",
                    test_method="Test method",
                    effect_size_requirement="Test effect",
                    alternative_criterion="Test alternative",
                    priority_level="Medium",
                    current_status="Proposed",
                )
            ],
            overall_status="Proposed",
        )
        registry.register_standard(standard)

        retrieved = registry.get_standard("GET-001")
        assert retrieved == standard

    def test_list_standards(self):
        """Test listing all standards."""
        registry = StandardsRegistry()
        standards = registry.list_standards()
        assert isinstance(standards, list)


class TestRegisterStandard:
    """Test standalone register function."""

    def test_register_global_standard(self):
        """Test registering to global registry."""
        from utils.seven_standards_registry import StandardCategory, ValidationCriterion

        standard = Standard(
            standard_id="GLOBAL-001",
            name="Global Test",
            category=StandardCategory.NEUROBIOLOGICAL,
            description="Test global registration",
            rationale="Test rationale",
            criteria=[
                ValidationCriterion(
                    name="Test Criterion",
                    description="Test description",
                    falsification_threshold="Test threshold",
                    test_method="Test method",
                    effect_size_requirement="Test effect",
                    alternative_criterion="Test alternative",
                    priority_level="Medium",
                    current_status="Proposed",
                )
            ],
            overall_status="Proposed",
        )
        register_standard(standard)
        # Should be accessible via get_standard
        retrieved = get_standard("GLOBAL-001")
        assert retrieved is not None


class TestGetStandard:
    """Test standalone get function."""

    def test_get_existing_standard(self):
        """Test getting an existing standard."""
        from utils.seven_standards_registry import StandardCategory, ValidationCriterion

        # Register first
        standard = Standard(
            standard_id="EXIST-001",
            name="Existing",
            category=StandardCategory.PREDICTIVE,
            description="Test existing",
            rationale="Test rationale",
            criteria=[
                ValidationCriterion(
                    name="Test Criterion",
                    description="Test description",
                    falsification_threshold="Test threshold",
                    test_method="Test method",
                    effect_size_requirement="Test effect",
                    alternative_criterion="Test alternative",
                    priority_level="Medium",
                    current_status="Proposed",
                )
            ],
            overall_status="Proposed",
        )
        register_standard(standard)

        result = get_standard("EXIST-001")
        assert result is not None
        assert result.name == "Existing"

    def test_get_nonexistent_standard(self):
        """Test getting non-existent standard."""
        result = get_standard("NONEXISTENT-999")
        assert result is None


class TestListStandards:
    """Test listing standards."""

    def test_list_returns_standards(self):
        """Test that listing returns standards."""
        standards = list_standards()
        assert isinstance(standards, list)

    def test_list_filtered_by_category(self):
        """Test filtering standards by category."""
        standards = list_standards(category=StandardCategory.EMPIRICAL)
        assert isinstance(standards, list)


class TestCheckCompliance:
    """Test compliance checking."""

    def test_check_full_compliance(self):
        """Test checking full compliance."""
        from utils.seven_standards_registry import StandardCategory, ValidationCriterion

        standard = Standard(
            standard_id="COMP-001",
            name="Compliance Test",
            category=StandardCategory.CROSS_SPECIES,
            description="Test compliance",
            rationale="Test rationale",
            criteria=[
                ValidationCriterion(
                    name="Test Criterion 1",
                    description="Test description",
                    falsification_threshold="Test threshold",
                    test_method="Test method",
                    effect_size_requirement="Test effect",
                    alternative_criterion="Test alternative",
                    priority_level="Medium",
                    current_status="Implemented",
                ),
                ValidationCriterion(
                    name="Test Criterion 2",
                    description="Test description",
                    falsification_threshold="Test threshold",
                    test_method="Test method",
                    effect_size_requirement="Test effect",
                    alternative_criterion="Test alternative",
                    priority_level="Medium",
                    current_status="Implemented",
                ),
            ],
            overall_status="Implemented",
        )
        register_standard(standard)

        result = check_compliance("COMP-001", 0.95)
        assert result == ComplianceLevel.FULL

    def test_check_partial_compliance(self):
        """Test checking partial compliance."""
        from utils.seven_standards_registry import StandardCategory, ValidationCriterion

        standard = Standard(
            standard_id="PART-001",
            name="Partial Test",
            category=StandardCategory.CLINICAL,
            description="Test partial",
            rationale="Test rationale",
            criteria=[
                ValidationCriterion(
                    name="Test Criterion 1",
                    description="Test description",
                    falsification_threshold="Test threshold",
                    test_method="Test method",
                    effect_size_requirement="Test effect",
                    alternative_criterion="Test alternative",
                    priority_level="Medium",
                    current_status="Implemented",
                ),
                ValidationCriterion(
                    name="Test Criterion 2",
                    description="Test description",
                    falsification_threshold="Test threshold",
                    test_method="Test method",
                    effect_size_requirement="Test effect",
                    alternative_criterion="Test alternative",
                    priority_level="Medium",
                    current_status="Implemented",
                ),
                ValidationCriterion(
                    name="Test Criterion 3",
                    description="Test description",
                    falsification_threshold="Test threshold",
                    test_method="Test method",
                    effect_size_requirement="Test effect",
                    alternative_criterion="Test alternative",
                    priority_level="Medium",
                    current_status="Implemented",
                ),
            ],
            overall_status="Partial",
        )
        register_standard(standard)

        result = check_compliance("PART-001", 0.7)
        assert result == ComplianceLevel.PARTIAL


class TestSevenStandards:
    """Test the seven standards constant."""

    def test_seven_standards_exist(self):
        """Test that seven standards are defined."""
        assert SEVEN_STANDARDS is not None
        assert len(SEVEN_STANDARDS) == 7

    def test_seven_standards_are_valid(self):
        """Test that all seven standards are valid."""
        # SEVEN_STANDARDS is a dictionary, not a list of Standard objects
        assert isinstance(SEVEN_STANDARDS, dict)
        assert len(SEVEN_STANDARDS) == 7

        # Check that all keys are present
        expected_keys = [
            "mathematical",
            "empirical",
            "computational",
            "neurobiological",
            "predictive",
            "cross_species",
            "clinical",
        ]
        for key in expected_keys:
            assert key in SEVEN_STANDARDS
            assert isinstance(SEVEN_STANDARDS[key], str)
