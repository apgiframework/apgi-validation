"""
Tests for utils modules with 0% coverage
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAuthAdapter:
    """Test auth_adapter.py module."""

    def test_role_enum(self):
        """Test Role enum values."""
        from utils.auth_adapter import Role

        assert Role.ADMIN.value == "admin"  # nosec B101
        assert Role.RESEARCHER.value == "researcher"  # nosec B101
        assert Role.AUDITOR.value == "auditor"  # nosec B101
        assert Role.GUEST.value == "guest"  # nosec B101

    def test_default_role(self):
        """Test default role is GUEST."""
        from utils.auth_adapter import DEFAULT_ROLE, Role

        assert DEFAULT_ROLE == Role.GUEST  # nosec B101

    def test_auth_session_init(self):
        """Test AuthSession initialization."""
        import time

        from utils.auth_adapter import AuthSession, Role

        session = AuthSession("user123", Role.ADMIN, "token123", time.time() + 3600)
        assert session.user_id == "user123"  # nosec B101
        assert session.role == Role.ADMIN  # nosec B101
        assert session.token == "token123"  # nosec B101  # nosec B105

    def test_auth_session_is_valid(self):
        """Test AuthSession validity check."""
        import time

        from utils.auth_adapter import AuthSession, Role

        # Valid session
        expires_at = time.time() + 3600
        valid_session = AuthSession("user123", Role.ADMIN, "token123", expires_at)
        assert valid_session.is_valid() is True  # nosec B101

        # Expired session
        expired_at = time.time() - 3600
        expired_session = AuthSession("user123", Role.ADMIN, "token123", expired_at)
        assert expired_session.is_valid() is False  # nosec B101

    def test_auth_adapter_init(self):
        """Test AuthAdapter initialization."""
        from utils.auth_adapter import AuthAdapter

        adapter = AuthAdapter()
        assert adapter.algorithm == "HS256"  # nosec B101
        assert isinstance(adapter._sessions, dict)  # nosec B101

    def test_auth_adapter_generate_token(self):
        """Test token generation."""
        from utils.auth_adapter import AuthAdapter, Role

        adapter = AuthAdapter()
        token = adapter.generate_token("user123", Role.ADMIN)
        assert isinstance(token, str)  # nosec B101
        assert len(token) > 0  # nosec B101

    def test_auth_adapter_validate_token_valid(self):
        """Test valid token validation."""
        from utils.auth_adapter import AuthAdapter, Role

        adapter = AuthAdapter()
        token = adapter.generate_token("user123", Role.ADMIN)
        session = adapter.validate_token(token)
        assert session is not None  # nosec B101
        assert session.user_id == "user123"  # nosec B101
        assert session.role == Role.ADMIN  # nosec B101

    def test_auth_adapter_validate_token_invalid(self):
        """Test invalid token validation."""
        from utils.auth_adapter import AuthAdapter

        adapter = AuthAdapter()
        session = adapter.validate_token("invalid_token")
        assert session is None  # nosec B101

    def test_auth_adapter_check_permission_granted(self):
        """Test permission check when granted."""
        from utils.auth_adapter import AuthAdapter, Role

        adapter = AuthAdapter()
        token = adapter.generate_token("user123", Role.ADMIN)
        result = adapter.check_permission(token, [Role.ADMIN, Role.RESEARCHER])
        assert result is True  # nosec B101  # nosec B101

    def test_auth_adapter_check_permission_denied(self):
        """Test permission check when denied."""
        from utils.auth_adapter import AuthAdapter, Role

        adapter = AuthAdapter()
        token = adapter.generate_token("user123", Role.GUEST)
        result = adapter.check_permission(token, [Role.ADMIN])
        assert result is False  # nosec B101  # nosec B101

    def test_auth_adapter_check_permission_no_admin_override(self):
        """APGI-API-002: no implicit admin bypass; ADMIN must be listed explicitly."""
        from utils.auth_adapter import AuthAdapter, Role

        adapter = AuthAdapter()
        token = adapter.generate_token("user123", Role.ADMIN)
        assert adapter.check_permission(token, [Role.RESEARCHER]) is False  # nosec B101
        assert adapter.check_permission(token, [Role.RESEARCHER, Role.ADMIN]) is True  # nosec B101

    def test_auth_manager_global(self):
        """Test global auth_manager instance."""
        from utils.auth_adapter import get_auth_manager

        auth_manager = get_auth_manager()
        assert auth_manager is not None  # nosec B101
        assert hasattr(auth_manager, "generate_token")  # nosec B101

    def test_require_roles_decorator(self):
        """Test require_roles decorator."""
        from utils.auth_adapter import Role, get_auth_manager, require_roles

        @require_roles([Role.ADMIN])
        def protected_function(token):
            return "success"

        # Generate admin token
        auth_manager = get_auth_manager()
        token = auth_manager.generate_token("user123", Role.ADMIN)
        result = protected_function(token)
        assert result == "success"  # nosec B101

    def test_require_roles_decorator_denied(self):
        """Test require_roles decorator with insufficient role."""
        from utils.auth_adapter import Role, get_auth_manager, require_roles

        @require_roles([Role.ADMIN])
        def protected_function(token):
            return "success"

        # Generate guest token
        auth_manager = get_auth_manager()
        token = auth_manager.generate_token("user123", Role.GUEST)
        try:
            protected_function(token)
            assert False, "Should have raised PermissionError"  # nosec B101
        except PermissionError:
            pass


class TestComplianceExport:
    """Test compliance_export.py module."""

    def test_compliance_exporter_init(self):
        """Test ComplianceExporter initialization."""
        from utils.compliance_export import ComplianceExporter

        exporter = ComplianceExporter()
        assert exporter.logs_dir is not None  # nosec B101
        assert exporter.export_dir is not None  # nosec B101

    def test_compliance_exporter_get_signing_key(self):
        """Test signing key retrieval."""
        from utils.compliance_export import ComplianceExporter

        exporter = ComplianceExporter()
        key = exporter._get_signing_key()
        assert isinstance(key, bytes)  # nosec B101  # nosec B101
        assert len(key) > 0  # nosec B101  # nosec B101

    def test_compliance_exporter_generate_report(self):
        """Test compliance report generation."""
        from utils.compliance_export import ComplianceExporter

        exporter = ComplianceExporter()
        report = exporter.generate_compliance_report()
        assert isinstance(report, dict)  # nosec B101  # nosec B101
        assert "report_timestamp" in report  # nosec B101  # nosec B101
        assert "controls_status" in report  # nosec B101  # nosec B101
        assert "recent_audit_events" in report  # nosec B101  # nosec B101

    def test_compliance_exporter_fetch_recent_events(self):
        """Test fetching recent audit events."""
        from utils.compliance_export import ComplianceExporter

        exporter = ComplianceExporter()
        events = exporter._fetch_recent_audit_events()
        assert isinstance(events, list)  # nosec B101  # nosec B101

    def test_compliance_exporter_export_audit_log(self):
        """Test audit log export."""
        from utils.compliance_export import ComplianceExporter

        exporter = ComplianceExporter()
        report_data = {"test": "data"}
        export_path = exporter.export_audit_log(report_data)
        assert isinstance(export_path, str)  # nosec B101
        assert export_path.endswith(".json")  # nosec B101

    def test_run_compliance_export(self):
        """Test run_compliance_export function."""
        from utils.compliance_export import run_compliance_export

        export_path = run_compliance_export()
        assert isinstance(export_path, str)  # nosec B101
        assert export_path.endswith(".json")  # nosec B101


class TestDTO:
    """Test dto.py module."""

    def test_performance_metric_dto(self):
        """Test PerformanceMetricDTO."""
        from utils.dto import PerformanceMetricDTO

        dto = PerformanceMetricDTO(
            p95_latency_ms=100.0,
            throughput_ops_per_sec=1000.0,
        )
        assert dto.p95_latency_ms == 100.0  # nosec B101
        assert dto.throughput_ops_per_sec == 1000.0  # nosec B101
        assert dto.timestamp is not None  # nosec B101

    def test_performance_metric_dto_optional_fields(self):
        """Test PerformanceMetricDTO with optional fields."""
        from utils.dto import PerformanceMetricDTO

        dto = PerformanceMetricDTO(
            p95_latency_ms=100.0,
            throughput_ops_per_sec=1000.0,
            cpu_usage_percent=50.0,
            memory_usage_mb=512.0,
        )
        assert dto.cpu_usage_percent == 50.0  # nosec B101
        assert dto.memory_usage_mb == 512.0  # nosec B101

    def test_validation_tier_summary_dto(self):
        """Test ValidationTierSummaryDTO."""
        from utils.dto import ValidationTierSummaryDTO

        dto = ValidationTierSummaryDTO(passed=10, total=15, pending=5)
        assert dto.passed == 10  # nosec B101
        assert dto.total == 15  # nosec B101
        assert dto.pending == 5  # nosec B101

    def test_validation_tier_summary_dto_defaults(self):
        """Test ValidationTierSummaryDTO defaults."""
        from utils.dto import ValidationTierSummaryDTO

        dto = ValidationTierSummaryDTO()
        assert dto.passed == 0  # nosec B101
        assert dto.total == 0  # nosec B101
        assert dto.pending == 0  # nosec B101
        assert dto.success_rate == 0.0  # nosec B101

    def test_master_validation_report_dto(self):
        """Test MasterValidationReportDTO."""
        from pydantic import BaseModel

        from utils.dto import MasterValidationReportDTO, ValidationTierSummaryDTO

        # Define ProtocolResult to resolve forward reference
        class ProtocolResult(BaseModel):  # type: ignore[misc]
            protocol_name: str
            status: str
            timestamp: str

            def __init__(self, protocol_name: str, status: str, timestamp: str) -> None:
                super().__init__(protocol_name=protocol_name, status=status, timestamp=timestamp)

        # Rebuild the model to resolve the forward reference
        MasterValidationReportDTO.model_rebuild()

        tier_summary = {"tier1": ValidationTierSummaryDTO(passed=5, total=10)}
        dto = MasterValidationReportDTO(
            overall_decision="PASS",
            total_protocols=10,
            completed_protocols=10,
            passed_protocols=8,
            pending_protocols=0,
            success_rate=0.8,
            weighted_score=0.85,
            tier_summary=tier_summary,
            protocol_results={},
            summary="Test summary",
        )
        assert dto.overall_decision == "PASS"  # nosec B101
        assert dto.success_rate == 0.8  # nosec B101

    def test_service_response_dto(self):
        """Test ServiceResponseDTO."""
        from utils.dto import ServiceResponseDTO

        dto = ServiceResponseDTO(success=True, data={"key": "value"})
        assert dto.success is True  # nosec B101
        assert dto.data == {"key": "value"}  # nosec B101

    def test_service_response_dto_error(self):
        """Test ServiceResponseDTO error case."""
        from utils.dto import ServiceResponseDTO

        dto = ServiceResponseDTO(success=False, error_code="ERR001", message="Error occurred")
        assert dto.success is False  # nosec B101  # nosec B101
        assert dto.error_code == "ERR001"  # nosec B101

    def test_audit_log_dto(self):
        """Test AuditLogDTO."""
        from utils.dto import AuditLogDTO

        dto = AuditLogDTO(
            event_type="login",
            user_id="user123",
            action="authenticate",
            resource="system",
            status="success",
        )
        assert dto.event_type == "login"  # nosec B101
        assert dto.user_id == "user123"  # nosec B101

    def test_config_explain_dto(self):
        """Test ConfigExplainDTO."""
        from utils.dto import ConfigExplainDTO

        dto = ConfigExplainDTO(
            resolved_config={"key": "value"},
            sources={"key": "env"},
            precedence_order=["env", "file"],
        )
        assert dto.resolved_config == {"key": "value"}  # nosec B101
        assert "env" in dto.precedence_order  # nosec B101

    def test_error_response_dto(self):
        """Test ErrorResponseDTO."""
        from utils.dto import ErrorResponseDTO

        dto = ErrorResponseDTO(
            code="ERR001",
            message="Test error",
            category="validation",
            severity="high",
            correlation_id="corr123",
        )
        assert dto.code == "ERR001"  # nosec B101
        assert dto.success is False  # nosec B101  # nosec B101
        assert dto.severity == "high"  # nosec B101


class TestFileOps:
    """Test file_ops.py module."""

    @patch("utils.file_ops.validate_file_path")
    @patch("utils.file_ops._ops")
    def test_read_text_file(self, mock_ops, mock_validate):
        """Test reading text file."""
        from utils.file_ops import read_text_file

        mock_validate.return_value = Path("/valid/path")
        mock_ops.safe_read.return_value = "file content"

        result = read_text_file("test.txt", "/base")
        assert result == "file content"  # nosec B101

    @patch("utils.file_ops.validate_file_path")
    @patch("utils.file_ops._ops")
    def test_read_text_file_error(self, mock_ops, mock_validate):
        """Test reading text file with error."""
        from utils.file_ops import read_text_file

        mock_validate.return_value = Path("/valid/path")
        mock_ops.safe_read.return_value = None

        try:
            read_text_file("test.txt", "/base")
            assert False, "Should have raised IOError"  # nosec B101  # nosec B101
        except IOError:
            pass

    @patch("utils.file_ops.validate_file_path")
    @patch("utils.file_ops._ops")
    def test_write_text_file(self, mock_ops, mock_validate):
        """Test writing text file."""
        from utils.file_ops import write_text_file

        mock_validate.return_value = Path("/valid/path")
        mock_ops.safe_write.return_value = True

        write_text_file("test.txt", "/base", "content")
        mock_ops.safe_write.assert_called_once()  # nosec B101

    @patch("utils.file_ops.validate_file_path")
    @patch("utils.file_ops._ops")
    def test_write_text_file_error(self, mock_ops, mock_validate):
        """Test writing text file with error."""
        from utils.file_ops import write_text_file

        mock_validate.return_value = Path("/valid/path")
        mock_ops.safe_write.return_value = False

        try:
            write_text_file("test.txt", "/base", "content")
            assert False, "Should have raised IOError"  # nosec B101  # nosec B101
        except IOError:
            pass

    @patch("utils.file_ops.read_text_file")
    def test_read_json_file(self, mock_read):
        """Test reading JSON file."""
        from utils.file_ops import read_json_file

        mock_read.return_value = '{"key": "value"}'
        result = read_json_file("test.json", "/base")
        assert result == {"key": "value"}  # nosec B101

    @patch("utils.file_ops.read_text_file")
    def test_read_json_file_invalid(self, mock_read):
        """Test reading invalid JSON file."""
        from utils.file_ops import read_json_file

        mock_read.return_value = "invalid json"
        with pytest.raises(ValueError):
            read_json_file("test.json", "/base")

    @patch("utils.file_ops.write_text_file")
    def test_write_json_file(self, mock_write):
        """Test writing JSON file."""
        from utils.file_ops import write_json_file

        write_json_file("test.json", "/base", {"key": "value"})
        mock_write.assert_called_once()  # nosec B101

    @patch("utils.file_ops.validate_file_path")
    @patch("utils.file_ops._ops")
    def test_delete_file(self, mock_ops, mock_validate):
        """Test deleting file."""
        from utils.file_ops import delete_file

        mock_validate.return_value = Path("/valid/path")
        mock_ops.safe_delete.return_value = True

        delete_file("test.txt", "/base")
        mock_ops.safe_delete.assert_called_once()  # nosec B101

    @patch("utils.file_ops.validate_file_path")
    @patch("utils.file_ops._ops")
    def test_delete_file_error(self, mock_ops, mock_validate):
        """Test deleting file with error."""
        from utils.file_ops import delete_file

        mock_validate.return_value = Path("/valid/path")
        mock_ops.safe_delete.return_value = False

        with pytest.raises(IOError):
            delete_file("test.txt", "/base")

    @patch("utils.file_ops.validate_file_path")
    @patch("utils.file_ops._ops")
    def test_file_exists(self, mock_ops, mock_validate):
        """Test checking if file exists."""
        from utils.file_ops import file_exists

        mock_validate.return_value = Path("/valid/path")
        mock_ops.safe_exists.return_value = True

        result = file_exists("test.txt", "/base")
        assert result is True  # nosec B101  # nosec B101

    @patch("utils.file_ops.validate_file_path")
    def test_file_exists_invalid_path(self, mock_validate):
        """Test file_exists with invalid path."""
        from utils.file_ops import file_exists

        mock_validate.side_effect = ValueError("Invalid path")
        result = file_exists("test.txt", "/base")
        assert result is False  # nosec B101  # nosec B101

    @patch("utils.file_ops.validate_file_path")
    @patch("utils.file_ops._ops")
    def test_make_directory(self, mock_ops, mock_validate):
        """Test creating directory."""
        from utils.file_ops import make_directory

        mock_validate.return_value = Path("/valid/path")
        mock_ops.safe_mkdir.return_value = True

        make_directory("/dir", "/base")
        mock_ops.safe_mkdir.assert_called_once()  # nosec B101  # nosec B101

    @patch("utils.file_ops.validate_file_path")
    @patch("utils.file_ops._ops")
    def test_make_directory_error(self, mock_ops, mock_validate):
        """Test creating directory with error."""
        from utils.file_ops import make_directory

        mock_validate.return_value = Path("/valid/path")
        mock_ops.safe_mkdir.return_value = False

        with pytest.raises(IOError):
            make_directory("/dir", "/base")


class TestSecretPolicyEnforcer:
    """Test secret_policy_enforcer.py module."""

    @patch.dict("os.environ", {"APGI_MASTER_KEY": "test_key_1234567890123456"})
    @patch("utils.secret_policy_enforcer.get_secure_key_manager")
    def test_enforce_secret_policy_success(self, mock_manager):
        """Test secret policy enforcement with valid setup."""
        from utils.secret_policy_enforcer import enforce_secret_policy

        mock_mgr = MagicMock()
        mock_mgr.get_pickle_secret_key.return_value = "key"
        mock_mgr.get_backup_hmac_key.return_value = "key"
        mock_manager.return_value = mock_mgr

        # Should not raise
        enforce_secret_policy()

    @patch.dict("os.environ", {}, clear=True)
    def test_enforce_secret_policy_no_master_key(self):
        """Test secret policy enforcement without master key."""
        from utils.secret_policy_enforcer import enforce_secret_policy

        try:
            enforce_secret_policy()
            assert False, "Should have exited"  # nosec B101
        except SystemExit:
            pass

    @patch.dict("os.environ", {"APGI_MASTER_KEY": "test_key"})
    @patch("utils.secret_policy_enforcer.get_secure_key_manager")
    def test_enforce_secret_policy_key_load_failure(self, mock_manager):
        """Test secret policy enforcement with key load failure."""
        from utils.secret_policy_enforcer import enforce_secret_policy

        mock_manager.side_effect = Exception("Key load failed")

        with pytest.raises(SystemExit):
            enforce_secret_policy()

    @patch.dict(
        "os.environ",
        {
            "APGI_MASTER_KEY": "test_key_1234567890123456",
            "APGI_JWT_SECRET": "test_weak_secret",  # nosec B105
        },
    )
    @patch("utils.secret_policy_enforcer.get_secure_key_manager")
    def test_enforce_secret_policy_weak_plaintext(self, mock_manager):
        """Test secret policy enforcement with weak plaintext."""
        from utils.secret_policy_enforcer import enforce_secret_policy

        mock_mgr = MagicMock()
        mock_mgr.get_pickle_secret_key.return_value = "key"
        mock_mgr.get_backup_hmac_key.return_value = "key"
        mock_manager.return_value = mock_mgr

        with pytest.raises(SystemExit):
            enforce_secret_policy()

    @patch.dict(
        "os.environ",
        {
            "APGI_MASTER_KEY": "test_key_1234567890123456",
            "APGI_JWT_SECRET": "secret",  # nosec B105 - forbidden value
        },
    )
    @patch("utils.secret_policy_enforcer.get_secure_key_manager")
    def test_enforce_secret_policy_fallback_detected(self, mock_manager):
        """Test secret policy enforcement with fallback secret."""
        from utils.secret_policy_enforcer import enforce_secret_policy

        mock_mgr = MagicMock()
        mock_mgr.get_pickle_secret_key.return_value = "key"
        mock_mgr.get_backup_hmac_key.return_value = "key"
        mock_manager.return_value = mock_mgr

        with pytest.raises(SystemExit):
            enforce_secret_policy()
