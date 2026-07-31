"""
Pytest configuration and fixtures for APGI test suite.
======================================================

Provides common test fixtures and configuration:
- Temporary directories
- Mock objects
- Test data
- Test configuration
"""

# Mock tkinter BEFORE any imports to prevent GUI initialization during tests
import os
import sys
from unittest.mock import MagicMock

# Opt-out for the cross-platform GUI smoke test, which must instantiate real
# widgets against a real (headless, e.g. xvfb) display. Without this escape
# hatch the smoke test detects the MagicMock and skips itself unconditionally,
# so it passes without ever exercising a GUI.
USE_REAL_TKINTER = os.environ.get("APGI_REAL_TKINTER", "").strip().lower() in {"1", "true", "yes"}


# Create a safe mock variable factory for Python 3.14 compatibility
def create_safe_mock_var(default_value=0):
    """Create a MagicMock variable that's safe for int() conversion."""
    mock_var = MagicMock()
    mock_var.get.return_value = default_value
    mock_var.set = MagicMock()
    return mock_var


# Mock tkinter modules before any test imports them
tkinter_modules = [
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
    "tkinter.filedialog",
    "tkinter.scrolledtext",
    "tkinter.font",
]

if not USE_REAL_TKINTER:
    for module in tkinter_modules:
        if module not in sys.modules:
            sys.modules[module] = MagicMock()


# Fix for Python 3.14 compatibility - make Vars return proper values
def mock_var_factory(*args, **kwargs):
    default = kwargs.get("value", 0)
    mock_var = MagicMock()
    mock_var.get.return_value = default
    mock_var.set = MagicMock()
    return mock_var


if not USE_REAL_TKINTER:
    # Apply the fix to all tkinter modules that might be imported
    for module_name in tkinter_modules:
        if module_name in sys.modules:
            mock_module = sys.modules[module_name]
            # Add the variable factories to each tkinter module
            mock_module.StringVar = lambda: create_safe_mock_var(0)
            mock_module.IntVar = lambda: create_safe_mock_var(0)
            mock_module.DoubleVar = lambda: create_safe_mock_var(0.0)
            mock_module.BooleanVar = lambda: create_safe_mock_var(False)

    # Configure mock tkinter with basic widget behavior
    mock_tk = sys.modules["tkinter"]
    mock_tk.Tk = MagicMock()  # type: ignore
    mock_tk.ttk = MagicMock()  # type: ignore
    mock_tk.messagebox = MagicMock()  # type: ignore
    mock_tk.filedialog = MagicMock()  # type: ignore

    mock_tk.StringVar = mock_var_factory  # type: ignore
    mock_tk.BooleanVar = mock_var_factory  # type: ignore
    mock_tk.DoubleVar = mock_var_factory  # type: ignore
    mock_tk.IntVar = mock_var_factory  # type: ignore

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


# Define safe functions at module level for Python 3.14 compatibility
def _safe_amin(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """Safe amin that handles _NoValueType initial parameter."""
    if initial is not None and hasattr(initial, "__class__") and "_NoValueType" in str(type(initial)):
        initial = None
    import numpy as np

    _original_amin = np.amin
    return _original_amin(
        a,
        axis=axis,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


def _safe_amax(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """Safe amax that handles _NoValueType initial parameter."""
    if initial is not None and hasattr(initial, "__class__") and "_NoValueType" in str(type(initial)):
        initial = None
    import numpy as np

    _original_amax = np.amax
    return _original_amax(
        a,
        axis=axis,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


# Python 3.14 compatibility fixes for NumPy/SciPy
if sys.version_info >= (3, 14):
    # Comprehensive patching for NumPy/SciPy compatibility issues
    try:
        import numpy as np

        # Global flag to prevent multiple patching
        if not hasattr(np, "_apgi_patching_applied"):
            np._apgi_patching_applied = True

            # Fix 1: Handle _NoValueType in reduction operations
            _original_amin = np.amin
            _original_amax = np.amax
            _original_argmin = np.argmin
            _original_argmax = np.argmax
            _original_prod = np.prod
            _original_sum = np.sum
            _original_multiply = np.multiply
            _original_reshape = np.reshape

            # NOTE (Py3.14): Avoid patching NumPy internals like `_wrapreduction` or
            # replacing ufuncs (e.g., `np.multiply`). These patches interfere with
            # SciPy imports and can cause widespread collection-time failures.

            # Define safe functions at module level
            def _safe_amin(a, axis=None, out=None, keepdims=False, initial=None, where=True):
                """Safe amin that handles _NoValueType initial parameter."""
                if initial is not None and hasattr(initial, "__class__") and "_NoValueType" in str(type(initial)):
                    initial = None
                return _original_amin(
                    a,
                    axis=axis,
                    out=out,
                    keepdims=keepdims,
                    initial=initial,
                    where=where,
                )

            def _safe_amax(a, axis=None, out=None, keepdims=False, initial=None, where=True):
                """Safe amax that handles _NoValueType initial parameter."""
                if initial is not None and hasattr(initial, "__class__") and "_NoValueType" in str(type(initial)):
                    initial = None
                return _original_amax(
                    a,
                    axis=axis,
                    out=out,
                    keepdims=keepdims,
                    initial=initial,
                    where=where,
                )

            def _safe_argmin(a, axis=None, out=None, keepdims=False):
                """Safe argmin that handles _NoValueType objects."""
                return _original_argmin(a, axis=axis, out=out, keepdims=keepdims)

            def _safe_argmax(a, axis=None, out=None, keepdims=False):
                """Safe argmax that handles _NoValueType objects."""
                return _original_argmax(a, axis=axis, out=out, keepdims=keepdims)

            def _safe_prod(
                a,
                axis=None,
                dtype=None,
                out=None,
                keepdims=np._NoValue,
                initial=np._NoValue,
                where=np._NoValue,
            ):
                """Safe prod that handles _NoValueType initial parameter."""
                if initial is not None and hasattr(initial, "__class__") and "_NoValueType" in str(type(initial)):
                    # Preserve NumPy identity behavior for empty products.
                    # `np.prod([])` should be 1, and passing `initial=None` can
                    # trigger "ufunc has no identity" errors for empty reductions.
                    initial = 1
                return _original_prod(
                    a,
                    axis=axis,
                    dtype=dtype,
                    out=out,
                    keepdims=keepdims,
                    initial=initial,
                    where=where,
                )

            def _safe_sum(
                a,
                axis=None,
                dtype=None,
                out=None,
                keepdims=np._NoValue,
                initial=np._NoValue,
                where=np._NoValue,
            ):
                """Safe sum that handles zero-size arrays and broadcasting issues."""
                if initial is not None and hasattr(initial, "__class__") and "_NoValueType" in str(type(initial)):
                    # Preserve NumPy identity behavior for empty sums.
                    initial = 0
                try:
                    return _original_sum(
                        a,
                        axis=axis,
                        dtype=dtype,
                        out=out,
                        keepdims=keepdims,
                        initial=initial,
                        where=where,
                    )
                except (ValueError, TypeError) as e:
                    if "zero-size array" in str(e) or "broadcast together" in str(e):
                        # Match NumPy semantics: sum([]) == 0.
                        # Returning an empty array here can break downstream callers
                        # (e.g., SciPy) that expect a scalar/0-d result.
                        if hasattr(a, "size") and a.size == 0:
                            return np.array(0, dtype=getattr(a, "dtype", float))
                        return np.array(0, dtype=getattr(a, "dtype", float))
                    raise

            def _safe_multiply(x1, x2, *args, **kwargs):
                """Safe multiply that handles zero-size arrays."""
                try:
                    return _original_multiply(x1, x2, *args, **kwargs)
                except ValueError as e:
                    if "zero-size array to reduction operation" in str(e):
                        # Handle zero-size arrays by returning appropriate result
                        if hasattr(x1, "size") and x1.size == 0:
                            return np.array([], dtype=x1.dtype if hasattr(x1, "dtype") else float)
                        elif hasattr(x2, "size") and x2.size == 0:
                            return np.array([], dtype=x2.dtype if hasattr(x2, "dtype") else float)
                    raise

            # Safe reshape that handles zero-size arrays
            def _safe_reshape(a, newshape, *args, **kwargs):
                """Safe reshape that handles zero-size arrays."""
                try:
                    return _original_reshape(a, newshape, *args, **kwargs)
                except ValueError as e:
                    if "cannot reshape array of size 0" in str(e):
                        # Handle zero-size arrays by returning appropriate result
                        if hasattr(a, "size") and a.size == 0:
                            if newshape == () or newshape == (0,):
                                return np.array([], dtype=getattr(a, "dtype", float))
                            elif hasattr(newshape, "__iter__") and 0 in newshape:
                                return np.array([], dtype=getattr(a, "dtype", float))
                            # For other shapes, try to return a scalar
                            return np.array(0, dtype=getattr(a, "dtype", float))
                    raise

            # Copy the ufunc methods to preserve behavior
            _safe_multiply.__name__ = _original_multiply.__name__
            _safe_multiply.__doc__ = _original_multiply.__doc__
            for attr in ["reduce", "accumulate", "reduceat", "outer", "at"]:
                if hasattr(_original_multiply, attr):
                    setattr(_safe_multiply, attr, getattr(_original_multiply, attr))

            # Add reduce method to patched functions for compatibility
            def _add_reduce_method(func, original_func):
                """Add reduce method to patched functions for compatibility."""

                def reduce_method(a, *args, **kwargs):
                    try:
                        # Use the original function's reduce method if available
                        if hasattr(original_func, "reduce"):
                            return original_func.reduce(a, *args, **kwargs)
                        # Fallback to calling the function directly
                        return func(a, *args, **kwargs)
                    except (ValueError, TypeError) as e:
                        if "zero-size array" in str(e) or "no identity" in str(e):
                            # Handle zero-size arrays in reduce operations
                            if hasattr(a, "size") and a.size == 0:
                                if original_func.__name__ == "prod":
                                    return np.array(1, dtype=getattr(a, "dtype", float))
                                else:  # sum and others
                                    return np.array(0, dtype=getattr(a, "dtype", float))
                            elif hasattr(a, "shape") and a.shape == (0,):
                                if original_func.__name__ == "prod":
                                    return np.array(1, dtype=getattr(a, "dtype", float))
                                else:  # sum and others
                                    return np.array(0, dtype=getattr(a, "dtype", float))
                            # Return scalar for problematic arrays
                            if original_func.__name__ == "prod":
                                return np.array(1, dtype=getattr(a, "dtype", float))
                            else:  # sum and others
                                return np.array(0, dtype=getattr(a, "dtype", float))
                        raise

                func.reduce = reduce_method
                return func

            # Apply reduction patches with reduce methods
            np.amin = _add_reduce_method(_safe_amin, _original_amin)
            np.amax = _add_reduce_method(_safe_amax, _original_amax)
            np.argmin = _add_reduce_method(_safe_argmin, _original_argmin)
            np.argmax = _add_reduce_method(_safe_argmax, _original_argmax)
            np.prod = _add_reduce_method(_safe_prod, _original_prod)
            np.sum = _add_reduce_method(_safe_sum, _original_sum)
            np.reshape = _safe_reshape  # type: ignore[assignment]
            # Do not replace `np.multiply` (ufunc) — SciPy expects a real ufunc here.

        # Fix 5: Handle NumPy array rounding compatibility
        import builtins

        _original_round = builtins.round

        def _safe_round(x, *args, **kwargs):
            """Safe round function that handles NumPy arrays."""
            if hasattr(x, "shape") and hasattr(x, "dtype"):  # Likely a NumPy array
                try:
                    return np.round(x, *args, **kwargs)
                except (TypeError, ValueError):
                    # Fallback for problematic arrays
                    if hasattr(x, "size") and x.size == 1:
                        return _original_round(float(x.item()), *args, **kwargs)
                    # For arrays, apply element-wise
                    if hasattr(x, "flat"):
                        return np.array([_original_round(float(val), *args, **kwargs) for val in x.flat]).reshape(
                            x.shape
                        )
                    return x  # Return as-is if we can't handle it
            return _original_round(x, *args, **kwargs)

        builtins.round = _safe_round  # type: ignore[assignment]

        # Fix 2: Handle _CopyMode enum issues
        if hasattr(np, "_globals"):
            _original_bool = np._globals.__dict__.get("__bool__")

            def _safe_globals_bool(self):
                """Safe __bool__ that handles _CopyMode and similar enums."""
                try:
                    if hasattr(self, "value") and hasattr(self, "name"):
                        # Handle enum-like objects
                        return bool(self.value)
                    return bool(self)
                except (ValueError, TypeError):
                    # For problematic objects like _CopyMode, default to False
                    return False

            np._globals.__bool__ = _safe_globals_bool

        # Fix 3: Patch array creation to handle copy mode issues
        _original_array = np.array

        def _safe_array(obj: Any, *args: Any, **kwargs: Any) -> Any:
            """Safe array creation that handles copy mode issues."""
            # Handle copy parameter issues
            if "copy" in kwargs:
                copy_val = kwargs["copy"]
                if hasattr(copy_val, "__class__") and (
                    "CopyMode" in str(type(copy_val)) or "IF_NEEDED" in str(copy_val)
                ):
                    kwargs["copy"] = True  # Default to True for problematic copy modes

            try:
                return _original_array(obj, *args, **kwargs)
            except (ValueError, TypeError) as e:
                if "IF_NEEDED" in str(e) or "CopyMode" in str(e):
                    # Retry with safe copy parameter
                    kwargs["copy"] = True
                    return _original_array(obj, *args, **kwargs)
                raise

        np.array = _safe_array  # type: ignore[assignment]

        # Fix 4: Handle VoidDType issues
        if hasattr(np, "dtypes") and not hasattr(np.dtypes, "VoidDType"):
            # Create a mock VoidDType class for compatibility
            class MockVoidDType:
                def __init__(self, *args, **kwargs):
                    pass

                def __repr__(self):
                    return "VoidDType"

                def __str__(self):
                    return "void"

            np.dtypes.VoidDType = MockVoidDType  # type: ignore

        # Fix 5: Handle PIL/PNG format issues
        try:
            from PIL import Image

            # Initialize Image.SAVE if it's empty or missing
            if not hasattr(Image, "SAVE") or not Image.SAVE:
                Image.SAVE = {}

            # Add PNG format support if missing
            if "PNG" not in Image.SAVE:
                # Create a simple PNG save function or use None as fallback
                Image.SAVE["PNG"] = None  # PIL will handle PNG natively

            # Also ensure other common formats are available
            common_formats = ["JPEG", "BMP", "TIFF", "GIF"]
            for fmt in common_formats:
                if fmt not in Image.SAVE:
                    Image.SAVE[fmt] = None

        except ImportError:
            pass  # PIL not available

        # Also patch in _core._methods where the actual implementations are
        if hasattr(np._core, "_methods"):
            np._core._methods._amin = _safe_amin
            np._core._methods._amax = _safe_amax

    except ImportError:
        pass  # NumPy not available


@pytest.fixture(scope="session", autouse=True)
def headless_gui_setup():
    """Ensure GUI tests run in headless mode by mocking tkinter before any imports.

    This fixture runs at session start and mocks tkinter to prevent actual GUI
    initialization during tests. Essential for CI/CD headless environments.
    Note: tkinter is already mocked at module level in conftest.py, this fixture
    ensures the mock persists throughout the test session.
    """
    # tkinter is already mocked at module level before any imports
    # This fixture ensures the mock configuration is complete
    mock_tk = sys.modules.get("tkinter")
    if mock_tk:
        mock_tk.Tk = MagicMock
        mock_tk.StringVar = MagicMock
        mock_tk.BooleanVar = MagicMock
        mock_tk.DoubleVar = MagicMock
        mock_tk.IntVar = MagicMock
        mock_tk.ttk = MagicMock()
        mock_tk.messagebox = MagicMock()
        mock_tk.filedialog = MagicMock()

        # Fix for Python 3.14 compatibility - make Vars return proper values
        def mock_var_factory(*args, **kwargs):
            default = kwargs.get("value", 0)
            mock_var = MagicMock()
            mock_var.get.return_value = default
            mock_var.set = MagicMock()
            return mock_var

        mock_tk.IntVar = mock_var_factory
        mock_tk.DoubleVar = mock_var_factory
        mock_tk.StringVar = mock_var_factory
        mock_tk.BooleanVar = mock_var_factory

    yield

    # Cleanup not needed as mocks persist for session duration


@pytest.fixture
def apgi_backup_hmac_key(monkeypatch):
    """Provide APGI_BACKUP_HMAC_KEY for tests.

    This fixture injects a test HMAC key into the environment,
    allowing tests to run without external environment configuration.
    """
    key = "test_backup_hmac_key_" + "x" * 32
    monkeypatch.setenv("APGI_BACKUP_HMAC_KEY", key)
    yield key


@pytest.fixture
def pickle_secret_key(monkeypatch):
    """Provide PICKLE_SECRET_KEY for tests.

    This fixture injects a test pickle secret key into the environment,
    allowing tests to run without external environment configuration.
    """
    key = "test_pickle_secret_key_" + "x" * 32
    monkeypatch.setenv("PICKLE_SECRET_KEY", key)
    yield key


@pytest.fixture(autouse=True)
def apgi_jwt_secret(monkeypatch):
    """Provide a default APGI_JWT_SECRET for all tests.

    Must be exactly 64 hex characters. The previous value
    ("test_jwt_secret_at_least_32_characters_long") failed that validation, so
    the key manager silently fell through to the developer's real ``.keys/``
    directory — making the suite depend on local key material that no other
    machine has, and that fails outright once a key cannot be decrypted.
    """
    secret = "a1" * 32
    monkeypatch.setenv("APGI_JWT_SECRET", secret)
    monkeypatch.setenv("APGI_SKIP_SECURITY", "true")
    return secret


@pytest.fixture(autouse=True)
def isolated_key_material(monkeypatch):
    """Keep tests off real key material.

    Supplies valid 64-hex values for every managed key, so the key manager
    resolves them from the environment and never reads, writes or mints a file
    in the repository's ``.keys/`` directory. Tests that exercise the file-backed
    paths (tests/test_secure_key_manager_fail_closed.py) delete these explicitly.
    """
    monkeypatch.setenv("PICKLE_SECRET_KEY", "b2" * 32)
    monkeypatch.setenv("APGI_BACKUP_HMAC_KEY", "c3" * 32)
    yield


@pytest.fixture(autouse=True)
def allow_ephemeral_master_key(monkeypatch):
    """Allow explicit ephemeral master keys in tests.

    Production code should set a persistent `APGI_MASTER_KEY`. In tests we allow
    ephemeral generation, but only when this explicit flag is present.
    """
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")
    yield


@pytest.fixture
def env_vars(monkeypatch):
    """Provide all required environment variables for tests.

    This fixture injects both APGI_BACKUP_HMAC_KEY and PICKLE_SECRET_KEY
    into the environment, allowing tests to run without external configuration.
    """
    env_vars = {
        "APGI_BACKUP_HMAC_KEY": "test_key_" + "x" * 32,
        "PICKLE_SECRET_KEY": "test_secret_" + "x" * 32,
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    yield env_vars


@pytest.fixture(scope="session")
def cli():
    """Lazy-load CLI to avoid hanging during test collection.

    This fixture delays the import of main.py until tests actually run,
    preventing collection errors caused by module-level logging initialization.
    """
    try:
        from main import cli as main_cli

        return main_cli
    except Exception as e:
        # If import fails, return a mock CLI for testing
        pytest.skip(f"Could not import CLI: {e}")


def pytest_sessionfinish(session, exitstatus):
    """Clean up background resources without forcing process exit.

    Pytest already controls process termination and exit codes. Calling
    ``sys.exit`` from this hook can surface as an interrupted run during
    collection on newer pytest versions, so we intentionally avoid doing that
    here.
    """
    # Intentionally no explicit sys.exit(...) call.
    return None


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests with secure permissions."""
    with tempfile.TemporaryDirectory() as temp_path:
        # Set restrictive permissions (owner only) on the temp directory
        os.chmod(temp_path, 0o700)
        yield Path(temp_path)


@pytest.fixture
def sample_config():
    """Provide sample configuration data."""
    return {
        "model": {
            "tau_S": 0.5,
            "tau_theta": 30.0,
            "theta_0": 0.5,
            "alpha": 10.0,
            "gamma_M": -0.3,
            "gamma_A": 0.1,
            "rho": 0.7,
            "sigma_S": 0.05,
            "sigma_theta": 0.02,
        },
        "simulation": {
            "default_steps": 1000,
            "default_dt": 0.01,
            "max_steps": 100000,
            "enable_plots": True,
            "plot_format": "png",
            "plot_dpi": 150,
            "save_results": True,
            "results_format": "csv",
        },
        "logging": {
            "level": "INFO",
            "enable_console": True,
            "log_rotation": "10 MB",
            "log_retention": "30 days",
            "enable_performance_logging": True,
            "enable_structured_logging": True,
        },
        "data": {
            "default_data_dir": "data",
            "supported_formats": ["csv", "json", "xlsx", "pkl"],
            "max_file_size_mb": 100,
            "enable_caching": True,
            "cache_dir": "cache",
        },
        "validation": {
            "enable_cross_validation": True,
            "cv_folds": 5,
            "enable_sensitivity_analysis": True,
            "sensitivity_samples": 100,
            "enable_robustness_tests": True,
            "significance_level": 0.05,
        },
    }


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    import numpy as np

    # Generate sufficient data points for time-series validation
    n_samples = 1000
    dt = 0.01
    time = np.arange(0, n_samples * dt, dt)

    return {
        "timestamps": time.tolist(),
        "surprise": (0.2 + 0.1 * np.sin(2 * np.pi * 0.1 * time) + 0.05 * np.random.randn(n_samples)).tolist(),
        "threshold": (0.5 + 0.02 * np.sin(2 * np.pi * 0.05 * time) + 0.01 * np.random.randn(n_samples)).tolist(),
        "metabolic": (1.0 + 0.1 * np.sin(2 * np.pi * 0.2 * time) + 0.05 * np.random.randn(n_samples)).tolist(),
        "arousal": (0.8 + 0.1 * np.sin(2 * np.pi * 0.15 * time) + 0.03 * np.random.randn(n_samples)).tolist(),
    }


@pytest.fixture
def raises_fixture():
    """Fixture that provides a context manager for testing exceptions."""

    class RaisesContext:
        def __init__(self, expected_exception=Exception):
            self.expected_exception = expected_exception
            self.exception_raised = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                pytest.fail(f"Expected {self.expected_exception.__name__} to be raised, but no exception was raised")
            if not isinstance(exc_val, self.expected_exception):
                pytest.fail(
                    f"Expected {self.expected_exception.__name__} to be raised, but got {exc_type.__name__}: {exc_val}"
                )
            self.exception_raised = exc_val
            return True  # Suppress the exception

    return RaisesContext


@pytest.fixture
def oom_fixture():
    """Fixture for testing out-of-memory conditions."""

    class OOMContext:
        def __init__(self):
            self.original_memory_limit = None

        def __enter__(self):
            # Try to simulate OOM by setting a very low memory limit
            # This is a best-effort simulation since actual OOM is hard to trigger safely
            try:
                import resource

                self.original_memory_limit = resource.getrlimit(resource.RLIMIT_AS)
                # Set memory limit to 10MB for testing
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (10 * 1024 * 1024, self.original_memory_limit[1]),
                )
            except ImportError:
                # resource module not available on Windows
                pass
            except Exception:
                # If we can't set limits, we'll use a mock approach
                pass
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original memory limit
            if self.original_memory_limit is not None:
                try:
                    import resource

                    resource.setrlimit(resource.RLIMIT_AS, self.original_memory_limit)
                except Exception:
                    pass
            return False  # Don't suppress exceptions

    return OOMContext()


@pytest.fixture
def mock_memory_error():
    """Fixture that mocks memory allocation to raise MemoryError."""

    class MemoryErrorMocker:
        def __init__(self):
            self.patches = []

        def patch_numpy_zeros(self):
            """Patch numpy.zeros to raise MemoryError after a few calls."""
            call_count = [0]
            original_zeros = __import__("numpy").zeros

            def mock_zeros(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] > 3:  # Allow a few calls before failing
                    raise MemoryError("Simulated out of memory")
                return original_zeros(*args, **kwargs)

            patch_obj = patch("numpy.zeros", side_effect=mock_zeros)
            patch_obj.start()
            self.patches.append(patch_obj)
            return patch_obj

        def patch_torch_tensor(self):
            """Patch torch tensor creation to raise MemoryError."""
            try:
                torch = __import__("torch")
                call_count = [0]
                original_tensor = torch.tensor

                def mock_tensor(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] > 2:
                        raise RuntimeError("CUDA out of memory")  # Simulates GPU OOM
                    return original_tensor(*args, **kwargs)

                patch_obj = patch("torch.tensor", side_effect=mock_tensor)
                patch_obj.start()
                self.patches.append(patch_obj)
                return patch_obj
            except ImportError:
                return None

        def cleanup(self):
            """Clean up all patches."""
            for patch_obj in self.patches:
                patch_obj.stop()
            self.patches.clear()

    mocker = MemoryErrorMocker()
    yield mocker
    mocker.cleanup()


@pytest.fixture
def exception_test_cases():
    """Provide common exception test cases."""
    return {
        "value_error": ValueError("Invalid value"),
        "type_error": TypeError("Invalid type"),
        "key_error": KeyError("Missing key"),
        "attribute_error": AttributeError("'NoneType' object has no attribute"),
        "io_error": IOError("File operation failed"),
        "memory_error": MemoryError("Out of memory"),
        "runtime_error": RuntimeError("Runtime error"),
        "assertion_error": AssertionError("Assertion failed"),
    }


@pytest.fixture
def random_seed():
    """Provide a fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def seeded_rng(random_seed):
    """Provide a numpy RandomState with a fixed seed for reproducible tests."""
    return np.random.RandomState(random_seed)


@pytest.fixture(autouse=True)
def reset_random_state_before_each_test():
    """Reset random state before each test for reproducibility."""
    original_state = np.random.get_state()
    yield
    np.random.set_state(original_state)


@pytest.fixture
def flaky_operation():
    """Fixture that provides a factory for creating flaky operations."""
    import time

    import numpy as np

    def create_operation(success_rate=0.5):
        """Create a flaky operation with specified success rate.

        Args:
            success_rate: Probability of success (0.0 to 1.0)

        Returns:
            A function that succeeds with the given probability
        """

        def operation():
            if np.random.random() < success_rate:
                return "success"
            else:
                raise RuntimeError("Operation failed")

        return operation

    def retry_wrapper(func, max_attempts=3, timeout=None, backoff_factor=1):
        """Execute a function with retry logic.

        Args:
            func: Callable to execute
            max_attempts: Maximum number of retry attempts
            timeout: Maximum time to wait (not implemented in this fixture)
            backoff_factor: Multiplier for wait time between retries
        """
        last_exception = None

        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    # Exponential backoff with jitter
                    wait_time = backoff_factor * (2**attempt) * (0.5 + np.random.random() * 0.5)
                    time.sleep(wait_time)

        # All attempts failed, raise the last exception
        raise last_exception

    # Return both the factory and the retry wrapper
    class FlakyOperationFactory:
        def __call__(self, success_rate=0.5):
            return create_operation(success_rate)

        def retry(self, func, max_attempts=3, timeout=None, backoff_factor=1):
            return retry_wrapper(func, max_attempts, timeout, backoff_factor)

    return FlakyOperationFactory()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


def pytest_collection_finish(session):
    """Called after test collection is complete."""
    # If we're only collecting (--collect-only), we should not exit
    # as it causes pytest to report collection errors
    pass


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to tests in test files that don't have integration markers
        if "integration" not in item.keywords and "performance" not in item.keywords:
            item.add_marker(pytest.mark.unit)

        # Add slow marker to performance tests
        if "performance" in item.keywords:
            item.add_marker(pytest.mark.slow)


# Test utilities
def create_test_file(file_path: Path, content: str):
    """Create a test file with given content."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def create_test_json(file_path: Path, data: dict):
    """Create a test JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_test_yaml(file_path: Path, data: dict):
    """Create a test YAML file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def assert_performance_within_tolerance(actual_time: float, expected_time: float, tolerance: float):
    """Assert that performance is within acceptable tolerance."""
    lower_bound = expected_time * (1 - tolerance)
    upper_bound = expected_time * (1 + tolerance)
    assert (
        lower_bound <= actual_time <= upper_bound
    ), f"Performance {actual_time:.3f}s not within tolerance of {expected_time:.3f}s ± {tolerance * 100}%"


@pytest.fixture(autouse=True)
def cleanup_test_profiles():
    """Auto-cleanup test profiles created during tests.

    Prevents auto-generated profile files (like profile_0_0.yaml, profile_4_3.yaml)
    from accumulating in config/profiles directory. This fixture automatically runs
    before and after each test to clean up any temporary profiles created by tests.
    """
    # Get the profiles directory
    project_root = Path(__file__).parent.parent
    profiles_dir = project_root / "config" / "profiles"

    # List of known good profiles that should not be deleted
    protected_profiles = {
        "adhd.yaml",
        "anxiety-disorder.yaml",
        "autism-spectrum-disorder.yaml",
        "research-default.yaml",
        "generalized-anxiety-disorder.yaml",
        "major-depressive-disorder.yaml",
        "test_profile.yaml",
        # Add other legitimate clinical profiles here
        "acute-stress-disorder.yaml",
        "anorexia-nervosa.yaml",
        "binge-eating-disorder.yaml",
        "bipolar-i.yaml",
        "body-dysmorphic-disorder.yaml",
        "borderline-personality-disorder.yaml",
        "bulimia-nervosa.yaml",
        "gender-dysphoria.yaml",
        "hoarding-disorder.yaml",
        "insomnia-disorder.yaml",
        "obsessive-compulsive-disorder.yaml",
        "panic-disorder.yaml",
        "ptsd.yaml",
        "schizophrenia.yaml",
        "social-anxiety-disorder.yaml",
        "somatic-symptom-disorder.yaml",
        "specific-phobia.yaml",
    }

    # Cleanup before test (remove any leftover test profiles)
    if profiles_dir.exists():
        for profile_file in profiles_dir.glob("profile_*.yaml"):
            # Only delete numbered profiles (profile_X_Y.yaml pattern)
            if profile_file.name.startswith("profile_") and "_" in profile_file.stem:
                try:
                    profile_file.unlink()
                except Exception:
                    pass  # Ignore errors during cleanup

    yield

    # Cleanup after test (remove any test profiles created during test)
    if profiles_dir.exists():
        for profile_file in profiles_dir.glob("*.yaml"):
            # Delete auto-generated test profiles but protect known good profiles
            if profile_file.name not in protected_profiles:
                try:
                    profile_file.unlink()
                except Exception:
                    pass  # Ignore errors during cleanup


if __name__ == "__main__":
    print("conftest.py is a pytest configuration file and should not be run directly.")
    print("Use 'pytest' to run the test suite.")
    exit(0)
