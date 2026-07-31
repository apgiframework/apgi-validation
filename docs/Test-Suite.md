# APGI Comprehensive Testing Framework

## Features

- **Adversarial Test Generation**: Targets edge cases and rarely-executed code paths
- **Mutation Testing**: Verifies test effectiveness by introducing artificial bugs
- **Performance & Stress Testing**: Tests under high load, memory pressure, and resource exhaustion
- **Security Testing**: Validates input sanitization, injection resistance, and data protection
- **Integration & E2E Testing**: Tests complete protocol pipelines and real-world workflows
- **Comprehensive Coverage Reporting**: Line, branch, and path coverage analysis
- **Deterministic Reproducibility**: Controlled seeding for consistent results

## Installation

```bash
# Install dependencies
pip install pytest pytest-cov hypothesis psutil numpy scipy pandas

# For mutation testing (optional)
pip install mutmut
```

## Usage

### Run All Tests

```bash
python3 -m tests.comprehensive.runner --all
```

### Run Specific Test Categories

```bash
# Unit tests only
python3 -m tests.comprehensive.runner --category unit

# Security tests
python3 -m tests.comprehensive.runner --category security

# Performance tests
python3 -m tests.comprehensive.runner --category performance

# Mutation testing
python3 -m tests.comprehensive.runner --mutation
```

### Run Individual Test Modules

```bash
# Mutation testing
python3 -m tests.comprehensive.mutation_tester

# Performance/stress testing
python3 -m tests.comprehensive.stress_test

# Security testing
python3 -m tests.comprehensive.security_tester

# Integration/E2E testing
python3 -m tests.comprehensive.integration_e2e
```

### Advanced Options

```bash
# Enforce coverage threshold (fails if below 95%)
python3 -m tests.comprehensive.runner --all --coverage-threshold 95

# Run with 8 parallel workers
python3 -m tests.comprehensive.runner --all --parallel 8

# Use custom random seed
python3 -m tests.comprehensive.runner --all --seed 12345

# Stop on first failure
python3 -m tests.comprehensive.runner --all --fail-fast

# Save reports to custom directory
python3 -m tests.comprehensive.runner --all --output ./my-reports
```

### Quick Test Run

```bash
# Run all tests with basic output
make test

# Run specific test file
python3 -m pytest tests/test_basic.py -v

# Run with coverage
python3 -m pytest tests/ --cov=. --cov-report=term
```

### Full Test Suite

The coverage scope and gate are configured in `pytest.ini`, so a plain
`pytest` run enforces them. Prefer the Make targets:

```bash
make test
```

```bash
make test-slow
```

To produce all coverage reports explicitly:

```bash
python3 -m pytest tests/ --cov-report=html --cov-report=xml --cov-report=term-missing
```

The gate is a **ratchet**: raise `--cov-fail-under` in `pytest.ini` as coverage
improves, never lower it to make a run pass.

```bash
# Run only unit tests
python3 -m pytest tests/ -m unit

# Run only integration tests
python3 -m pytest tests/ -m integration

# Run only slow tests
python3 -m pytest tests/ -m slow

# Skip slow tests
python3 -m pytest tests/ -m "not slow"
```

## Test Categories

### 1. Unit Tests (`__init__.py`)

Core testing framework providing:

- Test discovery and categorization
- Coverage analysis (line, branch, path)
- Report generation (JSON, HTML, Markdown)
- Deterministic reproducibility via seeding

### 2. Mutation Testing (`mutation_tester.py`)

Validates test effectiveness through:

- **Arithmetic mutations**: `+ → -`, `* → /`, etc.
- **Comparison mutations**: `> → >=`, `== → !=`, etc.
- **Boundary mutations**: Constant modification
- **Logical mutations**: `and → or`, `not` removal

**Output**: Mutation score (target: >80%)

### 3. Performance Testing (`stress_test.py`)

Tests system behavior under extreme conditions:

- **Concurrency testing**: Thread pools with 10-200 workers
- **Memory pressure**: Allocations up to 2GB
- **CPU stress**: Process pools utilizing all cores
- **I/O bottlenecks**: High-volume file operations
- **Resource exhaustion**: File descriptor limits
- **Regression detection**: Performance degradation monitoring

**Output**: Performance metrics, throughput analysis, resource usage

### 4. Security Testing (`security_tester.py`)

Validates security posture:

- **Input validation**: Type checking, boundary validation
- **Injection resistance**: SQL, command, code injection payloads
- **Path traversal**: Directory traversal prevention
- **XSS prevention**: Output sanitization
- **File operations**: Secure file handling
- **Environment variables**: Sensitive data protection
- **Logging safety**: Data masking verification

**Output**: Vulnerability report by severity (critical/high/medium/low)

### 5. Integration & E2E Testing (`integration_e2e.py`)

Tests component interactions and complete workflows:

- **Protocol → Aggregator flow**: Data pipeline validation
- **Schema integration**: ProtocolResult compatibility
- **Cross-protocol consistency**: Shared constants validation
- **Data pipeline**: Processing chain verification
- **Database transactions**: Persistence layer testing
- **Full FP pipeline**: 12 falsification protocols
- **Full VP pipeline**: 15 validation protocols
- **Condition A evaluation**: Framework falsification check
- **Condition B evaluation**: BIC model comparison
- **Real-world workflow**: Complete research pipeline

**Output**: Component interaction maps, workflow checkpoints

## Report Outputs

All test runs generate comprehensive reports:

### JSON Reports

Structured data for programmatic analysis:

```bash
reports/comprehensive_report.json
reports/mutation_report.json
reports/performance_report.json
reports/security_report.json
reports/integration_report.json
```

### HTML Reports

Interactive visual reports with:

- Test summaries with pass/fail indicators
- Coverage visualizations
- Performance charts
- Vulnerability dashboards

### Markdown Reports

GitHub-compatible documentation:

```bash
reports/comprehensive_report.md
```

## Test Coverage Targets

| Category | Target | Description |
| --- | --- | --- |
| Line Coverage | 95% | Every line executed |
| Branch Coverage | 90% | All code paths taken |
| Mutation Score | 80% | Tests detect artificial bugs |
| Integration Tests | 100% | All component pairs tested |
| E2E Tests | 100% | All workflows validated |

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Comprehensive Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python -m tests.comprehensive.runner --all --coverage-threshold 95
      - uses: actions/upload-artifact@v3
        with:
          name: test-reports
          path: reports/
```

## Architecture

```text
tests/comprehensive/
├── __init__.py          # Core framework, coverage analysis
├── mutation_tester.py   # Mutation testing engine
├── stress_test.py       # Performance & load testing
├── security_tester.py   # Security validation suite
├── integration_e2e.py   # Integration & E2E tests
└── runner.py            # Unified test orchestrator
```

## Deterministic Testing

All tests use controlled random seeds for reproducibility:

```python
# Default seed
python -m tests.comprehensive.runner --all --seed 42

# Custom seed for specific test scenarios
python -m tests.comprehensive.runner --all --seed 12345
```

This ensures that:

- Test results are reproducible across runs
- Random inputs are deterministic
- Performance measurements are comparable
- Fuzzing inputs follow predictable patterns

## Extending the Framework

### Adding New Test Cases

```python
from tests.comprehensive import AdversarialTestFramework

class MyTestSuite:
    def test_new_feature(self):
        # Your test implementation
        pass

# Register with framework
framework = AdversarialTestFramework()
framework.register_test_category("my_category", MyTestSuite)
```

### Adding Custom Mutation Operators

```python
from tests.comprehensive.mutation_tester import MutationOperator, MutationType

class CustomMutator(MutationOperator):
    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        # Custom mutation logic
        pass
```

## Troubleshooting

### High Memory Usage

```bash
# Reduce parallel workers
python -m tests.comprehensive.runner --all --parallel 1

# Skip memory-intensive tests
python -m tests.comprehensive.runner --category unit --category integration
```

### Slow Mutation Testing

```bash
# Run on specific modules only
python tests/comprehensive/mutation_tester.py --target utils/eeg_processing.py
```

### Coverage Not Detected

```bash
# Run with coverage explicitly
pytest --cov=. --cov-report=html tests/
```

## License

This testing framework is part of the APGI validation framework.

## Contributing

When adding new tests:

1. Follow the existing module structure
2. Include comprehensive docstrings
3. Add corresponding test cases to runner.py
4. Update this README with new features
5. Ensure deterministic behavior with seeds
