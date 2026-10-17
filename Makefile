# APGI Validation Framework Makefile
# ==================================

.PHONY: help clean test test-fast test-slow test-full gui-smoke lint format threshold-lint \
        lock lock-check install dev-install venv docs manifest manifest-check protocols-check

help:
	@echo "Available targets:"
	@echo "  venv           - Create virtual environment"
	@echo "  clean          - Remove temporary files and cache directories"
	@echo "  test           - Run test suite (excludes slow protocol tests)"
	@echo "  test-fast      - Run fast test suite (< 5 minutes)"
	@echo "  test-slow      - Run only the long-running protocol tests"
	@echo "  test-full      - Run everything, slow tests included"
	@echo "  gui-smoke      - Instantiate every GUI against a real display"
	@echo "  lint           - Run linting and type checks (fails on error)"
	@echo "  format         - Auto-format with black and isort"
	@echo "  lock           - Regenerate requirements.lock"
	@echo "  lock-check     - Verify requirements.lock is current (CI gate)"
	@echo "  manifest       - Recompute all protocol SHA-256 hashes"
	@echo "  manifest-check - Check manifest for drift without writing (CI gate)"
	@echo "  protocols-check - Validate the 1.1.0 protocol specification release (CI gate)"
	@echo "  install        - Install package and dependencies"
	@echo "  dev-install    - Install in development mode"
	@echo "  docs           - Generate documentation"

venv:
	python3 -m venv .venv

clean:
	@echo "Cleaning temporary files and cache directories..."
	python3 delete_pycache.py --yes

test:
	python3 -m pytest -m "not slow"

test-fast:
	@echo "Running fast test suite (< 5 minutes)..."
	python3 -m pytest -c pytest-fast.ini --ignore=tests/comprehensive

test-slow:
	@echo "Running long-running protocol tests..."
	python3 -m pytest -m "slow" --no-cov --timeout=1800

test-full:
	@echo "Running full test suite (slow tests included)..."
	python3 -m pytest tests/ --ignore=tests/comprehensive

gui-smoke:
	@echo "Instantiating every GUI against a real display..."
	APGI_REAL_TKINTER=1 python3 -m pytest tests/test_gui_smoke_cross_platform.py -v --no-cov

# Checks are gates, not suggestions: each one fails the target on error.
# These previously ended in `|| true` or `--exit-zero`, so `make lint` could
# not fail no matter what the tools reported.
lint:
	@echo "Checking formatting (black)..."
	python3 -m black --check .
	@echo "Checking import order (isort)..."
	python3 -m isort --check-only .
	@echo "Linting (ruff)..."
	python3 -m ruff check .
	@echo "Type checking (mypy)..."
	python3 -m mypy utils/ apgi_core/
	$(MAKE) threshold-lint

format:
	python3 -m black .
	python3 -m isort .

threshold-lint:
	python3 utils/threshold_lint.py

# --no-header keeps the output independent of the destination filename, so the
# check below compares resolved versions rather than the command that produced
# them.
UV_COMPILE = uv pip compile requirements.txt --python-version 3.11 --no-header

lock:
	@echo "Regenerating requirements.lock for Python 3.11..."
	$(UV_COMPILE) --output-file requirements.lock

# Checks that every declared requirement is pinned and that each pin satisfies
# its version specifiers. A byte-for-byte diff against a fresh resolution would
# instead fail whenever any transitive dependency publishes a release, which
# says nothing about this repository.
lock-check:
	@echo "Verifying requirements.lock is consistent with requirements.txt..."
	python3 -m utils.check_lockfile

manifest:
	@echo "Refreshing protocol manifest hashes..."
	python3 -m utils.protocol_manifest --refresh

manifest-check:
	@echo "Checking protocol manifest for drift..."
	python3 -m utils.protocol_manifest --check

protocols-check:
	@echo "Validating protocol specification release 1.1.0..."
	python3 scripts/validate_protocols.py

install: venv
	.venv/bin/pip install .

dev-install: venv
	.venv/bin/pip install -e .

docs:
	@echo "Generating documentation..."
	@echo "Converting Markdown to HTML..."
	@command -v pandoc >/dev/null 2>&1 || { echo "pandoc not installed. Install with: brew install pandoc (macOS) or apt install pandoc (Ubuntu)"; exit 1; }
	@pandoc docs/APGI_Equations.md -o docs/APGI_Equations.html --metadata title="APGI Equations" --css=github-markdown.css
	@pandoc docs/APGI-Parameter-Specifications.md -o docs/APGI-Parameter-Specifications.html --metadata title="APGI Parameter Specifications" --css=github-markdown.css
	@pandoc docs/APGI-Empirical-Credibility-Roadmap.md -o docs/APGI-Empirical-Credibility-Roadmap.html --metadata title="APGI Empirical Credibility Roadmap" --css=github-markdown.css
	@pandoc docs/APGI-Falsification-Criteria.md -o docs/APGI-Falsification-Criteria.html --metadata title="APGI Falsification Criteria" --css=github-markdown.css
	@pandoc docs/Tutorial.md -o docs/Tutorial.html --metadata title="APGI Tutorial" --css=github-markdown.css
	@pandoc docs/Innovations.md -o docs/Innovations.html --metadata title="APGI Innovations" --css=github-markdown.css
	@pandoc docs/Innovations-Software-Implementation.md -o docs/Innovations-Software-Implementation.html --metadata title="APGI Software Implementation" --css=github-markdown.css
	@pandoc docs/Multimodal-Integration.md -o docs/Multimodal-Integration.html --metadata title="APGI Multimodal Integration" --css=github-markdown.css
	@echo "Documentation generated in docs/ directory"
	@echo "Open HTML files in a web browser to view"
