## Summary
<!-- What changes and why. -->

## Change impact (APGI-REL-003)
- **Change type:** <!-- scientific formula/model · protocol definitions/loader · classification/reports · security/config · dependencies · docs only -->
- **Claims affected (CLM-nnn):** <!-- docs/validation/notation-and-claims.md, or "none" -->
- **Requirements affected (APGI-*):** <!-- docs/validation/traceability.md -->
- **Risks:** <!-- what could go wrong scientifically, operationally or for security -->
- **Registered-plan deviations:** <!-- utils/deviation_register.py, or "none" -->

## Regression selection
<!-- Tick the suites mandated for the change type (docs/validation/release-procedure.md §5) and link the CI run. -->
- [ ] Scientific: `test_scientific_conformance.py`, `test_theory_equations_core.py`, `test_apgi_engine.py`
- [ ] Protocol: `validate_protocols.py`, `test_protocol_lock_enforcement.py`, `test_protocol_spec_v1_1.py`
- [ ] Classification/reports: `test_kgate_classification.py`, `test_provenance_and_reports.py`, `test_psychometrics.py`
- [ ] Security/config: `test_data_governance_controls.py`, secret scan, SAST
- [ ] Dependencies: full suite on 3.11 and 3.12, `make lock-check`, pip-audit

## Version impact (APGI-REL-004)
<!-- major (registered decision / required field / canonical formula changes) · minor (additive) · patch (no fixture output changes) -->
