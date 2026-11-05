"""TC-SCI-04: parameter provenance registry is complete and honest (APGI-SCI-005)."""

from utils.apgi_settings import APGISettings
from utils.parameter_provenance import LABELS, registry, settings_provenance


def test_every_settings_default_is_registered():
    scientific = [
        n
        for n, f in APGISettings.model_fields.items()
        if isinstance(f.default, (int, float, list)) and not isinstance(f.default, bool)
    ]
    missing = [n for n in scientific if f"APGISettings.{n}" not in registry()]
    assert not missing, f"defaults without provenance entries: {missing}"


def test_labels_valid_and_derived_claims_cite_sources():
    for name, entry in registry().items():
        assert entry["label"] in LABELS, name
        if entry["label"] in ("analytically_derived", "post_hoc_calibrated"):
            assert entry["source"] and not str(entry["source"]).startswith("UNREFERENCED"), name


def test_settings_provenance_propagates():
    prov = settings_provenance()
    assert prov["APGISettings.theta0"]["label"] in LABELS
