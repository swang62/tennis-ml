"""Self-contained tests for the static IOC country reference (src/countries).

The reference is the module-level ``_COUNTRIES`` constant; these tests never
read a CSV or any external file.
"""

import src.countries as countries_mod

UNKNOWN = ("", "Country unknown")


def test_known_ioc_resolves():
    assert countries_mod.resolve_ioc("USA") == ("US", "United States")
    assert countries_mod.resolve_ioc("FRA") == ("FR", "France")
    assert countries_mod.resolve_ioc("GBR") == ("GB", "Great Britain")


def test_unk_resolves_to_sentinel_row():
    assert countries_mod.resolve_ioc(countries_mod.UNK) == UNKNOWN


def test_unknown_ioc_resolves_to_unk():
    assert countries_mod.valid_ioc("XYZ") == countries_mod.UNK
    assert countries_mod.resolve_ioc("XYZ") == UNKNOWN
    assert countries_mod.resolve_ioc("") == UNKNOWN


def test_normalize_ioc_none_and_whitespace():
    assert countries_mod.normalize_ioc(None) == countries_mod.UNK
    assert countries_mod.normalize_ioc("") == countries_mod.UNK
    assert countries_mod.normalize_ioc("   ") == countries_mod.UNK


def test_is_known_ioc():
    assert countries_mod.is_known_ioc("FRA")
    assert countries_mod.is_known_ioc("UNK")
    assert not countries_mod.is_known_ioc("XYZ")


def test_valid_ioc():
    assert countries_mod.valid_ioc("FRA") == "FRA"
    assert countries_mod.valid_ioc("UNK") == "UNK"
    assert countries_mod.valid_ioc("XYZ") == countries_mod.UNK
    assert countries_mod.valid_ioc("   ") == countries_mod.UNK
    assert countries_mod.valid_ioc(None) == countries_mod.UNK
    assert countries_mod.valid_ioc("") == countries_mod.UNK


def test_case_insensitivity():
    assert countries_mod.normalize_ioc("usa") == "USA"
    assert countries_mod.resolve_ioc("fra") == ("FR", "France")
    assert countries_mod.is_known_ioc("gbr")
    assert countries_mod.valid_ioc(" fra ") == "FRA"


def test_no_duplicate_ioc_codes():
    assert len(countries_mod._COUNTRIES) == len(set(countries_mod._COUNTRIES))


def test_every_non_unk_key_has_nonempty_iso2():
    assert countries_mod._COUNTRIES[countries_mod.UNK][0] == ""
    for code, (iso2, _) in countries_mod._COUNTRIES.items():
        if code != countries_mod.UNK:
            assert iso2, f"{code} has empty iso2"


def test_every_value_is_a_two_field_tuple():
    for code, value in countries_mod._COUNTRIES.items():
        assert isinstance(value, tuple) and len(value) == 2, code
        assert isinstance(value[0], str) and isinstance(value[1], str), code
