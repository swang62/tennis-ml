"""Hermetic tests for the static IOC country reference."""

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
