"""Hermetic tests for the static IOC country reference."""

import src.utils.countries as countries_mod

UNKNOWN = ("", "Country unknown")


def test_known_ioc_resolves():
    assert countries_mod.resolve_ioc("USA") == ("US", "United States")
    assert countries_mod.resolve_ioc("FRA") == ("FR", "France")
    assert countries_mod.resolve_ioc("GBR") == ("GB", "Great Britain")


def test_unknown_ioc_resolves_to_unk():
    assert countries_mod.valid_ioc("XYZ") == countries_mod.UNK
    assert countries_mod.resolve_ioc("XYZ") == UNKNOWN
    assert countries_mod.resolve_ioc("") == UNKNOWN


def test_case_insensitivity():
    assert countries_mod.normalize_ioc("usa") == "USA"
    assert countries_mod.resolve_ioc("fra") == ("FR", "France")
    assert countries_mod.is_known_ioc("gbr")
    assert countries_mod.valid_ioc(" fra ") == "FRA"
