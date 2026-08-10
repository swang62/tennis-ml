"""Shared IOC country-code reference: validation, resolution, and UNK fallback.

The versioned reference data lives in data/ioc_countries.csv (ioc, iso2,
country_name) and is the single source of truth for which IOC codes are valid
and how they resolve. Profile import (src/flows/ingest.py) and serving both
use these rules, so a code that is valid at import time resolves identically
at read time.

Conventions:
- A valid IOC is a code present in the reference CSV with a usable ISO
  alpha-2 code (the UNK sentinel is valid but has no ISO code).
- Missing or invalid source values normalize to UNK ("Country unknown");
  nationality is never inferred from names or other fields.
- No third-party country API is ever called at runtime.
"""

from __future__ import annotations

import csv
from functools import lru_cache

from src.constants import ROOT

IOC_CSV = ROOT / "data" / "ioc_countries.csv"

# Sentinel code stored when a profile's IOC is missing or not verifiable.
UNK = "UNK"
UNKNOWN_NAME = "Country unknown"


@lru_cache(maxsize=1)
def load_countries() -> dict[str, tuple[str, str]]:
    """Load the reference CSV into {ioc: (iso2, country_name)}.

    Codes are normalized to uppercase; iso2 is empty only for the UNK
    sentinel (there is no ISO code for "unknown"). Raises on a malformed
    reference so data errors surface at import time, never silently.
    """
    countries: dict[str, tuple[str, str]] = {}
    with open(IOC_CSV, newline="") as f:
        for row in csv.DictReader(f):
            code = str(row["ioc"]).strip().upper()
            if not code:
                raise ValueError(f"{IOC_CSV}: empty ioc row")
            if code in countries:
                raise ValueError(f"{IOC_CSV}: duplicate ioc code {code!r}")
            countries[code] = (str(row["iso2"]).strip(), str(row["country_name"]).strip())
    if UNK not in countries:
        raise ValueError(f"{IOC_CSV}: missing UNK sentinel row")
    return countries


def normalize_ioc(value: object) -> str:
    """Trim and uppercase a raw IOC value; empty/None normalize to UNK."""
    if value is None:
        return UNK
    code = str(value).strip().upper()
    return code or UNK


def is_known_ioc(code: str) -> bool:
    """True when the normalized code is present in the reference CSV."""
    return normalize_ioc(code) in load_countries()


def valid_ioc(value: object) -> str:
    """IOC to store for a profile: the verified code, or UNK when unverifiable.

    Missing/invalid values (empty, whitespace, unknown codes such as
    historical non-ISO codes or typos) resolve to the UNK sentinel only;
    verified codes are preserved exactly as normalized (trimmed/uppercased).
    """
    code = normalize_ioc(value)
    if code != UNK and code not in load_countries():
        return UNK
    return code


def resolve_ioc(ioc: str) -> tuple[str, str]:
    """Resolve a normalized IOC to (iso2, country_name).

    Known codes resolve to their reference row; UNK itself and any
    unknown/missing code resolve to the UNK row ("", "Country unknown").
    """
    countries = load_countries()
    return countries.get(normalize_ioc(ioc), countries[UNK])
