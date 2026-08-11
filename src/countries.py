"""Shared IOC country-code reference: validation, resolution, and UNK fallback.

The reference mapping lives in the module-level ``_COUNTRIES`` constant and is
the single source of truth for which IOC codes are valid and how they resolve.
Profile import (src/flows/ingest.py) and serving both use these rules, so a
code that is valid at import time resolves identically at read time.

Conventions:
- A valid IOC is a code present in the reference constant with a usable ISO
  alpha-2 code (the UNK sentinel is valid but has no ISO code).
- Missing or invalid source values normalize to UNK ("Country unknown");
  nationality is never inferred from names or other fields.
- No third-party country API is ever called at runtime.
"""

from __future__ import annotations

# Sentinel code stored when a profile's IOC is missing or not verifiable.
UNK = "UNK"
UNKNOWN_NAME = "Country unknown"

# {ioc: (iso2, country_name)}. Codes are uppercase; iso2 is empty only for the
# UNK sentinel (there is no ISO code for "unknown"). Source: the former
# data/ioc_countries.csv reference, inlined verbatim.
_COUNTRIES: dict[str, tuple[str, str]] = {
    "AFG": ("AF", "Afghanistan"),
    "ALB": ("AL", "Albania"),
    "ALG": ("DZ", "Algeria"),
    "AND": ("AD", "Andorra"),
    "ANG": ("AO", "Angola"),
    "ANT": ("AG", "Antigua and Barbuda"),
    "ARG": ("AR", "Argentina"),
    "ARM": ("AM", "Armenia"),
    "ARU": ("AW", "Aruba"),
    "ASA": ("AS", "American Samoa"),
    "ATG": ("AG", "Antigua and Barbuda"),
    "AUS": ("AU", "Australia"),
    "AUT": ("AT", "Austria"),
    "AZE": ("AZ", "Azerbaijan"),
    "BAH": ("BS", "Bahamas"),
    "BAN": ("BD", "Bangladesh"),
    "BAR": ("BB", "Barbados"),
    "BDI": ("BI", "Burundi"),
    "BEL": ("BE", "Belgium"),
    "BEN": ("BJ", "Benin"),
    "BER": ("BM", "Bermuda"),
    "BFA": ("BF", "Burkina Faso"),
    "BHU": ("BT", "Bhutan"),
    "BIH": ("BA", "Bosnia and Herzegovina"),
    "BIZ": ("BZ", "Belize"),
    "BLR": ("BY", "Belarus"),
    "BOL": ("BO", "Bolivia"),
    "BOT": ("BW", "Botswana"),
    "BRA": ("BR", "Brazil"),
    "BRN": ("BH", "Bahrain"),
    "BRU": ("BN", "Brunei"),
    "BUL": ("BG", "Bulgaria"),
    "BUR": ("BF", "Burkina Faso"),
    "BWA": ("BW", "Botswana"),
    "CAF": ("CF", "Central African Republic"),
    "CAM": ("KH", "Cambodia"),
    "CAN": ("CA", "Canada"),
    "CAY": ("KY", "Cayman Islands"),
    "CGO": ("CG", "Congo"),
    "CHA": ("TD", "Chad"),
    "CHI": ("CL", "Chile"),
    "CHN": ("CN", "China"),
    "CIV": ("CI", "Ivory Coast"),
    "CMR": ("CM", "Cameroon"),
    "COD": ("CD", "DR Congo"),
    "COK": ("CK", "Cook Islands"),
    "COL": ("CO", "Colombia"),
    "COM": ("KM", "Comoros"),
    "CPV": ("CV", "Cape Verde"),
    "CRC": ("CR", "Costa Rica"),
    "CRO": ("HR", "Croatia"),
    "CUB": ("CU", "Cuba"),
    "CUR": ("CW", "Curacao"),
    "CYP": ("CY", "Cyprus"),
    "CZE": ("CZ", "Czechia"),
    "DEN": ("DK", "Denmark"),
    "DJI": ("DJ", "Djibouti"),
    "DMA": ("DM", "Dominica"),
    "DOM": ("DO", "Dominican Republic"),
    "ECU": ("EC", "Ecuador"),
    "EGY": ("EG", "Egypt"),
    "ERI": ("ER", "Eritrea"),
    "ESA": ("SV", "El Salvador"),
    "ESP": ("ES", "Spain"),
    "EST": ("EE", "Estonia"),
    "ETH": ("ET", "Ethiopia"),
    "FIJ": ("FJ", "Fiji"),
    "FIN": ("FI", "Finland"),
    "FRA": ("FR", "France"),
    "GAB": ("GA", "Gabon"),
    "GAM": ("GM", "Gambia"),
    "GBR": ("GB", "Great Britain"),
    "GBS": ("GW", "Guinea-Bissau"),
    "GEO": ("GE", "Georgia"),
    "GEQ": ("GQ", "Equatorial Guinea"),
    "GER": ("DE", "Germany"),
    "GHA": ("GH", "Ghana"),
    "GRE": ("GR", "Greece"),
    "GRN": ("GD", "Grenada"),
    "GUA": ("GT", "Guatemala"),
    "GUI": ("GN", "Guinea"),
    "GUM": ("GU", "Guam"),
    "GUY": ("GY", "Guyana"),
    "HAI": ("HT", "Haiti"),
    "HKG": ("HK", "Hong Kong"),
    "HON": ("HN", "Honduras"),
    "HUN": ("HU", "Hungary"),
    "INA": ("ID", "Indonesia"),
    "IND": ("IN", "India"),
    "IRI": ("IR", "Iran"),
    "IRL": ("IE", "Ireland"),
    "IRQ": ("IQ", "Iraq"),
    "ISL": ("IS", "Iceland"),
    "ISR": ("IL", "Israel"),
    "ITA": ("IT", "Italy"),
    "IVB": ("VG", "British Virgin Islands"),
    "JAM": ("JM", "Jamaica"),
    "JOR": ("JO", "Jordan"),
    "JPN": ("JP", "Japan"),
    "KAZ": ("KZ", "Kazakhstan"),
    "KEN": ("KE", "Kenya"),
    "KGZ": ("KG", "Kyrgyzstan"),
    "KIR": ("KI", "Kiribati"),
    "KOR": ("KR", "South Korea"),
    "KOS": ("XK", "Kosovo"),
    "KSA": ("SA", "Saudi Arabia"),
    "KUW": ("KW", "Kuwait"),
    "LAO": ("LA", "Laos"),
    "LAT": ("LV", "Latvia"),
    "LBA": ("LY", "Libya"),
    "LBN": ("LB", "Lebanon"),
    "LBR": ("LR", "Liberia"),
    "LCA": ("LC", "Saint Lucia"),
    "LES": ("LS", "Lesotho"),
    "LIB": ("LR", "Liberia"),
    "LIE": ("LI", "Liechtenstein"),
    "LKA": ("LK", "Sri Lanka"),
    "LTU": ("LT", "Lithuania"),
    "LUX": ("LU", "Luxembourg"),
    "MAD": ("MG", "Madagascar"),
    "MAR": ("MA", "Morocco"),
    "MAS": ("MY", "Malaysia"),
    "MAW": ("MW", "Malawi"),
    "MCO": ("MC", "Monaco"),
    "MDA": ("MD", "Moldova"),
    "MDV": ("MV", "Maldives"),
    "MEX": ("MX", "Mexico"),
    "MGL": ("MN", "Mongolia"),
    "MHL": ("MH", "Marshall Islands"),
    "MKD": ("MK", "North Macedonia"),
    "MLI": ("ML", "Mali"),
    "MLT": ("MT", "Malta"),
    "MNE": ("ME", "Montenegro"),
    "MON": ("MC", "Monaco"),
    "MOZ": ("MZ", "Mozambique"),
    "MRI": ("MU", "Mauritius"),
    "MTN": ("MR", "Mauritania"),
    "MUS": ("MU", "Mauritius"),
    "MYA": ("MM", "Myanmar"),
    "NAM": ("NA", "Namibia"),
    "NCA": ("NI", "Nicaragua"),
    "NCL": ("NC", "New Caledonia"),
    "NED": ("NL", "Netherlands"),
    "NEP": ("NP", "Nepal"),
    "NGA": ("NG", "Nigeria"),
    "NGR": ("NG", "Nigeria"),
    "NIG": ("NE", "Niger"),
    "NMI": ("MP", "Northern Mariana Islands"),
    "NOR": ("NO", "Norway"),
    "NRU": ("NR", "Nauru"),
    "NZL": ("NZ", "New Zealand"),
    "OMA": ("OM", "Oman"),
    "PAK": ("PK", "Pakistan"),
    "PAN": ("PA", "Panama"),
    "PAR": ("PY", "Paraguay"),
    "PER": ("PE", "Peru"),
    "PHI": ("PH", "Philippines"),
    "PLE": ("PS", "Palestine"),
    "PLW": ("PW", "Palau"),
    "PNG": ("PG", "Papua New Guinea"),
    "POL": ("PL", "Poland"),
    "POR": ("PT", "Portugal"),
    "PRI": ("PR", "Puerto Rico"),
    "PRK": ("KP", "North Korea"),
    "PRT": ("PT", "Portugal"),
    "PRY": ("PY", "Paraguay"),
    "PUR": ("PR", "Puerto Rico"),
    "QAT": ("QA", "Qatar"),
    "ROU": ("RO", "Romania"),
    "RSA": ("ZA", "South Africa"),
    "RUS": ("RU", "Russia"),
    "RWA": ("RW", "Rwanda"),
    "SAM": ("WS", "Samoa"),
    "SDN": ("SD", "Sudan"),
    "SEN": ("SN", "Senegal"),
    "SEY": ("SC", "Seychelles"),
    "SGP": ("SG", "Singapore"),
    "SIN": ("SG", "Singapore"),
    "SKN": ("KN", "Saint Kitts and Nevis"),
    "SLE": ("SL", "Sierra Leone"),
    "SLO": ("SI", "Slovenia"),
    "SMR": ("SM", "San Marino"),
    "SOL": ("SB", "Solomon Islands"),
    "SOM": ("SO", "Somalia"),
    "SRB": ("RS", "Serbia"),
    "SRI": ("LK", "Sri Lanka"),
    "SSD": ("SS", "South Sudan"),
    "STP": ("ST", "Sao Tome and Principe"),
    "SUD": ("SD", "Sudan"),
    "SUI": ("CH", "Switzerland"),
    "SUR": ("SR", "Suriname"),
    "SVK": ("SK", "Slovakia"),
    "SVN": ("SI", "Slovenia"),
    "SWE": ("SE", "Sweden"),
    "SWZ": ("SZ", "Eswatini"),
    "SYR": ("SY", "Syria"),
    "TAN": ("TZ", "Tanzania"),
    "TGA": ("TO", "Tonga"),
    "TGO": ("TG", "Togo"),
    "THA": ("TH", "Thailand"),
    "TJK": ("TJ", "Tajikistan"),
    "TKM": ("TM", "Turkmenistan"),
    "TLS": ("TL", "Timor-Leste"),
    "TOG": ("TG", "Togo"),
    "TPE": ("TW", "Chinese Taipei"),
    "TRI": ("TT", "Trinidad and Tobago"),
    "TTO": ("TT", "Trinidad and Tobago"),
    "TUN": ("TN", "Tunisia"),
    "TUR": ("TR", "Turkey"),
    "TUV": ("TV", "Tuvalu"),
    "TWN": ("TW", "Chinese Taipei"),
    "UAE": ("AE", "United Arab Emirates"),
    "UGA": ("UG", "Uganda"),
    "UKR": ("UA", "Ukraine"),
    "UNK": ("", "Country unknown"),
    "URU": ("UY", "Uruguay"),
    "URY": ("UY", "Uruguay"),
    "USA": ("US", "United States"),
    "UZB": ("UZ", "Uzbekistan"),
    "VAN": ("VU", "Vanuatu"),
    "VEN": ("VE", "Venezuela"),
    "VIE": ("VN", "Vietnam"),
    "VIN": ("VC", "Saint Vincent and the Grenadines"),
    "VNM": ("VN", "Vietnam"),
    "YEM": ("YE", "Yemen"),
    "ZAM": ("ZM", "Zambia"),
    "ZIM": ("ZW", "Zimbabwe"),
    "ZWE": ("ZW", "Zimbabwe"),
}


def normalize_ioc(value: object) -> str:
    """Trim and uppercase a raw IOC value; empty/None normalize to UNK."""
    if value is None:
        return UNK
    code = str(value).strip().upper()
    return code or UNK


def is_known_ioc(code: str) -> bool:
    """True when the normalized code is present in the reference mapping."""
    return normalize_ioc(code) in _COUNTRIES


def valid_ioc(value: object) -> str:
    """IOC to store for a profile: the verified code, or UNK when unverifiable.

    Missing/invalid values (empty, whitespace, unknown codes such as
    historical non-ISO codes or typos) resolve to the UNK sentinel only;
    verified codes are preserved exactly as normalized (trimmed/uppercased).
    """
    code = normalize_ioc(value)
    if code != UNK and code not in _COUNTRIES:
        return UNK
    return code


def resolve_ioc(ioc: str) -> tuple[str, str]:
    """Resolve a normalized IOC to (iso2, country_name).

    Known codes resolve to their reference row; UNK itself and any
    unknown/missing code resolve to the UNK row ("", "Country unknown").
    """
    return _COUNTRIES.get(normalize_ioc(ioc), _COUNTRIES[UNK])
