"""Derive dbt connection fields from the single DATABASE_URL contract.

dbt-postgres needs discrete host/port/user/password/dbname fields, so the
entry points that spawn dbt (`just db-dbt`, the ETL flow) export POSTGRES_*
variables parsed from the one application connection URL. psycopg's conninfo
parser handles userinfo, host, port, dbname, and query parameters, so the
same URL works passwordless (Homebrew trust) or password-bearing (Compose).
"""

from __future__ import annotations

import os
import shlex

from psycopg.conninfo import conninfo_to_dict

# dbt/profiles.yml reads these (with defaults); entry points export them.
DBT_ENV_KEYS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
)


def dbt_env(url: str) -> dict[str, str]:
    """Map a DATABASE_URL to the dbt profile's POSTGRES_* field variables."""
    info = conninfo_to_dict(url)
    return {
        "POSTGRES_HOST": str(info["host"] or "127.0.0.1"),
        "POSTGRES_PORT": str(info["port"] or "5432"),
        "POSTGRES_USER": str(info["user"] or "postgres"),
        "POSTGRES_PASSWORD": str(info.get("password") or ""),
        "POSTGRES_DB": str(info["dbname"] or "tennis"),
    }


def dbt_exports() -> str:
    """Shell `export` lines for the current DATABASE_URL (captured by eval in
    `just db-dbt`, so the values never reach a terminal or log)."""
    env = dbt_env(os.environ["DATABASE_URL"])
    return "\n".join(f"export {key}={shlex.quote(env[key])}" for key in DBT_ENV_KEYS)


if __name__ == "__main__":
    print(dbt_exports())
