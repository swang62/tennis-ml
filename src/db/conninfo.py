"""Derive dbt's discrete connection fields from DATABASE_URL."""

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
    """Return shell exports for DATABASE_URL without printing values."""
    env = dbt_env(os.environ["DATABASE_URL"])
    return "\n".join(f"export {key}={shlex.quote(env[key])}" for key in DBT_ENV_KEYS)


if __name__ == "__main__":
    print(dbt_exports())
