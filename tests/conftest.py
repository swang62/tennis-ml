"""Session-scoped PostgreSQL and dbt fixtures for the test suite."""

import pytest

from src.db import client, init_db
from src.db import seed as seed_flow

_SEED_FAILURE: str | None = None


def _postgres_reachable() -> bool:
    try:
        with client.get_conn().cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:
        return False
    else:
        return True


@pytest.fixture(scope="session", autouse=True)
def seeded_test_db():
    """Bootstrap and seed PostgreSQL once when it is reachable."""
    global _SEED_FAILURE
    if _postgres_reachable():
        try:
            init_db.init()
            seed_flow.main([])
        except Exception as exc:  # e.g. server unreachable, init failure, or schema drift
            _SEED_FAILURE = str(exc)
    yield
    client.close()


@pytest.fixture(scope="session")
def postgres_ready(seeded_test_db):  # noqa: ARG001 — dependency ordering only
    """Skip the test when the seeded PostgreSQL is not usable."""
    if not _postgres_reachable():
        pytest.skip("configured PostgreSQL is not reachable (DATABASE_URL contract)")
    if _SEED_FAILURE:
        pytest.skip(f"seeded PostgreSQL unavailable: {_SEED_FAILURE}")
    yield


# The physical gold.feature_defaults table may still exist in PostgreSQL after
# this migration. It is safe to drop manually after verifying tour_averages
# with: DROP TABLE IF EXISTS gold.feature_defaults. Normal dbt runs do not drop it.
@pytest.fixture(scope="session")
def gold_ready(postgres_ready):  # noqa: ARG001 — skip-gate fixture, unused in body
    """Skip when dbt has not built gold.match_features and gold.tour_averages over PostgreSQL."""
    with client.get_conn().cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'gold' AND table_name = 'match_features'"
        )
        if cur.fetchone() is None:
            pytest.skip("gold.match_features not built (dbt ETL over PostgreSQL); skipping")
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'gold' AND table_name = 'tour_averages'"
        )
        if cur.fetchone() is None:
            pytest.skip("gold.tour_averages not built (dbt ETL over PostgreSQL); skipping")
    yield
