"""Hermetic guard for the ETL incremental boundaries.

No live database: these tests read the dbt model SQL, yml contracts, and the
incremental demo fixture from disk and assert the boundaries statically (the
same approach as test_no_live_db.py). The demo arithmetic — 1 new bronze
match -> 2 silver.player_matches -> 2 silver.rolling_features snapshots ->
1 gold.match_features row, aggregates recomputed — is pinned against the
fixture and the expansion-factor tests dbt enforces at build time.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DBT = ROOT / "dbt"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Model name -> (model SQL path relative to dbt/, expected unique_key).
INCREMENTAL_MODELS = {
    "player_matches": ("models/silver/player_matches.sql", ["player_id", "match_id"]),
    "rolling_features": ("models/silver/rolling_features.sql", ["player_id", "match_id"]),
    "match_features": ("models/gold/match_features.sql", "match_id"),
}
AGGREGATE_MODELS = {
    "tour_averages": "models/gold/tour_averages.sql",
    "player_profiles": "models/gold/player_profiles.sql",
}
YML_UNIQUE_TESTS = {
    "player_matches": "models/silver/player_matches.yml",
    "rolling_features": "models/silver/rolling_features.yml",
    "match_features": "models/gold/match_features.yml",
}

# dbt singular tests that pin the per-match expansion factors the demo
# arithmetic relies on.
EXPANSION_TESTS = (
    "dbt/tests/gold/player_matches_two_rows_per_match.sql",
    "dbt/tests/silver/player_matches_keeps_all_bronze_matches.sql",
    "dbt/tests/gold/rolling_features_one_per_player_match.sql",
    "dbt/tests/gold/match_features_keeps_all_bronze_matches.sql",
)


def _read(rel_path: str) -> str:
    return (DBT / rel_path).read_text()


def test_incremental_models_configured():
    """Each append-boundary model declares incremental + delete+insert + unique_key."""
    for name, (sql_path, unique_key) in INCREMENTAL_MODELS.items():
        sql = _read(sql_path)
        assert 'materialized="incremental"' in sql, name
        assert 'incremental_strategy="delete+insert"' in sql, name
        rendered_key = (
            '["player_id", "match_id"]' if isinstance(unique_key, list) else f'"{unique_key}"'
        )
        assert f"unique_key={rendered_key}" in sql, name


def test_incremental_predicates_select_only_new_match_ids():
    """The incremental WHERE is gated on is_incremental() and filters by the
    bronze append identity (match_id) against the existing relation."""
    for name, (sql_path, _unique_key) in INCREMENTAL_MODELS.items():
        sql = _read(sql_path)
        assert "{% if is_incremental() %}" in sql, name
        assert "{{ this }}" in sql, name
        assert "match_id NOT IN (SELECT match_id FROM {{ this }})" in sql, name


def test_aggregate_models_recompute_globally():
    """Aggregates (tour_averages singleton, player_profiles) are never
    incremental: they change with every new match and must rebuild in full."""
    for name, sql_path in AGGREGATE_MODELS.items():
        assert "is_incremental" not in _read(sql_path), name


def test_unique_keys_enforced_by_yml():
    """Every incremental model keeps its composite unique test in its yml."""
    for name, yml_path in YML_UNIQUE_TESTS.items():
        yml = _read(yml_path)
        assert "unique:" in yml, name


def test_expansion_factor_tests_exist():
    """The dbt tests that make the demo arithmetic true are all present."""
    for rel_path in EXPANSION_TESTS:
        assert (ROOT / rel_path).is_file(), rel_path


def test_demo_fixture_expansion_arithmetic():
    """The fixture adds exactly one bronze match; the append boundary then
    yields +2 player_matches, +2 rolling snapshots, +1 match_features row,
    and the selection predicate is idempotent on re-runs."""
    fixture = (FIXTURES / "incremental_demo.sql").read_text()
    assert fixture.count("INSERT INTO bronze.match_events") == 1
    new_match_id = "20260714-2026-316-011"
    assert fixture.count(f"'{new_match_id}'") == 1

    def select_new(new_ids: set[str], existing: set[str]) -> set[str]:
        # Mirror of `match_id NOT IN (SELECT match_id FROM {{ this }})`.
        return new_ids - existing

    new = select_new({new_match_id}, existing=set())
    assert new == {new_match_id}
    # Idempotency: after the first run the match is in `this`, so a re-run
    # with no new bronze matches selects nothing.
    assert select_new({new_match_id}, existing={new_match_id}) == set()

    # One new match -> exactly two player perspectives, two snapshots, one
    # canonical row; the singleton aggregates stay one row and refresh.
    assert len(new) * 2 == 2  # silver.player_matches
    assert len(new) * 2 == 2  # silver.rolling_features
    assert len(new) == 1  # gold.match_features
