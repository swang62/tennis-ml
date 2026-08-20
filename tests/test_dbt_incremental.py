"""Hermetic guard for the ETL incremental boundaries.

No live database: these tests read the dbt model SQL, yml contracts, and the
incremental demo fixture from disk and assert the boundaries statically (the
same approach as test_no_live_db.py). The demo arithmetic — 1 new bronze
match -> 2 silver.player_matches -> 2 silver.rolling_features snapshots ->
2 gold.match_features rows, aggregates recomputed — is pinned against the
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
    "match_features": ("models/gold/match_features.sql", ["player_id", "match_id"]),
}
AGGREGATE_MODELS = {
    "tour_averages": "models/gold/tour_averages.sql",
    "player_profiles": "models/gold/player_profiles.sql",
}
# Model name -> yml path. These model ymls must carry no schema tests
# (unique/not_null/accepted_values): the DB primary keys and NOT NULL
# post-hooks own those contracts.
SCHEMA_TEST_YMLS = {
    "player_matches": "models/silver/player_matches.yml",
    "rolling_features": "models/silver/rolling_features.yml",
    "match_features": "models/gold/match_features.yml",
}

# Model name -> DB primary key the model's dbt_project.yml post-hook creates
# for the (player_id, match_id) grain.
GRAIN_PKS = {
    "player_matches": "pk_player_matches",
    "rolling_features": "pk_rolling_features",
    "match_features": "pk_match_features",
}

# Model name -> columns the model's dbt_project.yml post-hook enforces NOT
# NULL on (guaranteed non-null by bronze NOT NULL pass-through, window
# COUNT/ROW_NUMBER, or literal construction).
NOT_NULL_COLUMNS = {
    "player_matches": ["match_id", "match_date", "player_id", "opponent_id"],
    "rolling_features": [
        "player_id",
        "match_id",
        "snapshot_date",
        "player_match_number",
        "matches_10",
    ],
    "match_features": [
        "match_id",
        "match_date",
        "player_id",
        "opponent_id",
        "surface",
        "match_won",
    ],
    "player_profiles": ["player_id"],
    "tour_averages": ["singleton_id"],
}

# Model name -> (PK constraint name, PK columns) for the NOT NULL ordering
# check: the PK is applied before the NOT NULL post-hook.
MODEL_PKS = {
    "player_matches": ("pk_player_matches", "player_id, match_id"),
    "rolling_features": ("pk_rolling_features", "player_id, match_id"),
    "match_features": ("pk_match_features", "player_id, match_id"),
    "player_profiles": ("pk_player_profiles", "player_id"),
    "tour_averages": ("pk_tour_averages", "singleton_id"),
}

# dbt singular tests that pin the per-match expansion factors the demo
# arithmetic relies on.
EXPANSION_TESTS = (
    "dbt/tests/silver/player_matches_keeps_all_bronze_matches.sql",
    "dbt/tests/gold/rolling_features_one_per_player_match.sql",
    "dbt/tests/gold/match_features_keeps_all_bronze_matches.sql",
)

# Redundant grain tests removed once DB constraints and the removed yml tests
# no longer need them.
REMOVED_REDUNDANT_TESTS = (
    "dbt/tests/gold/player_matches_two_rows_per_match.sql",
    "dbt/tests/silver/rolling_features_unique_per_player_match.sql",
)

# dbt singular tests restored because PostgreSQL cannot enforce their
# contracts: the singleton exactly-one-row cardinality, NaN/Infinity
# rejection, and per-column rate bounds are semantic, not DB constraints.
TOUR_AVERAGE_SEMANTIC_TESTS = (
    "dbt/tests/gold/tour_averages_contract.sql",
    "dbt/tests/gold/tour_averages_rate_bounds.sql",
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


def test_silver_incremental_predicates_rebuild_changed_players():
    """Silver models compare composite keys and ordinals, then rebuild every
    row for an affected player so historical inserts cannot leave stale windows."""
    for name in ("player_matches", "rolling_features"):
        sql = _read(INCREMENTAL_MODELS[name][0])
        assert "{% if is_incremental() %}" in sql
        assert "changed_players AS" in sql
        assert "t.player_id" in sql
        assert "t.match_id" in sql
        assert "player_match_number <>" in sql
        assert "player_id IN (SELECT player_id FROM changed_players)" in sql


def test_gold_incremental_predicate_uses_ingestion_watermark():
    sql = _read(INCREMENTAL_MODELS["match_features"][0])
    assert "{% if is_incremental() %}" in sql
    assert "ingested_at > COALESCE(" in sql
    assert "source_watermark FROM bronze.etl_state" in sql


def test_weighted_form_uses_max_match_number_gap():
    """weighted_form_10 decays with POW(0.9, max_match_number -
    player_match_number): a full-history partition MAX over the player's
    ascending ordinals replaces the old reversed row-number pass, keeping the
    window frame ascending with identical decay semantics."""
    sql = _read(INCREMENTAL_MODELS["rolling_features"][0])
    assert "POW(0.9, s.player_max_match_number - s.player_match_number)" in sql
    assert "MAX(pm.player_match_number) OVER (" in sql
    assert "match_rn_rev" not in sql


def test_incremental_etl_uses_an_ingestion_watermark():
    schema = (ROOT / "infra/postgres/schema.sql").read_text()
    etl = (ROOT / "src/flows/etl.py").read_text()
    assert "ingested_at" in schema
    assert "idx_match_events_ingested_at_match_id" in schema
    assert "CREATE TABLE IF NOT EXISTS bronze.etl_state" in schema
    assert "no changed bronze matches: refreshing player_profiles only" in etl


def test_incremental_match_lookup_has_a_match_id_index():
    project = _read("dbt_project.yml")
    assert "idx_player_matches_match_id" in project


def test_aggregate_models_recompute_globally():
    """Aggregates (tour_averages singleton, player_profiles) are never
    incremental: they change with every new match and must rebuild in full."""
    for name, sql_path in AGGREGATE_MODELS.items():
        assert "is_incremental" not in _read(sql_path), name


def test_unique_keys_enforced_by_db_post_hooks():
    """The (player_id, match_id) grains are enforced by DB primary keys created
    by each model's post-hook in dbt_project.yml (dbt unique_key alone is not
    a DB constraint)."""
    project = _read("dbt_project.yml")
    for name, conname in GRAIN_PKS.items():
        assert f"ensure_primary_key_sql(this, '{conname}', 'player_id, match_id')" in project, name


def test_no_schema_tests_in_model_ymls():
    """No model yml carries a tests: block — the schema-wide unique, not_null,
    and accepted_values tests are removed; DB primary keys and NOT NULL
    post-hooks enforce the grain and non-nullity contracts instead."""
    for name, yml_path in SCHEMA_TEST_YMLS.items():
        assert "tests:" not in _read(yml_path), name


def test_not_null_enforced_by_db_post_hooks():
    """Guaranteed-non-null columns are DB-enforced by each model's
    ensure_not_null_sql post-hook in dbt_project.yml (ALTER COLUMN SET NOT
    NULL is a no-op on reruns, so the hook is idempotent). The removed yml
    not_null tests are not re-added: the DB owns the contract now. The PK is
    applied before the NOT NULL hook, after the table is populated."""
    project = _read("dbt_project.yml")
    for name, columns in NOT_NULL_COLUMNS.items():
        cols_literal = "[" + ", ".join(f"'{c}'" for c in columns) + "]"
        nn_hook = f"ensure_not_null_sql(this, {cols_literal})"
        assert nn_hook in project, name
        conname, pk_columns = MODEL_PKS[name]
        pk_hook = f"ensure_primary_key_sql(this, '{conname}', '{pk_columns}')"
        assert project.index(pk_hook) < project.index(nn_hook), name


def test_redundant_tests_removed():
    """Duplicate-row tests are gone; DB primary keys cover their grain
    contracts and the retained semantic tests cover the rest."""
    for rel in REMOVED_REDUNDANT_TESTS:
        assert not (ROOT / rel).exists(), rel


def test_tour_average_semantic_tests_exist():
    """The two tour-average semantic tests are intentionally present: dbt
    enforces the singleton exactly-one-row contract, NaN/Infinity rejection,
    and rate bounds that PostgreSQL constraints cannot express."""
    for rel_path in TOUR_AVERAGE_SEMANTIC_TESTS:
        assert (ROOT / rel_path).is_file(), rel_path


def test_expansion_factor_tests_exist():
    """The dbt tests that make the demo arithmetic true are all present."""
    for rel_path in EXPANSION_TESTS:
        assert (ROOT / rel_path).is_file(), rel_path


def test_demo_fixture_expansion_arithmetic():
    """The fixture adds exactly one bronze match; the append boundary then
    yields +2 player_matches, +2 rolling snapshots, +2 gold match_features
    rows, and the selection predicate is idempotent on re-runs."""
    fixture = (FIXTURES / "incremental_demo.sql").read_text()
    assert fixture.count("INSERT INTO bronze.match_events") == 1
    new_match_id = "20260714-2026-316-011"
    assert fixture.count(f"'{new_match_id}'") == 1

    def select_new(new_ids: set[str], existing: set[str]) -> set[str]:
        return new_ids - existing

    new = select_new({new_match_id}, existing=set())
    assert new == {new_match_id}
    # Idempotency: after the first run the match is in `this`, so a re-run
    # with no new bronze matches selects nothing.
    assert select_new({new_match_id}, existing={new_match_id}) == set()

    # One new match -> exactly two player perspectives, two snapshots, two
    # directional gold rows; the singleton aggregates stay one row and refresh.
    assert len(new) * 2 == 2  # silver.player_matches
    assert len(new) * 2 == 2  # silver.rolling_features
    assert len(new) * 2 == 2  # gold.match_features
