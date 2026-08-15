"""Hermetic ingest tests with fake network, database, and ATP CSV seams."""

from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

import src.countries as countries_mod
import src.db.ingest as ingest
from src.constants import BRONZE_PROFILES_TABLE
from src.features.columns import BRONZE_COLUMNS


class _FakeCopy:
    """cur.copy() stand-in that records the COPY SQL and accepted rows."""

    def __init__(self, conn, sql):
        self._conn = conn
        self._sql = sql

    def __enter__(self):
        self._conn.statements.append((self._sql, None))
        return self

    def __exit__(self, *_exc):
        return False

    def write_row(self, row):
        self._conn.copied_rows.append(tuple(row))


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql, params=None):
        self._conn.statements.append((sql, params))
        self.rowcount = self._conn.rowcount
        return self

    def executemany(self, sql, seq_of_params):
        self._conn.statements.append((sql, list(seq_of_params)))
        return self

    def copy(self, sql):
        return _FakeCopy(self._conn, sql)

    def fetchall(self):
        return self._conn.fetchall_result


class _FakeTxn:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn.cursor(None)

    def __exit__(self, *_exc):
        return False


class _FakeConn:
    """Minimal psycopg-like connection recording statements and COPY rows.

    rowcount is the value reported after each execute — the fake's stand-in
    for the database's actual affected-row count (default 1: everything
    inserted). Tests exercising skip accounting set it explicitly.
    """

    def __init__(self):
        self.statements: list[tuple[str, object | None]] = []
        self.copied_rows: list[tuple[object, ...]] = []
        self.fetchall_result: list[tuple[object, ...]] = []
        self.rowcount = 1

    def cursor(self, row_factory=None):  # noqa: ARG002 — psycopg cursor API surface
        return _FakeCursor(self)

    def transaction(self):
        return _FakeTxn(self)

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        return _FakeCursor(self)


@pytest.fixture
def fake_ingest_conn(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(ingest, "connection", lambda: nullcontext(conn))
    return conn


# ── _parse_int / _parse_birthdate ─────────────────────────────────


def test_parse_int_parses_valid_and_maps_empty_and_zero_to_null():
    series = pd.Series(["85", "", "0", "70"])
    player_ids = pd.Series(["P1", "P2", "P3", "P4"])

    result = ingest._parse_int(series, "weight", player_ids, low=20, high=300)

    assert result.dtype == pd.Int64Dtype()
    pd.testing.assert_series_equal(result, pd.Series([85, pd.NA, pd.NA, 70], dtype="Int64"))


def test_parse_int_rejects_non_integer_values():
    with pytest.raises(ValueError, match="weight malformed"):
        ingest._parse_int(
            pd.Series(["5.5", "abc"]), "weight", pd.Series(["P1", "P2"]), low=20, high=300
        )


def test_parse_int_rejects_out_of_range():
    with pytest.raises(ValueError, match="weight malformed"):
        ingest._parse_int(pd.Series(["500"]), "weight", pd.Series(["P1"]), low=20, high=300)


def test_parse_birthdate_parses_valid_and_empty_to_null():
    result = ingest._parse_birthdate(pd.Series(["19930316", ""]), pd.Series(["P1", "P2"]))

    assert result.dtype == "datetime64[ns]"
    assert result[0] == pd.Timestamp("1993-03-16")
    assert pd.isna(result[1])


def test_parse_birthdate_rejects_malformed_date():
    with pytest.raises(ValueError, match="birthdate malformed"):
        ingest._parse_birthdate(pd.Series(["20261301"]), pd.Series(["P1"]))


def test_parse_birthdate_rejects_out_of_year():
    with pytest.raises(ValueError, match="birthdate malformed"):
        ingest._parse_birthdate(pd.Series(["18991231"]), pd.Series(["P1"]))


# ── extract_infobox_fields ────────────────────────────────────────


def test_extract_infobox_fields_extracts_known_fields():
    summary = (
        "Jan O'Brien is a tennis player.\n"
        "Plays: Right-handed\n"
        "Backhand: Two-handed\n"
        "Height: 1.85 m\n"
        "Turned pro: 2018\n"
    )

    assert ingest.extract_infobox_fields(summary) == {
        "plays": "Right-handed",
        "backhand": "Two-handed",
        "height": "1.85",
        "turned_pro": "2018",
    }


def test_extract_infobox_fields_omits_missing_fields():
    fields = ingest.extract_infobox_fields("Just a plain summary without infobox fields.")

    assert fields == {}


# ── atp_rows_to_bronze ────────────────────────────────────────────


def _raw_row(match_num=1, tourney_date="20260105", level="G") -> dict[str, object]:
    return {
        "tourney_id": "2026-9900",
        "tourney_date": tourney_date,
        "match_num": match_num,
        "winner_id": "W1",
        "loser_id": "L1",
        "winner_rank": 10,
        "winner_rank_points": 9000,
        "winner_age": "24.41",
        "loser_rank": 20,
        "loser_rank_points": 3000,
        "loser_age": "28.75",
        "tourney_level": level,
        "tourney_name": "Test Open",
        "round": "QF",
        "surface": "Hard",
        "score": "6-4 7-6(4)",
        "indoor": "O",
        "w_ace": 5,
        "w_df": 2,
        "w_svpt": 60,
        "w_1stIn": 40,
        "w_1stWon": 32,
        "w_2ndWon": 15,
        "w_SvGms": 18,
        "w_bpSaved": 3,
        "w_bpFaced": 6,
        "l_ace": 4,
        "l_df": 1,
        "l_svpt": 55,
        "l_1stIn": 35,
        "l_1stWon": 28,
        "l_2ndWon": 12,
        "l_SvGms": 16,
        "l_bpSaved": 2,
        "l_bpFaced": 5,
    }


def test_atp_rows_to_bronze_maps_raw_columns():
    df = ingest.atp_rows_to_bronze([_raw_row()])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["match_id"] == "20260105-2026-9900-001"
    assert row["match_date"] == "2026-01-05"
    assert row["tournament"] == "grand_slam"
    assert row["tournament_name"] == "Test Open"
    assert row["round"] == "qf"
    assert row["surface"] == "hard"
    assert row["score"] == "6-4 7-6"
    assert row["winner_id"] == row["player1_id"]
    assert row["player1_break_points_saved"] == 3
    assert row["player1_break_points_faced"] == 6
    assert row["player2_break_points_saved"] == 2
    assert row["player2_break_points_faced"] == 5
    assert row["player1_first_serve_points_won"] == 32
    assert row["player1_second_serve_points_won"] == 15
    assert row["player1_service_games"] == 18
    assert row["player2_first_serve_points_won"] == 28
    assert row["player2_second_serve_points_won"] == 12
    assert row["player2_service_games"] == 16
    assert row["player1_rank_points"] == 9000
    assert row["player2_rank_points"] == 3000
    assert row["player1_age"] == 24.41  # fractional years preserved, not rounded
    assert row["player2_age"] == 28.75


def test_atp_rows_to_bronze_score_strips_tiebreak_and_null_for_missing():
    def scored(value):
        row = _raw_row()
        row["score"] = value
        return ingest.atp_rows_to_bronze([row]).iloc[0]["score"]

    assert scored("6-4 7-6(4)") == "6-4 7-6"
    assert scored("6-7(5) 7-5 7-6(1)") == "6-7 7-5 7-6"
    assert scored("6-4 6-7(4) RET") == "6-4 6-7 RET"
    assert scored("W/O") == "W/O"
    assert scored(None) is None
    assert scored(0) is None
    assert scored("") is None


def test_atp_rows_to_bronze_float_parses_age_and_defaults_missing_stats():
    row = _raw_row()
    row["winner_age"] = ""
    row["loser_age"] = "abc"  # non-numeric ages default to 0.0 (pandas already
    # normalized real NaN/empty CSV cells to 0 before this transform)
    row["w_1stWon"] = None
    row["winner_rank_points"] = None

    bronze = ingest.atp_rows_to_bronze([row]).iloc[0]

    assert bronze["player1_age"] == 0.0
    assert bronze["player2_age"] == 0.0
    assert bronze["player1_first_serve_points_won"] == 0
    assert bronze["player1_rank_points"] == 0


def test_atp_rows_to_bronze_filters_by_selected_ids():
    rows = [_raw_row(match_num=1), _raw_row(match_num=2)]

    df = ingest.atp_rows_to_bronze(rows, selected_ids={"20260105-2026-9900-002"})

    assert len(df) == 1
    assert df.iloc[0]["match_id"] == "20260105-2026-9900-002"


def test_atp_rows_to_bronze_empty_rows_gives_schema_only():
    df = ingest.atp_rows_to_bronze([])

    assert df.empty
    assert list(df.columns) == list(BRONZE_COLUMNS)


# ── load_raw_atp_rows / load_atp_csv ──────────────────────────────


def _raw_atp_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tourney_id": "T1",
                "tourney_date": "20260101",
                "match_num": 1,
                "winner_id": "W1",
                "loser_id": "L1",
                "winner_rank": 10,
                "winner_rank_points": 9000,
                "winner_age": "24.41",
                "loser_rank": 20,
                "loser_rank_points": 3000,
                "loser_age": "28.75",
                "tourney_level": "G",
                "tourney_name": "Test Open",
                "round": "QF",
                "surface": "Hard",
                "score": "6-4 7-6(4)",
                "indoor": "O",
                "w_ace": 1,
                "w_df": 0,
                "w_svpt": 30,
                "w_1stIn": 20,
                "w_1stWon": 16,
                "w_2ndWon": 8,
                "w_SvGms": 10,
                "w_bpSaved": 1,
                "w_bpFaced": 2,
                "l_ace": 0,
                "l_df": 1,
                "l_svpt": 28,
                "l_1stIn": 18,
                "l_1stWon": 14,
                "l_2ndWon": 7,
                "l_SvGms": 9,
                "l_bpSaved": 0,
                "l_bpFaced": 1,
            },
        ]
    )


def test_load_raw_atp_rows_and_load_atp_csv_reject_missing_columns(tmp_path):
    csv = tmp_path / "missing.csv"
    _raw_atp_df().drop(columns=["w_ace"]).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="missing raw ATP columns"):
        ingest.load_raw_atp_rows(csv)
    with pytest.raises(ValueError, match="missing raw ATP columns"):
        ingest.load_atp_csv(csv)


def test_atp_rows_convert_missing_and_zero_ranks_to_null(tmp_path):
    df = _raw_atp_df()
    null_rank = _raw_atp_df()
    null_rank.loc[0, "winner_rank"] = None
    null_rank.loc[0, "winner_id"] = "W2"
    zero_rank = _raw_atp_df()
    zero_rank.loc[0, "winner_rank"] = 0
    zero_rank.loc[0, "winner_id"] = "W3"
    csv = tmp_path / "ranks.csv"
    pd.concat([df, null_rank, zero_rank], ignore_index=True).to_csv(csv, index=False)

    rows = ingest.atp_rows_to_bronze(ingest.load_raw_atp_rows(csv))

    assert len(rows) == 3
    by_id = rows.set_index("winner_id")
    assert pd.isna(by_id.loc["W2", "player1_ranking"])
    assert pd.isna(by_id.loc["W3", "player1_ranking"])
    assert by_id.loc["W3", "player2_ranking"] == 20


def test_load_raw_atp_rows_missing_indoor_column_fails(tmp_path):
    """The indoor column is required: a CSV without it is a schema error and
    fails before any rows are loaded."""
    csv = tmp_path / "no_indoor.csv"
    _raw_atp_df().drop(columns=["indoor"]).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="missing raw ATP columns"):
        ingest.load_raw_atp_rows(csv)
    with pytest.raises(ValueError, match="missing raw ATP columns"):
        ingest.load_atp_csv(csv)


def test_load_raw_atp_rows_other_required_columns_still_fail(tmp_path):
    """Any other required column continues to fail before any rows are
    written."""
    csv = tmp_path / "missing_other.csv"
    _raw_atp_df().drop(columns=["w_ace"]).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="missing raw ATP columns"):
        ingest.load_raw_atp_rows(csv)


def test_load_raw_atp_rows_preserves_present_indoor_values(tmp_path):
    """Files that do carry indoor keep their I/O values normalized after the
    full load+transform path."""
    csv = tmp_path / "with_indoor.csv"
    _raw_atp_df().to_csv(csv, index=False)

    rows = ingest.load_raw_atp_rows(csv)
    assert rows[0]["indoor"] == "O"
    assert ingest.atp_rows_to_bronze(rows).iloc[0]["is_indoor"] == 0


# ── insert_bronze_rows (bronze idempotency) ─────────────────────────


def test_insert_bronze_rows_copies_valid_rows_with_do_nothing(fake_ingest_conn):
    """INSERT ... SELECT carries ON CONFLICT (match_id) DO NOTHING so
    re-ingesting an existing match_id never overwrites or duplicates it."""
    df = ingest.atp_rows_to_bronze([_raw_row()])

    assert ingest.insert_bronze_rows(df) == 1

    assert fake_ingest_conn.copied_rows  # COPY streamed the valid row
    insert_sql, _params = fake_ingest_conn.statements[-1]
    assert insert_sql.startswith(f"INSERT INTO {ingest.BRONZE_TABLE}")
    assert "ON CONFLICT (match_id) DO NOTHING" in insert_sql
    assert "DO UPDATE" not in insert_sql


def test_insert_bronze_rows_returns_zero_when_all_invalid(fake_ingest_conn):
    df = ingest.atp_rows_to_bronze([_raw_row()])
    df.loc[0, "winner_id"] = "L1"  # winner must equal player1_id

    assert ingest.insert_bronze_rows(df) == 0
    assert fake_ingest_conn.copied_rows == []


def test_insert_bronze_rows_overwrite_uses_do_update(fake_ingest_conn):
    """The seed's overwrite path rewrites an existing match_id instead of
    skipping it; match_id itself is never updated."""
    df = ingest.atp_rows_to_bronze([_raw_row()])

    assert ingest.insert_bronze_rows(df, overwrite=True) == 1

    insert_sql, _params = fake_ingest_conn.statements[-1]
    assert insert_sql.startswith(f"INSERT INTO {ingest.BRONZE_TABLE}")
    assert "ON CONFLICT (match_id) DO UPDATE SET" in insert_sql
    assert "match_id = excluded.match_id" not in insert_sql
    assert "winner_id = excluded.winner_id" in insert_sql


def test_insert_bronze_rows_generic_path_still_do_nothing(fake_ingest_conn):
    """Generic ingestion stays idempotent: overwrite=False is DO NOTHING."""
    ingest.insert_bronze_rows(ingest.atp_rows_to_bronze([_raw_row()]))

    insert_sql, _params = fake_ingest_conn.statements[-1]
    assert "ON CONFLICT (match_id) DO NOTHING" in insert_sql
    assert "DO UPDATE" not in insert_sql


def test_insert_bronze_rows_returns_db_affected_count(fake_ingest_conn):
    """The return value is the database's actual inserted count: when every
    PK already exists the seed reports 0 inserted, not the input row count."""
    df = ingest.atp_rows_to_bronze([_raw_row()])
    fake_ingest_conn.rowcount = 0

    assert ingest.insert_bronze_rows(df) == 0

    # The row was still staged/attempted — the DB just skipped the conflict.
    assert len(fake_ingest_conn.copied_rows) == 1


# ── search_wikipedia / fetch_summary ──────────────────────────────


class _FakeResponse:
    """requests.get() stand-in exposing only .json()."""

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


def _patch_wiki_get(monkeypatch, payload: object) -> None:
    monkeypatch.setattr(ingest.requests, "get", lambda *_args, **_kwargs: _FakeResponse(payload))


def test_search_wikipedia_returns_first_page_title(monkeypatch):
    _patch_wiki_get(
        monkeypatch, {"query": {"search": [{"title": "Jan O'Brien"}, {"title": "Other"}]}}
    )

    assert ingest.search_wikipedia("Jan O'Brien") == "Jan O'Brien"


def test_search_wikipedia_returns_none_without_pages(monkeypatch):
    _patch_wiki_get(monkeypatch, {"query": {"search": []}})

    assert ingest.search_wikipedia("Nobody") is None


def test_search_wikipedia_rejects_tournament_result_for_player(monkeypatch):
    _patch_wiki_get(
        monkeypatch,
        {
            "query": {
                "search": [
                    {"title": "2026 Qatar ExxonMobil Open"},
                    {"title": "Carlos Alcaraz"},
                ]
            }
        },
    )

    assert ingest.search_wikipedia("Carlos Alcaraz") == "Carlos Alcaraz"


def test_search_wikipedia_matches_surname_first_title(monkeypatch):
    """Surname-first article titles match given-first player names (fallback)."""
    _patch_wiki_get(
        monkeypatch, {"query": {"search": [{"title": "Wu Yibing"}, {"title": "Other"}]}}
    )

    assert ingest.search_wikipedia("Yibing Wu") == "Wu Yibing"


def test_resolve_ranking_identities_matches_reversed_name_order():
    """Surname-first source names match given-first canonical names (fallback)."""
    source_names = {"212275": "Wu Yibing"}
    canonical = {"Y0001": "Yibing Wu"}

    resolved = ingest.resolve_ranking_identities({"212275"}, source_names, canonical)

    assert resolved == {"212275": "Y0001"}


def test_clean_bio_paragraph_truncates_at_last_period(monkeypatch):
    monkeypatch.setattr(ingest, "SUMMARY_MAX_CHARS", 28)

    assert (
        ingest.clean_bio_paragraph("First sentence. Second sentence is too long.")
        == "First sentence."
    )


def test_fetch_summary_returns_page_dict(monkeypatch):
    _patch_wiki_get(
        monkeypatch,
        {"query": {"pages": {"42": {"title": "Jan O'Brien", "extract": "Summary text."}}}},
    )

    assert ingest.fetch_summary("Jan O'Brien") == {
        "title": "Jan O'Brien",
        "summary": "Summary text.",
        "page_id": "42",
    }


def test_fetch_summary_returns_none_for_missing_page(monkeypatch):
    _patch_wiki_get(monkeypatch, {"query": {"pages": {"-1": {"title": "Missing", "extract": ""}}}})

    assert ingest.fetch_summary("Missing") is None


# ── load_profiles_for (seed status line) ─────────────────────────


def test_load_profiles_for_reports_inserted_and_skipped(monkeypatch, tmp_path, capsys):
    """The seed status line reports actual inserted vs skipped existing."""
    csv = tmp_path / "atp.csv"
    csv.write_text("x")
    monkeypatch.setattr(ingest, "ATP_DATABASE_CSV", csv)
    monkeypatch.setattr(ingest, "load_atp_profiles", lambda _path, **kwargs: 2)  # noqa: ARG005

    assert ingest.load_profiles_for(["P1", "P2", "P3"], "seeded") == 2

    out = capsys.readouterr().out
    assert "Loaded 2 player profiles for 3 seeded players (1 skipped existing)" in out


def test_load_profiles_for_force_prints_overwrite(monkeypatch, tmp_path, capsys):
    csv = tmp_path / "atp.csv"
    csv.write_text("x")
    monkeypatch.setattr(ingest, "ATP_DATABASE_CSV", csv)
    monkeypatch.setattr(ingest, "load_atp_profiles", lambda _path, **kwargs: 3)  # noqa: ARG005

    ingest.load_profiles_for(["P1", "P2", "P3"], "seeded", force=True)

    out = capsys.readouterr().out
    assert "Loaded 3 player profiles for 3 seeded players (overwrite)" in out


# ── load_atp_profiles ─────────────────────────────────────────────


def _write_profiles_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "atp_profiles.csv"
    pd.DataFrame(
        [
            {
                "id": "P1",
                "player": "Player One",
                "atpname": "P. One",
                "birthdate": "19930316",
                "weight": "85",
                "height": "185",
                "turnedpro": "2018",
                "birthplace": "Paris",
                "coaches": "",
                "hand": "R",
                "backhand": "2H",
                "ioc": "FRA",
            },
            {
                "id": "P2",
                "player": "Player Two",
                "atpname": "P. Two",
                "birthdate": "19850402",
                "weight": "80",
                "height": "190",
                "turnedpro": "2010",
                "birthplace": "London",
                "coaches": "Coach B",
                "hand": "L",
                "backhand": "1H",
                "ioc": "GBR",
            },
        ]
    ).to_csv(csv, index=False)
    return csv


def test_load_atp_profiles_bulk_copies_base_columns(fake_ingest_conn, tmp_path):
    csv = _write_profiles_csv(tmp_path)
    fake_ingest_conn.rowcount = 2

    assert ingest.load_atp_profiles(csv) == 2

    # COPY streams the rows into a temp stage, then a single INSERT ... SELECT
    # applies the idempotent PK insert (DO NOTHING without --force) — never a
    # DuckDB relation scan.
    assert any(
        sql == f"COPY stage ({', '.join(ingest.ATP_PROFILE_COLUMNS)}) FROM STDIN"
        for sql, _ in fake_ingest_conn.statements
    )
    assert len(fake_ingest_conn.copied_rows) == 2
    insert_sql, _params = fake_ingest_conn.statements[-1]
    assert insert_sql.startswith(f"INSERT INTO {ingest.BRONZE_PROFILES_TABLE}")
    assert "ON CONFLICT (player_id) DO NOTHING" in insert_sql


def test_load_atp_profiles_default_never_overwrites(fake_ingest_conn, tmp_path):
    """Without --force the profile load skips existing player_ids, never updates."""
    ingest.load_atp_profiles(_write_profiles_csv(tmp_path))

    assert not any("DO UPDATE SET" in s for s, _ in fake_ingest_conn.statements)


def test_load_atp_profiles_force_upsert_never_touches_enrichment(fake_ingest_conn, tmp_path):
    """--force overwrites ATP identity fields but preserves enrichment fields."""
    csv = _write_profiles_csv(tmp_path)
    ingest.load_atp_profiles(csv, force=True)

    update_sql = next(s for s, _ in fake_ingest_conn.statements if "DO UPDATE SET" in s)
    assert "ON CONFLICT (player_id) DO UPDATE SET" in update_sql
    for col in ("summary", "enriched_at"):
        assert col not in update_sql
    for col in ("weight", "height", "birthplace", "ioc"):
        assert f"{col} = excluded.{col}" in update_sql


def test_load_atp_profiles_filters_by_player_ids(fake_ingest_conn, tmp_path):
    csv = _write_profiles_csv(tmp_path)

    assert ingest.load_atp_profiles(csv, player_ids={"P2"}) == 1

    assert len(fake_ingest_conn.copied_rows) == 1
    assert fake_ingest_conn.copied_rows[0][0] == "P2"


def test_load_atp_profiles_dedupes_duplicate_ids_last_wins(fake_ingest_conn, tmp_path):
    csv = tmp_path / "dup.csv"
    pd.DataFrame(
        [
            {
                "id": "P1",
                "player": "First",
                "atpname": "",
                "birthdate": "19930101",
                "weight": "80",
                "height": "180",
                "turnedpro": "2015",
                "birthplace": "",
                "coaches": "",
                "hand": "R",
                "backhand": "2H",
                "ioc": "FRA",
            },
            {
                "id": "P1",
                "player": "Last",
                "atpname": "",
                "birthdate": "19930101",
                "weight": "90",
                "height": "180",
                "turnedpro": "2015",
                "birthplace": "",
                "coaches": "",
                "hand": "R",
                "backhand": "2H",
                "ioc": "FRA",
            },
        ]
    ).to_csv(csv, index=False)

    assert ingest.load_atp_profiles(csv) == 1

    assert len(fake_ingest_conn.copied_rows) == 1
    assert fake_ingest_conn.copied_rows[0][1] == "Last"
    assert fake_ingest_conn.copied_rows[0][4] == 90


# ── enrich_player / enrich_players / enrich_missing ───────────────


def test_enrich_player_apostrophe_name_binds_as_param(monkeypatch, fake_ingest_conn):
    monkeypatch.setattr(ingest, "search_wikipedia", lambda _name: "Jan O'Brien")
    monkeypatch.setattr(
        ingest,
        "fetch_summary",
        lambda _title: {
            "title": "Jan O'Brien",
            "summary": "Jan O'Brien is a player with an apostrophe.",
            "page_id": "1",
        },
    )

    assert ingest.enrich_player("Jan O'Brien") is True

    sql, params = fake_ingest_conn.statements[0]
    assert "%s" in sql  # psycopg placeholders, never string interpolation
    assert "O'Brien" not in sql
    assert "UPDATE" in sql
    assert "SET summary = %s" in sql
    # Enrichment writes bronze metadata only — never gold (dbt-derived).
    assert "UPDATE bronze.player_profiles" in sql
    assert "gold.player_profiles" not in sql
    assert len(params) == 2  # summary_text, player_id
    assert params[1] == "Jan O'Brien"  # player_id bound as param


def test_enrich_player_uses_explicit_player_id(monkeypatch, fake_ingest_conn):
    monkeypatch.setattr(ingest, "search_wikipedia", lambda _name: "Some Title")
    monkeypatch.setattr(
        ingest,
        "fetch_summary",
        lambda _title: {"title": "Some Title", "summary": "Bio text.", "page_id": "7"},
    )

    assert ingest.enrich_player("Name", "REALID") is True

    _sql, params = fake_ingest_conn.statements[0]
    assert params[1] == "REALID"


# ── extract_playing_style_paragraph / extract_lead_paragraph ──────


def test_extract_playing_style_paragraph_prefers_style_section():
    summary = (
        "Lead paragraph of the article.\n\n"
        "== Player profile ==\n\n"
        "=== Playing style ===\n"
        "Aggressive baseliner with a powerful serve.\n\n"
        "== Career ==\n\n"
        "Won many titles."
    )
    assert ingest.extract_playing_style_paragraph(summary) == (
        "Aggressive baseliner with a powerful serve."
    )


def test_extract_playing_style_returns_none_when_section_absent():
    summary = "Lead paragraph.\n\n== Career ==\nWon many titles."
    assert ingest.extract_playing_style_paragraph(summary) is None


def test_extract_playing_style_returns_none_when_section_empty():
    summary = "Lead paragraph.\n\n=== Playing style ===\n\n== Career ==\n"
    assert ingest.extract_playing_style_paragraph(summary) is None


def test_extract_lead_paragraph_returns_first_paragraph():
    summary = "First paragraph.\n\nSecond paragraph.\n\n== Career ==\n"
    assert ingest.extract_lead_paragraph(summary) == "First paragraph."


def test_extract_lead_paragraph_returns_none_when_empty():
    assert ingest.extract_lead_paragraph("== Career ==\n") is None


def test_enrich_player_stores_playing_style_paragraph(monkeypatch, fake_ingest_conn):
    """Playing style paragraph is preferred over the article lead."""
    monkeypatch.setattr(ingest, "search_wikipedia", lambda _name: "Title")
    monkeypatch.setattr(
        ingest,
        "fetch_summary",
        lambda _title: {
            "title": "Title",
            "page_id": "1",
            "summary": (
                "Lead paragraph.\n\n=== Playing style ===\nStyle paragraph.\n\n== Career ==\n"
            ),
        },
    )

    assert ingest.enrich_player("Player") is True

    _sql, params = fake_ingest_conn.statements[0]
    assert params[0] == "Style paragraph."


def test_enrich_player_falls_back_to_lead_paragraph(monkeypatch, fake_ingest_conn):
    """When Playing style is absent, the article lead paragraph is used."""
    monkeypatch.setattr(ingest, "search_wikipedia", lambda _name: "Title")
    monkeypatch.setattr(
        ingest,
        "fetch_summary",
        lambda _title: {
            "title": "Title",
            "page_id": "1",
            "summary": "Lead paragraph.\n\n== Career ==\n",
        },
    )

    assert ingest.enrich_player("Player") is True

    _sql, params = fake_ingest_conn.statements[0]
    assert params[0] == "Lead paragraph."


def test_enrich_player_skips_when_no_usable_paragraph(monkeypatch, fake_ingest_conn):
    """No Playing style section and no lead paragraph -> SKIP, not counted."""
    monkeypatch.setattr(ingest, "search_wikipedia", lambda _name: "Title")
    monkeypatch.setattr(
        ingest,
        "fetch_summary",
        lambda _title: {
            "title": "Title",
            "page_id": "1",
            "summary": "  \n\n== Career ==\n",
        },
    )

    assert ingest.enrich_player("Player") is False
    assert fake_ingest_conn.statements == []


def test_enrich_missing_enriches_missing_players(monkeypatch):
    monkeypatch.setattr(ingest, "get_players_without_summary", lambda: ["X", "Y"])
    monkeypatch.setattr(ingest, "enrich_players", lambda _ids: 2)

    assert ingest.enrich_missing() == 2


def test_enrich_missing_noop_when_none_missing(monkeypatch):
    monkeypatch.setattr(ingest, "get_players_without_summary", lambda: [])

    assert ingest.enrich_missing() == 0


def test_enrich_players_skips_already_enriched_and_nameless(monkeypatch, fake_ingest_conn):
    fake_ingest_conn.fetchall_result = [
        ("P1", "Player One", None),
        ("P2", "Player Two", "Already enriched."),
        ("P3", None, None),
    ]
    fetched: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ingest,
        "_fetch_wiki_bio",
        lambda name, pid: fetched.append((pid, name)) or ("Bio for " + name, "Title"),
    )

    assert ingest.enrich_players(["P1", "P2", "P3"]) == 1
    # Only the not-yet-enriched, named profile is fetched and written.
    assert fetched == [("P1", "Player One")]
    update_sqls = [s for s, _ in fake_ingest_conn.statements if "UPDATE" in s]
    assert len(update_sqls) == 1

    # The lookup must use %s placeholders with the ids bound as parameters,
    # and read from bronze metadata only — never gold (dbt-derived).
    sql, params = fake_ingest_conn.statements[0]
    assert "%s" in sql
    assert "FROM bronze.player_profiles" in sql
    assert "gold.player_profiles" not in sql
    assert params == ["P1", "P2", "P3"]


def test_enrich_players_reports_success_failure_and_error_counts(
    capsys, monkeypatch, fake_ingest_conn
):
    """Batch enrichment counts OK/no-bio/exception outcomes; no-bio and
    exceptions both count as failed, never skipped."""
    fake_ingest_conn.fetchall_result = [
        ("P1", "Player One", None),
        ("P2", "Player Two", None),
        ("P3", "Player Three", None),
    ]

    def fake_fetch(name, pid):
        if pid == "P1":
            return ("Bio for " + name, "Title")
        if pid == "P2":
            print(f"  SKIP {pid}: no Wikipedia match for {name!r}")
            return None
        raise RuntimeError("Wikipedia API timeout")

    monkeypatch.setattr(ingest, "_fetch_wiki_bio", fake_fetch)

    assert ingest.enrich_players(["P1", "P2", "P3"]) == 1

    out = capsys.readouterr().out
    # Per-player lines survive for failed players (no-bio SKIP, exception ERROR).
    assert "  SKIP P2: no Wikipedia match for 'Player Two'" in out
    assert "  ERROR P3 (Player Three): Wikipedia API timeout" in out
    # 2 of 3 attempts failed (no match + exception); nothing was "skipped".
    assert (
        "Enrichment summary: 3 attempted, 0 already enriched, 0 no name, 1 enriched, 2 failed"
        in out
    )


def test_enrich_players_counts_precheck_skips_in_summary_only(
    capsys, monkeypatch, fake_ingest_conn
):
    """Already-enriched and no-name profiles are pre-skips: no per-player
    lines, counted in the summary, and excluded from attempted enrichment."""
    fake_ingest_conn.fetchall_result = [
        ("P1", "Player One", None),
        ("P2", "Player Two", "Already enriched."),
        ("P3", None, None),
    ]
    monkeypatch.setattr(ingest, "_fetch_wiki_bio", lambda _name, _pid: ("Bio", "Title"))

    assert ingest.enrich_players(["P1", "P2", "P3"]) == 1

    out = capsys.readouterr().out
    assert "  SKIP P2: already enriched" not in out
    assert "  SKIP P3: no profile name for enrichment" not in out
    # Only P1 was attempted; the other two are pre-skips in the summary.
    assert (
        "Enrichment summary: 1 attempted, 1 already enriched, 1 no name, 1 enriched, 0 failed"
        in out
    )


def test_enrich_players_all_no_match_players_count_as_failed(capsys, monkeypatch, fake_ingest_conn):
    """7 no-match players report 7 attempted and 7 failed — never skipped."""
    fake_ingest_conn.fetchall_result = [(f"P{i}", f"Player {i}", None) for i in range(1, 8)]
    monkeypatch.setattr(ingest, "_fetch_wiki_bio", lambda _name, _pid: None)

    assert ingest.enrich_players([f"P{i}" for i in range(1, 8)]) == 0

    out = capsys.readouterr().out
    assert (
        "Enrichment summary: 7 attempted, 0 already enriched, 0 no name, 0 enriched, 7 failed"
        in out
    )


def test_enrich_players_never_overwrites_existing_summaries(monkeypatch, fake_ingest_conn):
    """Enrichment is idempotent: profiles with a summary are never re-fetched
    or overwritten — only nameless profiles are skipped before the fetch."""
    fake_ingest_conn.fetchall_result = [
        ("P1", "Player One", "Old summary."),
        ("P2", "Player Two", "Another old summary."),
    ]
    fetched: list[str] = []
    monkeypatch.setattr(
        ingest, "_fetch_wiki_bio", lambda _name, pid: fetched.append(pid) or ("Bio", "Title")
    )

    assert ingest.enrich_players(["P1", "P2"]) == 0
    assert fetched == []
    assert not [s for s, _ in fake_ingest_conn.statements if "UPDATE" in s]
    # The lookup still reads bronze metadata only, with ids bound as params.
    sql, params = fake_ingest_conn.statements[0]
    assert "FROM bronze.player_profiles" in sql
    assert params == ["P1", "P2"]


# ── Indoor normalization ────────────────────────────────────────


def test_normalize_indoor_I_is_1():
    assert ingest._normalize_indoor("I") == 1


def test_normalize_indoor_O_is_0():
    assert ingest._normalize_indoor("O") == 0


def test_normalize_indoor_empty_is_none():
    assert ingest._normalize_indoor("") is None


def test_normalize_indoor_nan_is_none():
    import math

    assert ingest._normalize_indoor(float("nan")) is None


def test_normalize_indoor_none_is_none():
    assert ingest._normalize_indoor(None) is None


def test_atp_rows_to_bronze_indoor_I_maps_to_1():
    row = _raw_row()
    row["indoor"] = "I"
    df = ingest.atp_rows_to_bronze([row])
    assert df.iloc[0]["is_indoor"] == 1


def test_atp_rows_to_bronze_indoor_O_maps_to_0():
    row = _raw_row()
    row["indoor"] = "O"
    df = ingest.atp_rows_to_bronze([row])
    assert df.iloc[0]["is_indoor"] == 0


def test_atp_rows_to_bronze_indoor_missing_maps_to_nan():
    row = _raw_row()
    row["indoor"] = None
    df = ingest.atp_rows_to_bronze([row])
    assert pd.isna(df.iloc[0]["is_indoor"])


# ── IOC country reference (src/countries) ─────────────────────────


def test_country_reference_is_well_formed():
    """The reference mapping includes UNK, has no duplicate codes, and every
    non-UNK row carries a usable ISO alpha-2 code."""
    countries = countries_mod._COUNTRIES

    assert countries_mod.UNK in countries
    assert countries_mod.resolve_ioc(countries_mod.UNK) == ("", "Country unknown")
    assert all(iso2 or code == countries_mod.UNK for code, (iso2, _) in countries.items())


def test_normalize_ioc_trims_and_uppercases():
    assert countries_mod.normalize_ioc(" fra ") == "FRA"
    assert countries_mod.normalize_ioc("gbr") == "GBR"


def test_normalize_ioc_empty_or_none_becomes_unk():
    assert countries_mod.normalize_ioc(None) == countries_mod.UNK
    assert countries_mod.normalize_ioc("") == countries_mod.UNK
    assert countries_mod.normalize_ioc("   ") == countries_mod.UNK


def test_valid_ioc_preserves_known_and_falls_back_to_unk():
    assert countries_mod.valid_ioc(" FRA ") == "FRA"  # trimmed/uppercased
    assert countries_mod.valid_ioc("gbr") == "GBR"
    assert countries_mod.valid_ioc("UNK") == "UNK"
    for bad in (None, "", "  ", "XYZ", "URS", float("nan")):
        assert countries_mod.valid_ioc(bad) == "UNK"


def test_resolve_ioc_known_codes():
    assert countries_mod.resolve_ioc("FRA") == ("FR", "France")
    assert countries_mod.resolve_ioc("USA") == ("US", "United States")


def test_resolve_ioc_unknown_falls_back_to_unk_row():
    unknown = ("", "Country unknown")
    assert countries_mod.resolve_ioc("XYZ") == unknown
    assert countries_mod.resolve_ioc("") == unknown
    assert countries_mod.resolve_ioc("UNK") == unknown


def test_is_known_ioc():
    assert countries_mod.is_known_ioc("fra")
    assert not countries_mod.is_known_ioc("XYZ")


# ── IOC normalization in load_atp_profiles ────────────────────────


def test_load_atp_profiles_normalizes_ioc_and_reports_unresolved(
    fake_ingest_conn, tmp_path, capsys
):
    """Valid IOC values are preserved (trimmed/uppercased); missing/invalid
    values become UNK and the unresolved count is reported."""
    csv = tmp_path / "ioc.csv"
    pd.DataFrame(
        [
            {
                "id": f"P{i}",
                "player": f"Player {i}",
                "atpname": "",
                "birthdate": "19930101",
                "weight": "80",
                "height": "180",
                "turnedpro": "2015",
                "birthplace": "",
                "coaches": "",
                "hand": "R",
                "backhand": "2H",
                "ioc": ioc,
            }
            for i, ioc in enumerate([" FRA ", "", "XYZ", "gbr"])
        ]
    ).to_csv(csv, index=False)

    fake_ingest_conn.rowcount = 4
    assert ingest.load_atp_profiles(csv) == 4

    copied = {row[0]: row[11] for row in fake_ingest_conn.copied_rows}
    assert copied == {"P0": "FRA", "P1": "UNK", "P2": "UNK", "P3": "GBR"}
    assert "IOC: 2/4 profiles unresolved (missing/invalid -> UNK)" in capsys.readouterr().out


def test_load_atp_profiles_reports_nothing_when_all_ioc_resolve(fake_ingest_conn, tmp_path, capsys):
    csv = _write_profiles_csv(tmp_path)  # P1=FRA, P2=GBR, both known

    ingest.load_atp_profiles(csv)

    assert "IOC:" not in capsys.readouterr().out
    assert {row[0]: row[11] for row in fake_ingest_conn.copied_rows} == {
        "P1": "FRA",
        "P2": "GBR",
    }


# ── Ranking identity map contract ────────────────────────────────


def _write_map_csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    csv = tmp_path / "ranking_player_map.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_canonical_players_loads_id_to_name_reference():
    players = ingest.canonical_players()

    assert isinstance(players, dict)
    assert "A0E2" in players  # canonical ATP_Database id for Carlos Alcaraz
    assert players["A0E2"] == "Carlos Alcaraz"
    assert all(pid and name for pid, name in players.items())


def test_load_ranking_player_map_returns_source_to_canonical(tmp_path):
    csv = _write_map_csv(
        tmp_path,
        [
            {"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"},
            {
                "ranking_player_id": "100644",
                "ranking_name": "Alexander Zverev",
                "player_id": "Z355",
            },
        ],
    )

    result = ingest.load_ranking_player_map(csv, canonical_ids={"A0E2", "Z355"})

    assert result == {"207989": "A0E2", "100644": "Z355"}


def test_load_ranking_player_map_rejects_missing_column(tmp_path):
    csv = _write_map_csv(tmp_path, [{"ranking_player_id": "207989", "player_id": "A0E2"}])

    with pytest.raises(ValueError, match="missing columns"):
        ingest.load_ranking_player_map(csv, canonical_ids={"A0E2"})


def test_load_ranking_player_map_rejects_empty_required_cell(tmp_path):
    csv = _write_map_csv(
        tmp_path,
        [
            {"ranking_player_id": "207989", "ranking_name": "", "player_id": "A0E2"},
        ],
    )

    with pytest.raises(ValueError, match="empty ranking_name"):
        ingest.load_ranking_player_map(csv, canonical_ids={"A0E2"})


def test_load_ranking_player_map_rejects_duplicate_source_id(tmp_path):
    csv = _write_map_csv(
        tmp_path,
        [
            {"ranking_player_id": "207989", "ranking_name": "A", "player_id": "A0E2"},
            {"ranking_player_id": "207989", "ranking_name": "B", "player_id": "Z355"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate ranking source ids"):
        ingest.load_ranking_player_map(csv, canonical_ids={"A0E2", "Z355"})


def test_load_ranking_player_map_allows_multiple_sources_to_same_target(tmp_path):
    """Multiple ranking source ids may legitimately map to the same canonical id."""
    csv = _write_map_csv(
        tmp_path,
        [
            {"ranking_player_id": "207989", "ranking_name": "A", "player_id": "A0E2"},
            {"ranking_player_id": "100644", "ranking_name": "B", "player_id": "A0E2"},
        ],
    )

    result = ingest.load_ranking_player_map(csv, canonical_ids={"A0E2"})
    assert result == {"207989": "A0E2", "100644": "A0E2"}


def test_load_ranking_player_map_rejects_unknown_canonical_id(tmp_path):
    csv = _write_map_csv(
        tmp_path,
        [
            {"ranking_player_id": "207989", "ranking_name": "A", "player_id": "NOT_A_PLAYER"},
        ],
    )

    with pytest.raises(ValueError, match="unknown canonical player ids"):
        ingest.load_ranking_player_map(csv, canonical_ids={"A0E2"})


def test_load_ranking_player_map_defaults_canonical_ids_from_reference(tmp_path):
    """Omitting canonical_ids validates targets against the canonical reference."""
    csv = _write_map_csv(
        tmp_path,
        [
            {"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"},
        ],
    )

    assert ingest.load_ranking_player_map(csv) == {"207989": "A0E2"}


def test_committed_ranking_player_map_is_valid_and_deterministic():
    """The reviewed map file loads cleanly and pins the approved source id."""
    result = ingest.load_ranking_player_map()

    assert "207989" in result
    assert result["207989"] == "A0E2"  # Alcaraz source id -> canonical id
    assert ingest.load_ranking_player_map() == result  # deterministic


def test_ranking_name_candidates_reports_ambiguous_and_exact():
    ranking_rows = [
        {"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz"},
        {"ranking_player_id": "100000", "ranking_name": "Chris Lewis"},
        {"ranking_player_id": "777777", "ranking_name": "Nobody Here"},
    ]
    canonical = {
        "A0E2": "Carlos Alcaraz",
        "L024": "Chris Lewis",
        "L639": "Chris Lewis",
        "Z999": "Someone Else",
    }

    report = ingest.ranking_name_candidates(ranking_rows, canonical=canonical)

    by_src = {r["ranking_player_id"]: r for r in report}
    assert by_src["207989"] == {
        "ranking_player_id": "207989",
        "ranking_name": "Carlos Alcaraz",
        "exact_candidates": ["A0E2"],
        "normalized_candidates": ["A0E2"],
        "ambiguous": False,
    }
    assert by_src["100000"]["ambiguous"] is True
    assert by_src["100000"]["normalized_candidates"] == ["L024", "L639"]
    assert by_src["777777"]["ambiguous"] is False
    assert by_src["777777"]["normalized_candidates"] == []


def test_ranking_name_candidates_is_deterministic_and_sorted():
    ranking_rows = [{"ranking_player_id": "200000", "ranking_name": "J. Smith"}]
    canonical = {"S1": "J Smith", "S2": "John Smith"}

    first = ingest.ranking_name_candidates(ranking_rows, canonical=canonical)
    second = ingest.ranking_name_candidates(ranking_rows, canonical=canonical)

    assert first == second


def test_ranking_name_candidates_normalizes_accents_and_punctuation():
    ranking_rows = [{"ranking_player_id": "300000", "ranking_name": "Daniil Medvedev"}]
    canonical = {"M001": "Daniil  Medvedev", "M002": "Daniil Medvedev-Something"}

    report = ingest.ranking_name_candidates(ranking_rows, canonical=canonical)

    assert report[0]["exact_candidates"] == []  # exact name differs
    assert report[0]["normalized_candidates"] == ["M001"]  # whitespace collapses
    assert report[0]["ambiguous"] is False


def test_unmapped_ranking_rows_reports_id_name_count():
    rank_map = {"207989": "A0E2"}
    rows = [
        {"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz"},
        {"ranking_player_id": "100000", "ranking_name": "Chris Lewis"},
        {"ranking_player_id": "100000", "ranking_name": "Chris Lewis"},
        {"ranking_player_id": "100001", "ranking_name": "Another"},
    ]

    report = ingest.unmapped_ranking_rows(rows, rank_map)

    assert report == [
        {"ranking_player_id": "100000", "ranking_name": "Chris Lewis", "count": 2},
        {"ranking_player_id": "100001", "ranking_name": "Another", "count": 1},
    ]


# ── Official rankings ingest ───────────────────────────────────


def _write_ranking_csv(tmp_path: Path, name: str, rows: list[list[object]]) -> Path:
    csv = tmp_path / name
    pd.DataFrame(rows, columns=ingest.RANKINGS_COLUMNS).to_csv(csv, index=False)
    return csv


def _write_players_csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    csv = tmp_path / "atp_players.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_discover_ranking_csvs_finds_only_atp_rankings_files(tmp_path):
    import csv

    def write(name):
        with open(tmp_path / name, "w") as f:
            csv.writer(f).writerow(["header"])

    write("atp_rankings_00s.csv")
    write("atp_rankings_10s.csv")
    write("atp_rankings_current.csv")
    write("atp_players.csv")
    write("notes.txt")
    write(".DS_Store")

    found = ingest.discover_ranking_csvs(tmp_path)

    assert [p.name for p in found] == [
        "atp_rankings_00s.csv",
        "atp_rankings_10s.csv",
        "atp_rankings_current.csv",
    ]


def test_discover_ranking_csvs_empty(tmp_path):
    assert ingest.discover_ranking_csvs(tmp_path) == []


def test_load_ranking_rows_combines_files_and_empty_points_null(tmp_path):
    _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, "207989", "12050"]])
    _write_ranking_csv(
        tmp_path,
        "atp_rankings_10s.csv",
        [
            ["19730827", 129, "100005", ""],
            ["19730827", 200, "100011", ""],
        ],
    )

    rows = ingest.load_ranking_rows(ingest.discover_ranking_csvs(tmp_path))

    assert len(rows) == 3
    assert rows["rank"].tolist() == [1, 129, 200]
    assert rows["player_id"].tolist() == ["207989", "100005", "100011"]
    assert rows["points"].iloc[0] == 12050
    assert rows["points"].isna().sum() == 2  # empty points -> NULL


def test_load_ranking_rows_rejects_missing_column(tmp_path):
    csv = tmp_path / "atp_rankings_00s.csv"
    pd.DataFrame([["20260105", 1, "207989"]], columns=["ranking_date", "rank", "player"]).to_csv(
        csv, index=False
    )

    with pytest.raises(ValueError, match="expected columns"):
        ingest.load_ranking_rows([csv])


def test_load_ranking_rows_rejects_extra_column(tmp_path):
    csv = tmp_path / "atp_rankings_00s.csv"
    pd.DataFrame(
        [["20260105", 1, "207989", "12050", "extra"]],
        columns=["ranking_date", "rank", "player", "points", "extra"],
    ).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="expected columns"):
        ingest.load_ranking_rows([csv])


def test_load_ranking_rows_rejects_malformed_date(tmp_path):
    csv = _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20261301", 1, "207989", "1"]])

    with pytest.raises(ValueError, match="ranking_date malformed"):
        ingest.load_ranking_rows([csv])


@pytest.mark.parametrize("rank", ["abc", "0", "-3", ""])
def test_load_ranking_rows_rejects_malformed_rank(tmp_path, rank):
    csv = _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", rank, "207989", "1"]])

    with pytest.raises(ValueError, match="rank malformed"):
        ingest.load_ranking_rows([csv])


@pytest.mark.parametrize("player", ["abc", ""])
def test_load_ranking_rows_rejects_malformed_player(tmp_path, player):
    csv = _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, player, "1"]])

    with pytest.raises(ValueError, match="player malformed"):
        ingest.load_ranking_rows([csv])


def test_load_ranking_rows_rejects_malformed_points(tmp_path):
    csv = _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, "207989", "abc"]])

    with pytest.raises(ValueError, match="points malformed"):
        ingest.load_ranking_rows([csv])


def test_load_ranking_rows_filters_rank_limit_before_validation(tmp_path):
    """rank_limit drops rows before validation, so malformed rows outside the
    requested rank scope never raise."""
    _write_ranking_csv(
        tmp_path,
        "atp_rankings_00s.csv",
        [
            ["20260105", 1, "207989", "12050"],  # kept
            ["20260105", 150, "207989", "1000"],  # kept
            ["20260105", 201, "207989", "500"],  # dropped (rank > 200)
            ["20260105", 300, "999999", "abc"],  # dropped before points validation
        ],
    )

    rows = ingest.load_ranking_rows(ingest.discover_ranking_csvs(tmp_path), rank_limit=200)

    assert len(rows) == 2
    assert rows["rank"].tolist() == [1, 150]
    assert (rows["player_id"] == "207989").all()


def test_ingest_rankings_upserts_only_mapped_canonical_ids(fake_ingest_conn, tmp_path):
    """Raw ranking source ids never reach the table: only canonical ids from the
    approved map are copied, and unmapped top-200 rows are skipped."""
    _write_ranking_csv(
        tmp_path,
        "atp_rankings_00s.csv",
        [
            ["20260105", 1, "207989", "12050"],  # mapped -> A0E2
            ["20260105", 2, "999999", "10000"],  # unmapped -> skipped
            ["20260105", 300, "207989", "5000"],  # out of top-200 -> dropped
        ],
    )
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    _write_players_csv(
        tmp_path,
        [
            {"player_id": "207989", "name_first": "Carlos", "name_last": "Alcaraz", "ioc": "ESP"},
            {"player_id": "999999", "name_first": "Nobody", "name_last": "Here", "ioc": "XYZ"},
        ],
    )

    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
    )

    assert summary == {
        "files": 1,
        "source_rows": 2,  # the rank-300 row is skipped before validation
        "top200": 2,
        "upserted": 1,
        "skipped_existing": 0,
        "unmapped": 1,
        "auto_mapped": 0,
        "unresolved": 1,
    }
    # Only the single mapped top-200 row is copied; the raw source id 999999
    # and the rank-300 row never reach the table.
    assert len(fake_ingest_conn.copied_rows) == 1
    row = fake_ingest_conn.copied_rows[0]
    assert str(row[0]) == "2026-01-05"
    assert row[1] == "A0E2"
    assert int(row[2]) == 1
    assert int(row[3]) == 12050


def test_ingest_rankings_top200_boundary_keeps_200_drops_201(fake_ingest_conn, tmp_path):
    """Rank exactly 200 imports; rank 201 is the first filtered-out rank."""
    _write_ranking_csv(
        tmp_path,
        "atp_rankings_00s.csv",
        [
            ["20260105", 1, "207989", "12050"],
            ["20260105", 200, "100644", "50"],
            ["20260105", 201, "100644", "40"],
        ],
    )
    _write_map_csv(
        tmp_path,
        [
            {"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"},
            {
                "ranking_player_id": "100644",
                "ranking_name": "Alexander Zverev",
                "player_id": "Z355",
            },
        ],
    )
    _write_players_csv(
        tmp_path,
        [{"player_id": "100644", "name_first": "Alexander", "name_last": "Zverev", "ioc": ""}],
    )
    fake_ingest_conn.rowcount = 2

    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
    )

    assert summary["source_rows"] == 2  # the rank-201 row is skipped before validation
    assert summary["top200"] == 2  # 201 is filtered out before mapping
    assert summary["upserted"] == 2
    assert summary["skipped_existing"] == 0
    assert sorted(int(r[2]) for r in fake_ingest_conn.copied_rows) == [1, 200]


def test_ingest_rankings_default_skips_existing_pk(fake_ingest_conn, tmp_path):
    """No --force: an existing (ranking_date, player_id) row is skipped."""
    _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, "207989", "12050"]])
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    _write_players_csv(
        tmp_path, [{"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": ""}]
    )

    ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
    )

    insert_sql = next(
        s
        for s, _ in fake_ingest_conn.statements
        if s.startswith(f"INSERT INTO {ingest.BRONZE_RANKINGS_TABLE}")
    )
    assert insert_sql.startswith(f"INSERT INTO {ingest.BRONZE_RANKINGS_TABLE}")
    assert "ON CONFLICT (ranking_date, player_id) DO NOTHING" in insert_sql
    assert not any("DO UPDATE SET" in s for s, _ in fake_ingest_conn.statements)


def test_ingest_rankings_force_overwrites_existing_rows(fake_ingest_conn, tmp_path):
    """--force: rank history upserts overwrite existing rows."""
    _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, "207989", "12050"]])
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    _write_players_csv(
        tmp_path, [{"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": ""}]
    )

    ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
        force=True,
    )

    insert_sql = next(
        s
        for s, _ in fake_ingest_conn.statements
        if s.startswith(f"INSERT INTO {ingest.BRONZE_RANKINGS_TABLE}")
    )
    assert "ON CONFLICT (ranking_date, player_id) DO UPDATE SET" in insert_sql
    assert "rank = excluded.rank" in insert_sql
    assert "points = excluded.points" in insert_sql


def test_ingest_rankings_reports_unmapped_rows(fake_ingest_conn, tmp_path, capsys):  # noqa: ARG001 — fixture patches ingest.connection
    _write_ranking_csv(
        tmp_path,
        "atp_rankings_00s.csv",
        [
            ["20260105", 2, "999999", "10000"],
            ["20260106", 3, "999999", "9000"],
            ["20260105", 1, "207989", "12050"],
        ],
    )
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    _write_players_csv(
        tmp_path, [{"player_id": "999999", "name_first": "Nobody", "name_last": "Here", "ioc": ""}]
    )

    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
    )

    assert summary["unmapped"] == 2
    out = capsys.readouterr().out
    assert "unmapped: source_id=999999 name='Nobody Here' rows=2" in out


def test_ingest_rankings_print_reports_skipped_existing(fake_ingest_conn, tmp_path, capsys):
    """A repeat run whose PKs all exist reports 0 inserted and N skipped."""
    _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, "207989", "12050"]])
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    _write_players_csv(
        tmp_path, [{"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": ""}]
    )
    fake_ingest_conn.rowcount = 0  # the (date, player) PK already exists

    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
    )

    assert summary["upserted"] == 0
    assert summary["skipped_existing"] == 1
    assert (
        "Rankings import: 0 inserted/updated rows (1 skipped existing)" in capsys.readouterr().out
    )


def test_ingest_rankings_filters_to_requested_canonical_ids(fake_ingest_conn, tmp_path):
    """player_ids restricts the import to those canonical players only."""
    _write_ranking_csv(
        tmp_path,
        "atp_rankings_00s.csv",
        [
            ["20260105", 1, "207989", "12050"],  # mapped -> A0E2 (requested)
            ["20260105", 2, "100644", "10000"],  # mapped -> Z355 (not requested)
        ],
    )
    _write_map_csv(
        tmp_path,
        [
            {"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"},
            {
                "ranking_player_id": "100644",
                "ranking_name": "Alexander Zverev",
                "player_id": "Z355",
            },
        ],
    )
    _write_players_csv(
        tmp_path,
        [
            {"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": ""},
            {"player_id": "100644", "name_first": "A", "name_last": "Z", "ioc": ""},
        ],
    )

    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
        player_ids={"A0E2"},
    )

    # Both archive rows are top-200; only the seeded player's row is imported.
    assert summary["source_rows"] == 2
    assert summary["upserted"] == 1
    assert len(fake_ingest_conn.copied_rows) == 1
    assert fake_ingest_conn.copied_rows[0][1] == "A0E2"


def test_ingest_rankings_no_archive_returns_zero_import(fake_ingest_conn, tmp_path, capsys):
    """An empty local ranking archive is a successful zero-import seed."""
    summary = ingest.ingest_rankings(tmp_path)

    assert summary == {
        "files": 0,
        "source_rows": 0,
        "top200": 0,
        "upserted": 0,
        "skipped_existing": 0,
        "unmapped": 0,
    }
    assert "No atp_rankings_*.csv files found under data/raw/rankings; nothing to import" in (
        capsys.readouterr().out
    )
    assert fake_ingest_conn.copied_rows == []


def test_ingest_rankings_filtered_path_is_silent_and_seeded_scoped(
    fake_ingest_conn, tmp_path, capsys
):
    """The seed path (player_ids) reports nothing about uncovered or unmapped
    rows: archive players outside the seed set and map gaps are not seed
    warnings, and the summary counts only the seeded result."""
    _write_ranking_csv(
        tmp_path,
        "atp_rankings_00s.csv",
        [
            ["20260105", 1, "207989", "12050"],  # mapped -> A0E2 (requested)
            ["20260106", 2, "207989", "11000"],  # mapped -> A0E2 (requested)
            ["20260105", 3, "999999", "10000"],  # unmapped -> never reported
        ],
    )
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    _write_players_csv(
        tmp_path,
        [
            {"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": ""},
            {"player_id": "999999", "name_first": "Nobody", "name_last": "Here", "ioc": ""},
        ],
    )

    fake_ingest_conn.rowcount = 2
    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
        player_ids={"A0E2", "NOPE1"},  # NOPE1 also has no map coverage
    )

    out = capsys.readouterr().out
    assert "uncovered: seeded player_id=" not in out
    assert "unmapped: source_id=" not in out
    # The summary describes only the seeded result, not the global archive.
    assert summary == {
        "files": 1,
        "source_rows": 3,
        "top200": 2,
        "upserted": 2,
        "skipped_existing": 0,
        "unmapped": 0,
        "auto_mapped": 0,
        "unresolved": 1,
        "coverage": {
            "seeded": 2,
            "covered": 1,
            "auto_mapped": 0,
            "unresolved": 0,
        },
    }
    assert len(fake_ingest_conn.copied_rows) == 2
    assert {row[1] for row in fake_ingest_conn.copied_rows} == {"A0E2"}


def test_ingest_rankings_seeded_player_without_rank_rows_is_silent(
    fake_ingest_conn, tmp_path, capsys
):
    """A seeded player with no rank rows in the archive is a normal seed, not a
    warning; nothing is written."""
    _write_ranking_csv(
        tmp_path,
        "atp_rankings_00s.csv",
        [["20260105", 1, "207989", "12050"]],  # only for a player not seeded
    )
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    _write_players_csv(
        tmp_path, [{"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": ""}]
    )

    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
        player_ids={"Z355"},  # seeded but absent from the archive/map
    )

    assert summary == {
        "files": 1,
        "source_rows": 1,
        "top200": 0,
        "upserted": 0,
        "skipped_existing": 0,
        "unmapped": 0,
        "auto_mapped": 0,
        "unresolved": 0,
        "coverage": {
            "seeded": 1,
            "covered": 0,
            "auto_mapped": 0,
            "unresolved": 0,
        },
    }
    assert fake_ingest_conn.copied_rows == []
    out = capsys.readouterr().out
    assert "uncovered" not in out
    assert "unmapped: source_id=" not in out


def test_ingest_rankings_filtered_ioc_backfill_restricted_to_seeded(fake_ingest_conn, tmp_path):
    """player_ids backfills IOCs only for the seeded canonical ids — the full
    map is never mutated during a seed."""
    _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, "207989", "12050"]])
    _write_map_csv(
        tmp_path,
        [
            {"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"},
            {
                "ranking_player_id": "100644",
                "ranking_name": "Alexander Zverev",
                "player_id": "Z355",
            },
        ],
    )
    _write_players_csv(
        tmp_path,
        [
            {"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": "ESP"},
            {"player_id": "100644", "name_first": "A", "name_last": "Z", "ioc": "GER"},
        ],
    )

    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
        player_ids={"A0E2"},
    )

    assert summary["upserted"] == 1
    ioc_stmt = next(
        (s, p)
        for s, p in fake_ingest_conn.statements
        if s.startswith(f"UPDATE {ingest.BRONZE_PROFILES_TABLE}")
    )
    assert ioc_stmt[1] == [("ESP", "A0E2")]  # Z355 never touched during seed


def test_ingest_rankings_seed_path_keeps_ioc_fallback(fake_ingest_conn, tmp_path, capsys):
    """The seed path keeps the ranking-source IOC fallback, scoped to its
    player_ids; the summary reports ranking rows only, no fallback wording."""
    _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, "207989", "12050"]])
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    _write_players_csv(
        tmp_path, [{"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": "ESP"}]
    )

    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
        player_ids={"A0E2"},
    )

    assert summary["upserted"] == 1
    ioc_stmt = next(
        (s, p)
        for s, p in fake_ingest_conn.statements
        if s.startswith(f"UPDATE {ingest.BRONZE_PROFILES_TABLE}")
    )
    # The fallback only replaces NULL/empty/UNK — a verified IOC is never
    # overwritten.
    assert "WHERE player_id = %s AND (ioc IS NULL OR ioc = '' OR ioc = 'UNK')" in ioc_stmt[0]
    assert ioc_stmt[1] == [("ESP", "A0E2")]
    out = capsys.readouterr().out
    assert "Rankings import: 1 inserted/updated rows" in out
    assert "IOC" not in out  # summary reports ranking rows only


def test_ingest_rankings_seed_path_skips_fallback_without_source_ioc(
    fake_ingest_conn, tmp_path, capsys
):
    """A seed whose ranking source carries no usable IOC emits no fallback
    UPDATE, and the summary stays ranking-rows-only."""
    _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, "207989", "12050"]])
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    # The ranking source carries no usable IOC for the seeded player.
    _write_players_csv(
        tmp_path, [{"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": ""}]
    )

    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
        player_ids={"A0E2"},
    )

    assert summary["upserted"] == 1
    assert not any(
        s.startswith(f"UPDATE {ingest.BRONZE_PROFILES_TABLE}")
        for s, _ in fake_ingest_conn.statements
    )
    assert "Rankings import: 1 inserted/updated rows" in capsys.readouterr().out


def test_ingest_rankings_full_path_keeps_ioc_fallback(fake_ingest_conn, tmp_path, capsys):
    """The full scrape path (player_ids=None) keeps the ranking-source IOC
    fallback; the summary reports ranking rows only, no fallback wording."""
    _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, "207989", "12050"]])
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    _write_players_csv(
        tmp_path, [{"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": "ESP"}]
    )

    summary = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
    )

    assert summary["upserted"] == 1
    ioc_stmt = next(
        (s, p)
        for s, p in fake_ingest_conn.statements
        if s.startswith(f"UPDATE {ingest.BRONZE_PROFILES_TABLE}")
    )
    assert ioc_stmt[1] == [("ESP", "A0E2")]
    out = capsys.readouterr().out
    assert "Rankings import: 1 inserted/updated rows" in out
    assert "IOC" not in out  # summary reports ranking rows only


def test_ingest_rankings_repeat_run_is_idempotent(fake_ingest_conn, tmp_path):
    """Re-running the same seed upserts the same rows; the PK conflict key prevents
    growth."""
    _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", 1, "207989", "12050"]])
    _write_map_csv(
        tmp_path,
        [{"ranking_player_id": "207989", "ranking_name": "Carlos Alcaraz", "player_id": "A0E2"}],
    )
    _write_players_csv(
        tmp_path, [{"player_id": "207989", "name_first": "C", "name_last": "A", "ioc": ""}]
    )

    first = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
        player_ids={"A0E2"},
    )
    second = ingest.ingest_rankings(
        tmp_path,
        map_csv=tmp_path / "ranking_player_map.csv",
        players_csv=tmp_path / "atp_players.csv",
        player_ids={"A0E2"},
    )

    assert first == second
    assert first["upserted"] == 1
    inserts = [
        s
        for s, _ in fake_ingest_conn.statements
        if s.startswith(f"INSERT INTO {ingest.BRONZE_RANKINGS_TABLE}")
    ]
    assert len(inserts) == 2  # one COPY-stage INSERT per run
    # Default (idempotent) ingest is DO NOTHING — an existing PK is never
    # overwritten.
    assert "ON CONFLICT (ranking_date, player_id) DO NOTHING" in inserts[0]


def test_backfill_profile_iocs_updates_only_unk(fake_ingest_conn, tmp_path):
    players = _write_players_csv(
        tmp_path, [{"player_id": "100004", "name_first": "G", "name_last": "M", "ioc": "ITA"}]
    )

    ingest.backfill_profile_iocs({"100004": "M276"}, {"M276"}, players_csv=players)

    sql, params = fake_ingest_conn.statements[0]
    assert sql.startswith(f"UPDATE {ingest.BRONZE_PROFILES_TABLE}")
    assert "WHERE player_id = %s AND (ioc IS NULL OR ioc = '' OR ioc = 'UNK')" in sql
    assert params == [("ITA", "M276")]


def test_backfill_profile_iocs_skips_invalid_ioc(fake_ingest_conn, tmp_path):
    players = _write_players_csv(
        tmp_path, [{"player_id": "100004", "name_first": "G", "name_last": "M", "ioc": "XYZ"}]
    )

    ingest.backfill_profile_iocs({"100004": "M276"}, {"M276"}, players_csv=players)
    assert fake_ingest_conn.statements == []


def test_backfill_profile_iocs_skips_players_not_in_map(fake_ingest_conn, tmp_path):
    players = _write_players_csv(
        tmp_path, [{"player_id": "100004", "name_first": "G", "name_last": "M", "ioc": "ITA"}]
    )

    # The map only covers 207989, so 100004's IOC is never touched.
    ingest.backfill_profile_iocs({"207989": "A0E2"}, {"A0E2"}, players_csv=players)
    assert fake_ingest_conn.statements == []
