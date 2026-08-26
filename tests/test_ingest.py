"""Hermetic ingest tests with fake network, database, and ATP CSV seams."""

from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

import src.db.ingest as ingest
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
    """Minimal psycopg-like connection recording statements, COPY rows, and rowcount."""

    def __init__(self):
        self.statements: list[tuple[str, object | None]] = []
        self.copied_rows: list[tuple[object, ...]] = []
        self.fetchall_result: list[tuple[object, ...]] = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

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
    monkeypatch.setattr(ingest.psycopg, "connect", lambda *_args, **_kwargs: conn)
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


def test_parse_birthdate_parses_valid_and_empty_to_null():
    result = ingest._parse_birthdate(pd.Series(["19930316", ""]), pd.Series(["P1", "P2"]))

    assert result.dtype == "datetime64[ns]"
    assert result[0] == pd.Timestamp("1993-03-16")
    assert pd.isna(result[1])


def test_parse_birthdate_rejects_malformed_date():
    with pytest.raises(ValueError, match="birthdate malformed"):
        ingest._parse_birthdate(pd.Series(["20261301"]), pd.Series(["P1"]))


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
        "best_of": 3,
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
    assert row["match_id"] == "2026-9900-001"
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


def test_atp_rows_to_bronze_surface_canonicalizes():
    """Exactly the four canonical surfaces pass through; absent/0/unmapped
    source values default to hard."""
    cases = [
        ("Clay", "clay"),
        ("GRASS", "grass"),
        ("Hard", "hard"),
        ("carpet", "carpet"),
        ("", "hard"),
        ("0", "hard"),
        (0, "hard"),
        (0.0, "hard"),
        (None, "hard"),
        ("nan", "hard"),
        ("turf", "hard"),
    ]
    for raw, expected in cases:
        row = _raw_row()
        row["surface"] = raw
        assert ingest.atp_rows_to_bronze([row]).iloc[0]["surface"] == expected


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


def test_match_id_is_date_free_and_stable():
    """Same opaque tourney_id + match sequence -> same id, whatever the date."""
    ids = {
        ingest.atp_rows_to_bronze([_raw_row(match_num=26, tourney_date=d)]).iloc[0]["match_id"]
        for d in ("20260102", "20260207", "20260314")
    }
    assert ids == {"2026-9900-026"}


def test_match_id_keeps_opaque_dashed_tournament_ids_distinct():
    """Dashes inside tourney_id are preserved, never parsed; Davis Cup style
    non-numeric ids pass through (prefixed with the edition year). ``2026-41-8``
    and ``2026-418`` cannot collide."""
    a = _raw_row(match_num=1)
    a["tourney_id"] = "2026-41-8"
    b = _raw_row(match_num=1)
    b["tourney_id"] = "2026-418"
    c = _raw_row(match_num=1)
    c["tourney_id"] = "1967-southern-pro"
    c["tourney_level"] = "D"

    df = ingest.atp_rows_to_bronze([a, b, c])

    assert sorted(df["match_id"]) == [
        "2026-1967-southern-pro-001",
        "2026-41-8-001",
        "2026-418-001",
    ]
    assert (
        df.loc[df["match_id"] == "2026-1967-southern-pro-001", "tournament"].iloc[0] == "davis_cup"
    )


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
                "best_of": 3,
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
    assert insert_sql.startswith(f"INSERT INTO {ingest.BRONZE_MATCHES_TABLE}")
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
    assert insert_sql.startswith(f"INSERT INTO {ingest.BRONZE_MATCHES_TABLE}")
    assert "ON CONFLICT (match_id) DO UPDATE SET" in insert_sql
    assert "match_id = excluded.match_id" not in insert_sql
    assert "winner_id = excluded.winner_id" in insert_sql
    assert "ingested_at = CURRENT_TIMESTAMP" in insert_sql


def test_insert_bronze_rows_returns_db_affected_count(fake_ingest_conn):
    """The return value is the database's actual inserted count: when every
    PK already exists the seed reports 0 inserted, not the input row count."""
    df = ingest.atp_rows_to_bronze([_raw_row()])
    fake_ingest_conn.rowcount = 0

    assert ingest.insert_bronze_rows(df) == 0

    # The row was still staged/attempted — the DB just skipped the conflict.
    assert len(fake_ingest_conn.copied_rows) == 1


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

    # COPY stages rows before the idempotent INSERT ... SELECT.
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


def test_atp_rows_to_bronze_indoor_missing_defaults_to_outdoor():
    row = _raw_row()
    row["indoor"] = None
    df = ingest.atp_rows_to_bronze([row])
    assert df.iloc[0]["is_indoor"] == 0


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


# ── Ranking identity map contract ────────────────────────────────


def _write_map_csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    csv = tmp_path / "ranking_player_map.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


# ── ATP player persistence ───────────────────────────────────────


def _valid_candidate() -> dict[str, object]:
    """A well-formed, unseen ATP identity candidate."""
    return {
        "id": "XQ999",
        "player": "Test Player",
        "atpname": "T. Player",
        "birthdate": "19950101",
        "weight": "80",
        "height": "185",
        "turnedpro": "2018",
        "birthplace": "Paris",
        "coaches": "",
        "hand": "R",
        "backhand": "2H",
        "ioc": "fra",
    }


def _init_persist_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create the canonical and map CSVs with their real headers."""
    profiles = tmp_path / "ATP_player_database.csv"
    profiles.write_text(
        "id,player,atpname,birthdate,weight,height,turnedpro,birthplace,"
        "coaches,hand,backhand,ioc\n"
        '"P001","Existing One","E. One","19900101","75","180",'
        '"2015","","","R","2H","FRA"\n'
    )
    map_csv = tmp_path / "ranking_player_map.csv"
    map_csv.write_text('ranking_player_id,ranking_name,player_id\n"P001","Existing One","P001"\n')
    return profiles, map_csv


def test_persist_atp_player_appends_one_row_and_inserts(fake_ingest_conn, tmp_path):
    """A valid unseen identity yields one canonical row and bronze insert."""
    profiles, _ = _init_persist_files(tmp_path)

    assert ingest.persist_atp_player(_valid_candidate(), profiles_csv=profiles) == 1

    canonical = pd.read_csv(profiles)
    assert len(canonical) == 2  # header + one new row, existing untouched
    new_row = canonical[canonical["id"] == "XQ999"].iloc[0]
    assert new_row["player"] == "Test Player"
    assert new_row["ioc"] == "FRA"  # normalized, uppercased

    # One bronze profile upsert routed through load_atp_profiles.
    insert_sql = next(
        s
        for s, _ in fake_ingest_conn.statements
        if s.startswith(f"INSERT INTO {ingest.BRONZE_PROFILES_TABLE}")
    )
    assert "ON CONFLICT (player_id) DO UPDATE SET" in insert_sql
    assert len(fake_ingest_conn.copied_rows) == 1
    assert fake_ingest_conn.copied_rows[0][0] == "XQ999"


def test_persist_atp_player_rejects_invalid_candidate_before_any_write(tmp_path):
    """Invalid required data or malformed optional typed data is rejected with
    a reason before any dependent row is appended."""
    profiles, _ = _init_persist_files(tmp_path)

    for bad, pattern in [
        (dict(_valid_candidate(), id=""), "candidate missing ATP player id"),
        (dict(_valid_candidate(), player=""), "candidate missing display name"),
        (dict(_valid_candidate(), birthdate="1995-33-01"), "birthdate malformed"),
        (dict(_valid_candidate(), weight="abc"), "weight malformed"),
        (dict(_valid_candidate(), weight="500"), "weight malformed"),
        (dict(_valid_candidate(), birthdate="18991231"), "birthdate out of range"),
    ]:
        with pytest.raises(ValueError, match=pattern):
            ingest.persist_atp_player(bad, profiles_csv=profiles)

    # Nothing was written for any rejected candidate.
    assert len(pd.read_csv(profiles)) == 1


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


def test_committed_ranking_player_map_is_valid_and_deterministic():
    """The reviewed map file loads cleanly and pins the approved source id."""
    result = ingest.load_ranking_player_map()

    assert "207989" in result
    assert result["207989"] == "A0E2"  # Alcaraz source id -> canonical id
    assert ingest.load_ranking_player_map() == result  # deterministic


# ── Official rankings ingest ───────────────────────────────────


def _write_ranking_csv(tmp_path: Path, name: str, rows: list[list[object]]) -> Path:
    csv = tmp_path / name
    pd.DataFrame(rows, columns=ingest.RANKINGS_COLUMNS).to_csv(csv, index=False)
    return csv


def _write_players_csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    csv = tmp_path / "atp_players.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


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


@pytest.mark.parametrize("rank", ["abc", "-3"])
def test_load_ranking_rows_rejects_malformed_rank(tmp_path, rank):
    csv = _write_ranking_csv(tmp_path, "atp_rankings_00s.csv", [["20260105", rank, "207989", "1"]])

    with pytest.raises(ValueError, match="rank malformed"):
        ingest.load_ranking_rows([csv])


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
