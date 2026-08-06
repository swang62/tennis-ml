"""Hermetic tests for src/flows/ingest.py (no network, no real DuckDB file)."""

import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import src.flows.ingest as ingest
from src.constants import ROOT
from src.features.columns import BRONZE_COLUMNS

PROFILES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gold.player_profiles (
    player_id    VARCHAR PRIMARY KEY,
    display_name VARCHAR,
    atp_name     VARCHAR,
    birthdate    DATE,
    weight       SMALLINT,
    height       SMALLINT,
    turned_pro   INTEGER,
    birthplace   VARCHAR,
    coaches      VARCHAR,
    handedness   VARCHAR,
    backhand     VARCHAR,
    ioc          VARCHAR,
    summary      VARCHAR,
    enriched_at  TIMESTAMP
)
"""


@pytest.fixture
def profiles_conn(monkeypatch):
    """In-memory DuckDB with the gold.player_profiles schema wired to ingest.get_conn."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA IF NOT EXISTS gold")
    conn.execute(PROFILES_SCHEMA_SQL)
    monkeypatch.setattr(ingest, "get_conn", lambda: conn)
    yield conn
    conn.close()


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
        "round": "QF",
        "surface": "Hard",
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
    assert row["round"] == "qf"
    assert row["surface"] == "hard"
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
                "round": "QF",
                "surface": "Hard",
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


def test_load_raw_atp_rows_passes_ranks_through_raw(tmp_path):
    """Ranks are NOT imputed at ingest: 0 stays 0 (ATP missing marker) and
    empty cells become 0, so silver NULLIFs them and gold imputes at train
    time. Only missing player ids are dropped."""
    df = _raw_atp_df()
    null_rank = _raw_atp_df()
    null_rank.loc[0, "winner_rank"] = None
    null_rank.loc[0, "winner_id"] = "W2"
    zero_rank = _raw_atp_df()
    zero_rank.loc[0, "winner_rank"] = 0
    zero_rank.loc[0, "winner_id"] = "W3"
    csv = tmp_path / "ranks.csv"
    pd.concat([df, null_rank, zero_rank], ignore_index=True).to_csv(csv, index=False)

    rows = ingest.load_raw_atp_rows(csv)

    assert len(rows) == 3  # missing and 0 ranks pass through, not dropped
    by_id = {r["winner_id"]: r for r in rows}
    assert by_id["W2"]["winner_rank"] == 0  # empty rank -> 0, no median fill
    assert by_id["W3"]["winner_rank"] == 0  # rank 0 passes through unchanged
    assert by_id["W3"]["loser_rank"] == 20


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


def test_load_atp_profiles_inserts_base_columns(profiles_conn, tmp_path):
    csv = _write_profiles_csv(tmp_path)

    assert ingest.load_atp_profiles(csv) == 2

    row = profiles_conn.execute(
        "SELECT player_id, display_name, atp_name, birthdate, weight, height, turned_pro, "
        "handedness, backhand, ioc FROM gold.player_profiles WHERE player_id = 'P1'"
    ).fetchone()
    assert row == (
        "P1",
        "Player One",
        "P. One",
        datetime.date(1993, 3, 16),
        85,
        185,
        2018,
        "R",
        "2H",
        "FRA",
    )


def test_load_atp_profiles_preserves_enrichment_on_reload(profiles_conn, tmp_path):
    csv = _write_profiles_csv(tmp_path)
    ingest.load_atp_profiles(csv)
    profiles_conn.execute(
        "UPDATE gold.player_profiles SET summary = 'Existing enrichment' WHERE player_id = 'P1'"
    )

    ingest.load_atp_profiles(csv)

    summary = profiles_conn.execute(
        "SELECT summary FROM gold.player_profiles WHERE player_id = 'P1'"
    ).fetchone()[0]
    assert summary == "Existing enrichment"


def test_load_atp_profiles_filters_by_player_ids(profiles_conn, tmp_path):
    csv = _write_profiles_csv(tmp_path)

    assert ingest.load_atp_profiles(csv, player_ids={"P2"}) == 1

    rows = profiles_conn.execute("SELECT player_id FROM gold.player_profiles").fetchall()
    assert [r[0] for r in rows] == ["P2"]


def test_load_atp_profiles_dedupes_duplicate_ids_last_wins(profiles_conn, tmp_path):
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

    row = profiles_conn.execute(
        "SELECT player_id, display_name, weight FROM gold.player_profiles"
    ).fetchone()
    assert row == ("P1", "Last", 90)


# ── enrich_player / enrich_players / enrich_missing ───────────────


def test_enrich_player_apostrophe_name_never_stores_none(monkeypatch, profiles_conn):
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

    row = profiles_conn.execute(
        "SELECT player_id, display_name, summary FROM gold.player_profiles"
    ).fetchone()
    assert row[0] == "Jan O'Brien"
    assert row[1] == "Jan O'Brien"
    assert row[2] == "Jan O'Brien is a player with an apostrophe."
    assert profiles_conn.execute("SELECT count(*) FROM gold.player_profiles").fetchone()[0] == 1


def test_enrich_player_uses_explicit_player_id(monkeypatch, profiles_conn):
    monkeypatch.setattr(ingest, "search_wikipedia", lambda _name: "Some Title")
    monkeypatch.setattr(
        ingest,
        "fetch_summary",
        lambda _title: {"title": "Some Title", "summary": "Bio text.", "page_id": "7"},
    )

    assert ingest.enrich_player("Name", "REALID") is True

    row = profiles_conn.execute("SELECT player_id FROM gold.player_profiles").fetchone()
    assert row[0] == "REALID"


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


def test_enrich_player_stores_playing_style_paragraph(monkeypatch, profiles_conn):
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

    summary = profiles_conn.execute("SELECT summary FROM gold.player_profiles").fetchone()[0]
    assert summary == "Style paragraph."


def test_enrich_player_falls_back_to_lead_paragraph(monkeypatch, profiles_conn):
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

    summary = profiles_conn.execute("SELECT summary FROM gold.player_profiles").fetchone()[0]
    assert summary == "Lead paragraph."


def test_enrich_player_skips_when_no_usable_paragraph(monkeypatch, profiles_conn):
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
    assert profiles_conn.execute("SELECT count(*) FROM gold.player_profiles").fetchone()[0] == 0


def test_enrich_missing_enriches_missing_players(monkeypatch):
    monkeypatch.setattr(ingest, "get_players_without_profiles", lambda: ["X"])
    monkeypatch.setattr(ingest, "enrich_player", lambda _name: True)

    assert ingest.enrich_missing() == 1


def test_enrich_missing_noop_when_none_missing(monkeypatch):
    monkeypatch.setattr(ingest, "get_players_without_profiles", lambda: [])

    assert ingest.enrich_missing() == 0


def test_enrich_players_skips_already_enriched_and_nameless(monkeypatch, profiles_conn):
    profiles_conn.execute(
        "INSERT INTO gold.player_profiles (player_id, display_name, summary) VALUES "
        "('P1', 'Player One', NULL), "
        "('P2', 'Player Two', 'Already enriched.'), "
        "('P3', NULL, NULL)"
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ingest, "enrich_player", lambda name, pid: calls.append((pid, name)) or True
    )

    assert ingest.enrich_players(["P1", "P2", "P3"]) == 1
    assert calls == [("P1", "Player One")]


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
