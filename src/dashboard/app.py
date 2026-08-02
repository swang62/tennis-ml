"""Panel dashboard for tennis match data exploration.

Usage:
    panel serve src/dashboard/app.py
"""

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go

from src.db.client import to_dataframe

pn.extension("plotly", "tabulator")

GOLD_TABLE = "gold.match_features"


def get_players() -> list[str]:
    df = to_dataframe(
        f"SELECT DISTINCT player_id FROM {GOLD_TABLE} "
        f"UNION SELECT DISTINCT opponent_id FROM {GOLD_TABLE} ORDER BY 1"
    )
    return df["player_id"].tolist()


def _orient(df: pd.DataFrame, player: str) -> pd.DataFrame:
    """Flip canonical rows so `player` appears on the player_* side.

    gold.match_features is one row per match with the canonical (lower ATP id)
    player on the player_* side; match_won and player_ranking refer to that
    side. This rewrites the rows from `player`'s perspective.
    """
    out = df.copy()
    side = out["player_id"] == player
    pid = out["player_id"].to_numpy()
    oid = out["opponent_id"].to_numpy()
    p_rank = out["player_ranking"].to_numpy()
    o_rank = out["opponent_ranking"].to_numpy()
    won = out["match_won"].to_numpy()
    out["player_id"] = np.where(side, pid, oid)
    out["opponent_id"] = np.where(side, oid, pid)
    out["player_ranking"] = np.where(side, p_rank, o_rank)
    out["opponent_ranking"] = np.where(side, o_rank, p_rank)
    out["match_won"] = np.where(side, won, 1 - won)
    return out


def get_head_to_head(player_a: str, player_b: str) -> pd.DataFrame:
    return to_dataframe(f"""
        SELECT * FROM {GOLD_TABLE}
        WHERE (player_id = '{player_a}' AND opponent_id = '{player_b}')
           OR (player_id = '{player_b}' AND opponent_id = '{player_a}')
        ORDER BY match_date
    """)


def get_player_rank_history(player: str) -> pd.DataFrame:
    df = to_dataframe(f"""
        SELECT match_date, player_id, opponent_id,
               player_ranking, opponent_ranking, match_won, surface
        FROM {GOLD_TABLE}
        WHERE player_id = '{player}' OR opponent_id = '{player}'
        ORDER BY match_date
    """)
    return _orient(df, player)


def get_player_match_history(player: str, limit: int = 50) -> pd.DataFrame:
    df = to_dataframe(f"""
        SELECT match_date, player_id, opponent_id, surface, tournament, round,
               player_ranking, opponent_ranking, match_won,
               ace_rate, double_fault_rate, first_serve_pct
        FROM {GOLD_TABLE}
        WHERE player_id = '{player}' OR opponent_id = '{player}'
        ORDER BY match_date DESC
        LIMIT {limit}
    """)
    return _orient(df, player)


def compute_h2h_summary(df: pd.DataFrame, player_a: str, player_b: str):
    a = _orient(df, player_a)
    b = _orient(df, player_b)
    return {
        "a": {"wins": int(a["match_won"].sum()), "losses": int((1 - a["match_won"]).sum())},
        "b": {"wins": int(b["match_won"].sum()), "losses": int((1 - b["match_won"]).sum())},
    }


def draw_form_sequence(df: pd.DataFrame, player: str):
    player_rows = df[df["player_id"] == player].sort_values("match_date")
    results = player_rows["match_won"].tolist()
    opponents = player_rows["opponent_id"].tolist()

    colors = ["#2ecc71" if r == 1 else "#e74c3c" for r in results]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(len(results))),
            y=[1] * len(results),
            marker_color=colors,
            text=[
                f"{'W' if r == 1 else 'L'} vs {o}" for r, o in zip(results, opponents, strict=False)
            ],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=120,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        xaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
        yaxis={"showticklabels": False, "showgrid": False, "range": [0, 1.5]},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Load data & create widgets ──

try:
    players = get_players()
except Exception as e:
    print(f"Cannot connect to DuckDB: {e}")
    print(
        "Make sure the database is initialized and seeded (run `just db-reset` then `just db-seed`)."
    )
    raise

initial_a = players[0] if players else None
initial_b = players[1] if len(players) > 1 else None

player_a = pn.widgets.Select(name="Player A", options=players, value=initial_a)
player_b = pn.widgets.Select(
    name="Player B",
    options=[p for p in players if p != initial_a] if players else [],
    value=initial_b,
)


def _update_b_options(event):
    remaining = [p for p in players if p != event.new]
    player_b.options = remaining
    if event.new == player_b.value or player_b.value not in remaining:
        idx = 1 if len(remaining) > 1 else 0
        player_b.value = remaining[idx] if remaining else None


player_a.param.watch(_update_b_options, "value")


# ── Matchup tab ──


@pn.depends(player_a.param.value, player_b.param.value)
def matchup_content(a, b):
    if not a or not b:
        return pn.pane.Markdown("*Select both players*")

    h2h = get_head_to_head(a, b)

    if h2h.empty:
        return pn.pane.Markdown(f"*No matches found between {a} and {b}*")

    h2h_summary = compute_h2h_summary(h2h, a, b)
    wins_a = h2h_summary["a"]["wins"]
    losses_a = h2h_summary["a"]["losses"]
    wins_b = h2h_summary["b"]["wins"]
    losses_b = h2h_summary["b"]["losses"]
    total_a = wins_a + losses_a
    total_b = wins_b + losses_b
    pct_a = f"{wins_a / total_a * 100:.0f}%" if total_a > 0 else ""
    pct_b = f"{wins_b / total_b * 100:.0f}%" if total_b > 0 else ""

    h2h_fig = go.Figure()
    h2h_fig.add_trace(
        go.Bar(
            x=[a, b],
            y=[wins_a, wins_b],
            marker_color=["#3498db", "#e67e22"],
            text=[str(wins_a), str(wins_b)],
            textposition="outside",
        )
    )
    h2h_fig.update_layout(
        height=250,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        yaxis={"title": "Wins", "dtick": 1},
        plot_bgcolor="rgba(0,0,0,0)",
    )

    rank_a = get_player_rank_history(a)
    rank_a["player_id"] = a
    rank_b = get_player_rank_history(b)
    rank_b["player_id"] = b
    rank_both = pd.concat([rank_a, rank_b])

    rank_fig = px.line(
        rank_both,
        x="match_date",
        y="player_ranking",
        color="player_id",
        markers=True,
        color_discrete_map={a: "#3498db", b: "#e67e22"},
    )
    rank_fig.update_layout(
        yaxis={"autorange": "reversed", "title": "Ranking"},
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    display_cols = [
        "match_date",
        "player_id",
        "opponent_id",
        "surface",
        "tournament",
        "round",
        "player_ranking",
        "match_won",
        "ace_rate",
        "double_fault_rate",
        "first_serve_pct",
    ]
    available_cols = [c for c in display_cols if c in h2h.columns]
    match_table = h2h[available_cols].sort_values("match_date", ascending=False)

    return pn.Column(
        pn.Row(
            pn.Column(
                pn.pane.Markdown(f"**{a}**"),
                pn.pane.Markdown(f"### {wins_a}-{losses_a}"),
                pn.pane.Markdown(f"*{pct_a}*"),
                width=180,
            ),
            pn.pane.Plotly(h2h_fig, sizing_mode="stretch_width"),
            pn.Column(
                pn.pane.Markdown(f"**{b}**"),
                pn.pane.Markdown(f"### {wins_b}-{losses_b}"),
                pn.pane.Markdown(f"*{pct_b}*"),
                width=180,
            ),
        ),
        pn.pane.Markdown("#### Recent Form"),
        pn.Row(
            pn.pane.Plotly(draw_form_sequence(_orient(h2h, a), a), sizing_mode="stretch_width"),
            pn.pane.Plotly(draw_form_sequence(_orient(h2h, b), b), sizing_mode="stretch_width"),
        ),
        pn.pane.Markdown("#### Match History"),
        pn.widgets.Tabulator(match_table, sizing_mode="stretch_width"),
        pn.pane.Markdown("#### Rank Progression"),
        pn.pane.Plotly(rank_fig, sizing_mode="stretch_width"),
    )


# ── Explorer tab ──

explorer_player = pn.widgets.Select(name="Select Player", options=players, value=initial_a)


@pn.depends(explorer_player.param.value)
def explorer_content(player):
    if not player:
        return pn.pane.Markdown("*Select a player*")

    rank_history = get_player_rank_history(player)

    rank_fig = px.line(
        rank_history,
        x="match_date",
        y="player_ranking",
        markers=True,
    )
    rank_fig.update_layout(
        yaxis={"autorange": "reversed", "title": "Ranking"},
        height=350,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    surface_stats = (
        rank_history.groupby("surface")["match_won"].agg(["mean", "count"]).reset_index()
    )
    surface_stats.columns = ["surface", "win_rate", "matches"]
    surface_stats["win_rate"] = (surface_stats["win_rate"] * 100).round(1)

    surface_fig = px.bar(
        surface_stats,
        x="surface",
        y="win_rate",
        text=surface_stats.apply(lambda r: f"{r['win_rate']}% (n={r['matches']})", axis=1),
        color="win_rate",
        color_continuous_scale="RdYlGn",
    )
    surface_fig.update_layout(
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"title": "Win %", "range": [0, 100]},
    )

    match_history = get_player_match_history(player)

    return pn.Column(
        pn.pane.Markdown("#### Rank Over Time"),
        pn.pane.Plotly(rank_fig, sizing_mode="stretch_width"),
        pn.pane.Markdown("#### Win Rate by Surface"),
        pn.pane.Plotly(surface_fig, sizing_mode="stretch_width"),
        pn.pane.Markdown("#### Recent Matches"),
        pn.widgets.Tabulator(match_history, sizing_mode="stretch_width"),
    )


# ── Layout ──

matchup_tab = pn.Column(
    pn.Row(player_a, player_b),
    matchup_content,
)

explorer_tab = pn.Column(
    explorer_player,
    explorer_content,
)

tabs = pn.Tabs(
    ("Matchup Analysis", matchup_tab),
    ("Player Explorer", explorer_tab),
)

pn.template.FastListTemplate(
    title="Tennis Matchup Explorer",
    main=[tabs],
    accent_base_color="#3498db",
    header_background="#057052",
).servable()
