"""Streamlit dashboard for tennis match data exploration.

Usage:
    streamlit run src/dashboard/app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.db.client import to_dataframe

GOLD_TABLE = "gold.match_features"


@st.cache_data(ttl=60)
def get_players() -> list[str]:
    sql = f"SELECT DISTINCT player_id FROM {GOLD_TABLE} ORDER BY player_id"
    df = to_dataframe(sql)
    return df["player_id"].tolist()


@st.cache_data(ttl=60)
def get_head_to_head(player_a: str, player_b: str) -> pd.DataFrame:
    sql = f"""
        SELECT * FROM {GOLD_TABLE}
        WHERE (player_id = '{player_a}' AND opponent_id = '{player_b}')
           OR (player_id = '{player_b}' AND opponent_id = '{player_a}')
        ORDER BY match_date
    """
    return to_dataframe(sql)


@st.cache_data(ttl=60)
def get_player_rank_history(player: str) -> pd.DataFrame:
    sql = f"""
        SELECT match_date, player_ranking, opponent_ranking, match_won, surface
        FROM {GOLD_TABLE}
        WHERE player_id = '{player}'
        ORDER BY match_date
    """
    return to_dataframe(sql)


@st.cache_data(ttl=60)
def get_player_match_history(player: str, limit: int = 50) -> pd.DataFrame:
    sql = f"""
        SELECT match_date, opponent_id, surface, tournament, round,
               player_ranking, opponent_ranking, match_won,
               ace_rate, double_fault_rate, first_serve_pct
        FROM {GOLD_TABLE}
        WHERE player_id = '{player}'
        ORDER BY match_date DESC
        LIMIT {limit}
    """
    return to_dataframe(sql)


def compute_h2h_summary(df: pd.DataFrame, player_a: str, player_b: str):
    a_wins = df[(df["player_id"] == player_a) & (df["match_won"] == 1)].shape[0]
    a_losses = df[(df["player_id"] == player_a) & (df["match_won"] == 0)].shape[0]
    b_wins = df[(df["player_id"] == player_b) & (df["match_won"] == 1)].shape[0]
    b_losses = df[(df["player_id"] == player_b) & (df["match_won"] == 0)].shape[0]
    return {
        "a": {"player_id": player_a, "wins": a_wins, "losses": a_losses},
        "b": {"player_id": player_b, "wins": b_wins, "losses": b_losses},
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


st.set_page_config(
    page_title="Tennis Matchup Explorer",
    page_icon="🎾",
    layout="wide",
)

st.title("🎾 Tennis Matchup Explorer")

try:
    players = get_players()
except Exception as e:
    st.error(f"Cannot connect to ClickHouse: {e}")
    st.info("Make sure the cluster is running and ClickHouse is accessible.")
    st.stop()

tab_matchup, tab_explorer = st.tabs(["Matchup Analysis", "Player Explorer"])

# ── Matchup Tab ──────────────────────────────────────────────

with tab_matchup:
    col_a, col_b = st.columns(2)
    with col_a:
        player_a = st.selectbox("Player A", players, index=0 if players else None, key="a")
    with col_b:
        available_b = [p for p in players if p != player_a]
        player_b = st.selectbox(
            "Player B",
            available_b,
            index=min(1, len(available_b) - 1) if available_b else None,
            key="b",
        )

    if player_a and player_b:
        h2h = get_head_to_head(player_a, player_b)

        if h2h.empty:
            st.info(f"No matches found between {player_a} and {player_b}.")
        else:
            h2h_summary = compute_h2h_summary(h2h, player_a, player_b)

            st.subheader("Head-to-Head")
            ca, cb, cc = st.columns([1, 2, 1])
            with ca:
                wins_a = h2h_summary["a"]["wins"]
                losses_a = h2h_summary["a"]["losses"]
                total_a = wins_a + losses_a
                pct_a = wins_a / total_a * 100 if total_a > 0 else 0
                st.metric(
                    f"{player_a}", f"{wins_a}-{losses_a}", f"{pct_a:.0f}%" if total_a > 0 else None
                )

            with cb:
                h2h_fig = go.Figure()
                h2h_fig.add_trace(
                    go.Bar(
                        x=[player_a, player_b],
                        y=[wins_a, h2h_summary["b"]["wins"]],
                        marker_color=["#3498db", "#e67e22"],
                        text=[str(wins_a), str(h2h_summary["b"]["wins"])],
                        textposition="outside",
                    )
                )
                h2h_fig.update_layout(
                    height=250,
                    margin={"l": 0, "r": 0, "t": 0, "b": 0},
                    yaxis={"title": "Wins", "dtick": 1},
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(h2h_fig, use_container_width=True)

            with cc:
                wins_b = h2h_summary["b"]["wins"]
                losses_b = h2h_summary["b"]["losses"]
                total_b = wins_b + losses_b
                pct_b = wins_b / total_b * 100 if total_b > 0 else 0
                st.metric(
                    f"{player_b}", f"{wins_b}-{losses_b}", f"{pct_b:.0f}%" if total_b > 0 else None
                )

            st.subheader("Recent Form (last 10 matches)")
            fcol_a, fcol_b = st.columns(2)
            with fcol_a:
                fig_a = draw_form_sequence(h2h, player_a)
                st.plotly_chart(fig_a, use_container_width=True)
            with fcol_b:
                fig_b = draw_form_sequence(h2h, player_b)
                st.plotly_chart(fig_b, use_container_width=True)

            st.subheader("Match History")
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
            st.dataframe(
                h2h[available_cols].sort_values("match_date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Rank Progression")
            rank_a = get_player_rank_history(player_a)
            rank_a["player_id"] = player_a
            rank_b = get_player_rank_history(player_b)
            rank_b["player_id"] = player_b
            rank_both = pd.concat([rank_a, rank_b])

            rank_fig = px.line(
                rank_both,
                x="match_date",
                y="player_ranking",
                color="player_id",
                markers=True,
                color_discrete_map={player_a: "#3498db", player_b: "#e67e22"},
            )
            rank_fig.update_layout(
                yaxis={"autorange": "reversed", "title": "Ranking"},
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(rank_fig, use_container_width=True)

# ── Player Explorer Tab ──────────────────────────────────────

with tab_explorer:
    player = st.selectbox("Select Player", players, key="explore")

    if player:
        rank_history = get_player_rank_history(player)

        st.subheader("Rank Over Time")
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
        st.plotly_chart(rank_fig, use_container_width=True)

        st.subheader("Win Rate by Surface")
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
        st.plotly_chart(surface_fig, use_container_width=True)

        st.subheader("Recent Matches")
        match_history = get_player_match_history(player)
        st.dataframe(match_history, use_container_width=True, hide_index=True)
