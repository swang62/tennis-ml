-- gold.match_features: canonical match-level training table.
--
-- ONE ROW PER MATCH (not per player). Each player-match row is paired with
-- that player's IMMEDIATELY PRECEDING post-match snapshot
-- (player_match_number - 1) from gold.rolling_features, so all rolling
-- history values come strictly from completed matches BEFORE the current
-- event. The current event (silver.player_matches) supplies CURRENT rankings,
-- rank points, age, and the correct pre-match 30-day activity count.
-- tournament / round / is_indoor are joined from bronze by match_id (kept out
-- of silver.player_matches). The two perspectives then collapse into one
-- canonical row: the lower/stable ATP id is the `player_*` side, the other is
-- `opponent_*`, so the table is order-invariant (swapping player1/player2 in
-- bronze changes nothing).
--
-- match_won = 1 iff the canonical player_* side won, else 0. It is the LABEL,
-- not a feature.
--
-- Head-to-head (player_h2h_matches / player_h2h_wins): pair-level aggregates
-- over prior meetings between the canonical pair (strictly before this match's
-- date, deduped to distinct match_ids, most recent 5 only). The final contract
-- keeps only the canonical player side's counts; the opponent perspective and
-- the win rate are derived on demand.
--
-- Cold start: a player's first match has no prior snapshot, so all rolling
-- features are NULL (no zero filling). The only documented fallback is
-- days_since_last_match = 365; matches_30d is naturally 0 (it is a COUNT).
--
-- Rankings are never a row filter: unranked players (rank NULL after the
-- 0 -> NULL mapping in silver.player_matches) are kept, so matches are never
-- silently dropped for missing rankings. NULL rankings, rank_trend, and
-- rank_diff are imputed at train time (median) alongside the other NULLs.
--
-- Task 6: current-match per-side serve/break analysis rates are REMOVED from
-- this contract. Where the dashboard/analysis needs them, they are derived on
-- demand from bronze raw counts with the existing NULLIF zero-denominator
-- behavior. height is NOT a model feature here (it stays in gold.player_profiles
-- for the profile API); only is_left_handed and years_pro are model profile
-- features. The output columns are exactly FEATURE_COLS (36) plus the metadata
-- (match_id, match_date, player_id, opponent_id, tournament, round, surface,
-- match_won), in that order.

WITH player_match_enriched AS (
    SELECT
        pm.match_id,
        pm.match_date,
        bron.match_date       AS bronze_date,
        bron.tournament,
        bron.round,
        pm.surface,
        bron.is_indoor,
        pm.player_id,
        pm.opponent_id,
        pm.match_won,
        -- CURRENT event rankings/rank points/age, not the prior snapshot's
        pm.player_ranking,
        pm.opponent_ranking,
        pm.player_rank_points,
        pm.player_age,
        pm.player_match_number,

        -- Correct pre-match activity count (0 on first match; it is a COUNT)
        CAST(pm.matches_30d_before AS INTEGER) AS matches_30d,

        -- Days since the player's previous completed match; 365 on first match
        CAST(COALESCE(DATEDIFF('day', pr.snapshot_date, pm.match_date), 365)
            AS INTEGER) AS days_since_last_match,

        -- Rolling form from the immediately preceding snapshot (NULL on cold
        -- start — no zero filling)
        pr.weighted_form_10,
        pr.win_rate_10,
        pr.ace_rate_10,
        pr.first_serve_pct_10,
        pr.break_points_saved_pct_10,
        pr.first_serve_win_pct_10,
        pr.second_serve_win_pct_10,
        pr.serve_win_pct_10,
        pr.df_rate_10,
        pr.aces_per_svc_game_10,

        -- Rank trend: prior rolling avg ranking minus CURRENT event ranking
        pr.avg_player_rank_10 - pm.player_ranking AS rank_trend_10,

        -- Strength of schedule: prior snapshot's rolling average opponent rank
        pr.avg_rank_faced_10,

        CAST(pr.streak AS INTEGER) AS streak,

        -- Surface-specific form: prior snapshot's rate on the current surface
        CASE pm.surface
            WHEN 'clay'  THEN pr.clay_win_rate_10
            WHEN 'grass' THEN pr.grass_win_rate_10
            WHEN 'hard'  THEN pr.hard_win_rate_10
        END AS surface_win_rate_10,

        -- Profile-derived identity (static; NULL when the player has no
        -- profile or a cell is missing, same no-zero-filling policy as the
        -- rolling features — train-time imputation handles the NULLs)
        -- Non-L/R handedness (including NULL) stays NULL so train-time
        -- imputation treats missing handedness like any other missing cell
        CAST(CASE WHEN prof.handedness = 'L' THEN 1
                  WHEN prof.handedness = 'R' THEN 0 END AS UTINYINT)
            AS is_left_handed,
        -- Years pro AT THIS MATCH (time-aware), not the raw turned_pro year
        CAST(YEAR(pm.match_date) - prof.turned_pro AS INTEGER) AS years_pro

    FROM {{ ref('player_matches') }} pm
    LEFT JOIN {{ ref('rolling_features') }} pr
        ON pr.player_id = pm.player_id
       AND pr.player_match_number = pm.player_match_number - 1
    LEFT JOIN {{ source('bronze', 'match_events') }} bron
        ON bron.match_id = pm.match_id
    LEFT JOIN gold.player_profiles prof
        ON prof.player_id = pm.player_id
),
-- One row per distinct match between a canonical pair. silver.player_matches
-- has TWO rows per match (one per player perspective); this dedupes to match
-- level. a/b are the canonical ids (lower id is `a`), and a_won is 1 iff the
-- canonical a-side won that meeting — both perspective rows of a match agree
-- on it (the loser's row reports match_won = 1 - the winner's), so MAX is
-- safe. H2H aggregates read from here, never from raw silver rows.
pair_meetings AS (
    SELECT
        CASE WHEN player_id < opponent_id THEN player_id ELSE opponent_id END AS a,
        CASE WHEN player_id < opponent_id THEN opponent_id ELSE player_id END AS b,
        match_id,
        match_date,
        MAX(CASE WHEN player_id < opponent_id THEN match_won
                 ELSE 1 - match_won END) AS a_won
    FROM {{ ref('player_matches') }}
    GROUP BY 1, 2, 3, 4
),
-- Prior-meeting aggregates for the canonical pair (p.player_id vs
-- p.opponent_id): matches with match_date STRICTLY BEFORE the current row's
-- match_date (same-date meetings excluded), deduped to distinct match_ids,
-- restricted to the MOST RECENT 5 meetings (ORDER BY match_date DESC,
-- match_id DESC LIMIT 5 — the locked last-5 recency window). The INNER JOIN
-- already keeps only the canonical p-side perspective of each current match
-- (the o-side row's player_id is the higher id, so no meeting matches it), so
-- the window partitions cleanly per current match. Matches with no prior
-- meeting produce no row here; the final select COALESCEs to 0 matches / 0 wins.
prior_meeting_rows AS (
    SELECT
        current_match.match_id,
        meeting.a_won,
        ROW_NUMBER() OVER (
            PARTITION BY current_match.match_id
            ORDER BY meeting.match_date DESC, meeting.match_id DESC
        ) AS rn
    FROM player_match_enriched current_match
    JOIN pair_meetings meeting
        ON meeting.a = current_match.player_id
       AND meeting.b = current_match.opponent_id
       AND meeting.match_date < current_match.match_date
),
prior_h2h AS (
    SELECT
        match_id,
        COUNT(*) AS player_h2h_matches,
        COALESCE(SUM(a_won), 0) AS player_h2h_wins
    FROM prior_meeting_rows
    WHERE rn <= 5
    GROUP BY match_id
)
-- Collapse the two player-perspective rows of each match into one canonical
-- row: the lower ATP id is the player_* side (p), the higher is the
-- opponent_* side (o). Exactly one row per match, regardless of the raw
-- player1/player2 assignment. Output columns follow FEATURE_COLS order after
-- the metadata block.
SELECT
    p.match_id,
    p.match_date,
    p.player_id,
    p.opponent_id,
    p.tournament,
    p.round,
    p.surface,
    p.match_won,

    -- ── Matchup differences (canonical side minus opponent) ──
    p.player_ranking - o.player_ranking AS rank_diff,
    p.player_rank_points - o.player_rank_points AS rank_points_diff,
    p.player_age - o.player_age AS age_diff,
    p.win_rate_10 - o.win_rate_10 AS win_rate_diff,
    p.ace_rate_10 - o.ace_rate_10 AS ace_rate_diff,
    p.first_serve_pct_10 - o.first_serve_pct_10 AS first_serve_pct_diff,
    p.break_points_saved_pct_10 - o.break_points_saved_pct_10
        AS break_points_saved_pct_diff,
    p.first_serve_win_pct_10 - o.first_serve_win_pct_10
        AS first_serve_win_pct_diff,
    p.second_serve_win_pct_10 - o.second_serve_win_pct_10
        AS second_serve_win_pct_diff,
    p.serve_win_pct_10 - o.serve_win_pct_10 AS serve_win_pct_diff,
    p.df_rate_10 - o.df_rate_10 AS df_rate_diff,
    p.aces_per_svc_game_10 - o.aces_per_svc_game_10 AS aces_per_svc_game_diff,
    p.rank_trend_10 - o.rank_trend_10 AS rank_trend_diff,
    p.avg_rank_faced_10 - o.avg_rank_faced_10 AS avg_rank_faced_diff,
    p.streak - o.streak AS streak_diff,

    -- ── Absolute state values where both sides matter ──
    p.weighted_form_10      AS player_weighted_form_10,
    o.weighted_form_10      AS opponent_weighted_form_10,
    p.days_since_last_match AS player_days_since_last_match,
    o.days_since_last_match AS opponent_days_since_last_match,
    p.matches_30d           AS player_matches_30d,
    o.matches_30d           AS opponent_matches_30d,
    p.surface_win_rate_10   AS player_surface_win_rate_10,
    o.surface_win_rate_10   AS opponent_surface_win_rate_10,
    p.is_left_handed        AS player_is_left_handed,
    o.is_left_handed        AS opponent_is_left_handed,
    p.years_pro             AS player_years_pro,
    o.years_pro             AS opponent_years_pro,

    -- ── Canonical-player head-to-head history (0/0 when never met) ──
    COALESCE(h.player_h2h_matches, 0) AS player_h2h_matches,
    COALESCE(h.player_h2h_wins, 0)    AS player_h2h_wins,

    -- ── Numeric match context (one-hot surface for linear and neural models) ──
    CAST(CASE WHEN p.surface = 'clay'  THEN 1 ELSE 0 END AS UTINYINT) AS is_clay,
    CAST(CASE WHEN p.surface = 'grass' THEN 1 ELSE 0 END AS UTINYINT) AS is_grass,
    CAST(CASE WHEN p.surface = 'hard'  THEN 1 ELSE 0 END AS UTINYINT) AS is_hard,
    CAST(CASE WHEN p.surface = 'carpet' THEN 1 ELSE 0 END AS UTINYINT) AS is_carpet,
    b.is_indoor,
    CAST(CASE p.tournament
        WHEN 'grand_slam' THEN 4 WHEN 'masters' THEN 3
        WHEN 'atp_500' THEN 2 WHEN 'atp_250' THEN 1 ELSE 0
    END AS UTINYINT) AS tournament_level,
    CAST(CASE p.round
        WHEN 'r128' THEN 1 WHEN 'r64' THEN 2 WHEN 'r32' THEN 3 WHEN 'r16' THEN 4
        WHEN 'qf' THEN 5 WHEN 'sf' THEN 6 WHEN 'f' THEN 7 ELSE 0
    END AS UTINYINT) AS round_encoded
FROM player_match_enriched p
JOIN player_match_enriched o
    ON o.match_id = p.match_id
   AND o.player_id = p.opponent_id
LEFT JOIN bronze.match_events b
    ON b.match_id = p.match_id
LEFT JOIN prior_h2h h
    ON h.match_id = p.match_id
WHERE p.player_id < o.player_id
ORDER BY p.match_date, p.match_id