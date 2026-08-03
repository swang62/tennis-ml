-- Surface isolation: a surface-specific win rate changes ONLY on a snapshot
-- whose match is on that surface. For snapshots after the first
-- (player_match_number >= 2, so the LAG has a non-NULL prior), the rates for
-- surfaces NOT played in this match must equal the previous snapshot's value.
-- The first snapshot is skipped: it has no prior (LAG is NULL) and legitimately
-- initializes its own-surface rate from this match.
WITH ordered AS (
    SELECT
        player_id,
        player_match_number,
        surface,
        clay_win_rate_10,
        grass_win_rate_10,
        hard_win_rate_10,
        LAG(clay_win_rate_10)  OVER (PARTITION BY player_id ORDER BY player_match_number) AS prev_clay,
        LAG(grass_win_rate_10) OVER (PARTITION BY player_id ORDER BY player_match_number) AS prev_grass,
        LAG(hard_win_rate_10)  OVER (PARTITION BY player_id ORDER BY player_match_number) AS prev_hard
    FROM {{ ref('rolling_features') }}
)
SELECT player_id, player_match_number, surface
FROM ordered
WHERE player_match_number >= 2
  AND (
        (surface <> 'clay'  AND clay_win_rate_10  IS DISTINCT FROM prev_clay)
     OR (surface <> 'grass' AND grass_win_rate_10 IS DISTINCT FROM prev_grass)
     OR (surface <> 'hard'  AND hard_win_rate_10  IS DISTINCT FROM prev_hard)
  )
