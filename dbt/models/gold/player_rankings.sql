-- Player ranking observations derived ONLY from bronze.match_events: each
-- raw match (one row holding both players) expands into two ranking rows,
-- one per player at that match's date. This is the ranking time series;
-- gold.match_features keeps ranking only as a per-match feature column.
WITH expanded AS (
    SELECT
        match_id, match_date,
        player1_id AS player_id,
        player1_ranking AS ranking
    FROM {{ source('bronze', 'match_events') }}

    UNION ALL

    SELECT
        match_id, match_date,
        player2_id AS player_id,
        player2_ranking AS ranking
    FROM {{ source('bronze', 'match_events') }}
)
SELECT match_id, match_date, player_id, ranking
FROM expanded
WHERE ranking IS NOT NULL AND ranking > 0
ORDER BY match_date, player_id, match_id
