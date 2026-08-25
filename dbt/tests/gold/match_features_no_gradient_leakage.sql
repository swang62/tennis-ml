{#- Causal-leakage guard for the Elo-gradient features.

    Re-derives player_elo_gradient_10 / opponent_elo_gradient_10 from each
    player's own completed-Elo history that is *strictly before* the match
    under test (the same window the gold model uses) and asserts equality with
    the stored value.

    This fails if a future change lets the current match's own post_elo (which
    encodes the target outcome) or any same-day / future rating enter the
    gradient window, because the recomputed slope would then differ from the
    stored value. It also fails if the gradient is computed from pre_elo rather
    than post_elo, or from a window wider than the most recent ten.
#}
WITH sampled_matches AS (
    SELECT match_id FROM (
        (SELECT DISTINCT match_id, match_date FROM {{ ref('player_matches') }}
         ORDER BY match_date, match_id LIMIT 25)
        UNION ALL
        (SELECT DISTINCT match_id, match_date FROM {{ ref('player_matches') }}
         ORDER BY match_date DESC, match_id DESC LIMIT 25)
    ) strata
    GROUP BY match_id
),
player_grad AS (
    SELECT
        pm.match_id,
        pm.player_id,
        CASE WHEN COUNT(g.post_elo) < 2 THEN 0.0
        ELSE (COUNT(g.post_elo) * SUM(g.idx * g.post_elo)
                 - SUM(g.idx) * SUM(g.post_elo))
             / (COUNT(g.post_elo) * SUM(g.idx * g.idx)
                 - SUM(g.idx) * SUM(g.idx))
        END AS exp_grad
    FROM {{ ref('player_matches') }} pm
    LEFT JOIN LATERAL (
        SELECT s.post_elo,
               ROW_NUMBER() OVER (
                   ORDER BY s.match_date, s.match_num, s.match_id
               ) AS idx
        FROM {{ source('silver', 'elo_snapshots') }} s
        WHERE s.player_id = pm.player_id
          AND (s.match_date, s.match_num, s.match_id)
              < (pm.match_date, pm.match_num, pm.match_id)
        ORDER BY s.match_date DESC, s.match_num DESC, s.match_id DESC
        LIMIT 10
    ) g ON true
    WHERE pm.match_id IN (SELECT match_id FROM sampled_matches)
    GROUP BY pm.match_id, pm.player_id
),
opponent_grad AS (
    SELECT
        pm.match_id,
        pm.player_id,
        CASE WHEN COUNT(g.post_elo) < 2 THEN 0.0
        ELSE (COUNT(g.post_elo) * SUM(g.idx * g.post_elo)
                 - SUM(g.idx) * SUM(g.post_elo))
             / (COUNT(g.post_elo) * SUM(g.idx * g.idx)
                 - SUM(g.idx) * SUM(g.idx))
        END AS exp_grad
    FROM {{ ref('player_matches') }} pm
    LEFT JOIN LATERAL (
        SELECT s.post_elo,
               ROW_NUMBER() OVER (
                   ORDER BY s.match_date, s.match_num, s.match_id
               ) AS idx
        FROM {{ source('silver', 'elo_snapshots') }} s
        WHERE s.player_id = pm.opponent_id
          AND (s.match_date, s.match_num, s.match_id)
              < (pm.match_date, pm.match_num, pm.match_id)
        ORDER BY s.match_date DESC, s.match_num DESC, s.match_id DESC
        LIMIT 10
    ) g ON true
    WHERE pm.match_id IN (SELECT match_id FROM sampled_matches)
    GROUP BY pm.match_id, pm.player_id
),
violations AS (
    SELECT mf.match_id, mf.player_id, mf.opponent_id,
           'player_elo_gradient_10' AS feature,
           mf.player_elo_gradient_10 AS actual,
           pg.exp_grad AS expected
    FROM {{ ref('match_features') }} mf
    JOIN player_grad pg
      ON pg.match_id = mf.match_id AND pg.player_id = mf.player_id
    WHERE ABS(mf.player_elo_gradient_10 - pg.exp_grad) > 1e-6
    UNION ALL
    SELECT mf.match_id, mf.player_id, mf.opponent_id,
           'opponent_elo_gradient_10' AS feature,
           mf.opponent_elo_gradient_10 AS actual,
           og.exp_grad AS expected
    FROM {{ ref('match_features') }} mf
    JOIN opponent_grad og
      ON og.match_id = mf.match_id AND og.player_id = mf.player_id
    WHERE ABS(mf.opponent_elo_gradient_10 - og.exp_grad) > 1e-6
)
SELECT * FROM violations
ORDER BY match_id, player_id, feature
