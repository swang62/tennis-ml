-- Assert each match_features row's rolling features come from the player's
-- PRIOR snapshot only (player_match_number = current match number - 1), for
-- both the canonical player and the opponent side. Re-derive the prior
-- snapshot's win_rate_10 and compare against the stored value: any mismatch
-- means the row used the current match's snapshot (leakage) or a wrong one.
WITH pm_canonical AS (
    SELECT
        mf.match_id,
        mf.player_id,
        mf.player_win_rate_10 AS player_mf_val,
        mf.opponent_id,
        mf.opponent_win_rate_10 AS opponent_mf_val,
        pm.player_match_number AS player_cur_num,
        po.player_match_number AS opponent_cur_num
    FROM {{ ref('match_features') }} mf
    JOIN {{ ref('player_matches') }} pm
      ON pm.match_id = mf.match_id AND pm.player_id = mf.player_id
    JOIN {{ ref('player_matches') }} po
      ON po.match_id = mf.match_id AND po.player_id = mf.opponent_id
),
prior_snapshot AS (
    SELECT
        pmc.match_id,
        pmc.player_id,
        pmc.player_mf_val,
        pmc.opponent_mf_val,
        prp.win_rate_10 AS player_prior_val,
        pro.win_rate_10 AS opponent_prior_val
    FROM pm_canonical pmc
    LEFT JOIN {{ ref('rolling_features') }} prp
      ON prp.player_id = pmc.player_id
     AND prp.player_match_number = pmc.player_cur_num - 1
    LEFT JOIN {{ ref('rolling_features') }} pro
      ON pro.player_id = pmc.opponent_id
     AND pro.player_match_number = pmc.opponent_cur_num - 1
)
SELECT
    match_id,
    player_id,
    player_mf_val,
    opponent_mf_val,
    player_prior_val,
    opponent_prior_val
FROM prior_snapshot
WHERE player_mf_val IS DISTINCT FROM player_prior_val
   OR opponent_mf_val IS DISTINCT FROM opponent_prior_val
