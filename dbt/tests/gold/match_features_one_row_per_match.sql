-- Assert one canonical row per match in match_features. Defensive duplicate
-- of the yml unique test on match_id: catches duplicates regardless of
-- nullability.
SELECT match_id FROM {{ ref('match_features') }} GROUP BY match_id HAVING COUNT(*) > 1
