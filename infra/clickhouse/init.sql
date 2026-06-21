CREATE DATABASE IF NOT EXISTS bronze;
CREATE DATABASE IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS bronze.match_events
(
    match_id           String,
    match_date         Date,
    player_id          String,
    opponent_id        String,
    tournament         String,
    round              String,
    surface            String,
    player_ranking     UInt32,
    opponent_ranking   UInt32,
    wins_last_10       UInt8,
    matches_last_10    UInt8,
    aces               UInt8,
    double_faults      UInt8,
    first_serves_made  UInt8,
    total_serve_points UInt8,
    break_points_won   UInt8,
    break_points_total UInt8,
    match_won          UInt8
)
ENGINE = MergeTree()
ORDER BY (match_date, match_id);

CREATE TABLE IF NOT EXISTS gold.match_features
(
    match_id                 String,
    match_date               Date,
    player_id                String,
    opponent_id              String,
    tournament               String,
    round                    String,
    surface                  String,
    player_ranking           UInt32,
    opponent_ranking         UInt32,
    wins_last_10             UInt8,
    matches_last_10          UInt8,
    aces                     UInt8,
    double_faults            UInt8,
    first_serves_made        UInt8,
    total_serve_points       UInt8,
    break_points_won         UInt8,
    break_points_total       UInt8,
    match_won                UInt8,

    -- Base rates
    win_rate_last_10         Float64,
    ace_rate                 Float64,
    double_fault_rate        Float64,
    first_serve_pct          Float64,
    break_points_converted_pct Float64,

    -- Rolling features (pre-computed via window functions)
    win_rate_5               Float64,
    win_rate_10              Float64,
    win_rate_20              Float64,
    ace_rate_5               Float64,
    ace_rate_10              Float64,
    first_serve_pct_5        Float64,
    first_serve_pct_10       Float64,
    break_pct_5              Float64,
    break_pct_10             Float64,
    avg_opp_rank_10          Float64,
    avg_opp_rank_20          Float64,
    rank_trend_10            Float64,
    rank_trend_20            Float64,
    win_streak               UInt8,
    days_since_last_match    Int32,
    matches_30d              UInt8,
    surface_win_rate_10      Float64,

    -- Context
    is_clay                  UInt8,
    is_grass                 UInt8,
    is_hard                  UInt8
)
ENGINE = MergeTree()
ORDER BY (match_date, match_id);

CREATE TABLE IF NOT EXISTS gold.player_profiles
(
    player_id    String,
    display_name String,
    summary      String,
    handedness   String,
    backhand     String,
    play_style   String,
    height       String,
    turned_pro   UInt32
)
ENGINE = MergeTree()
PRIMARY KEY (player_id)
ORDER BY (player_id);
