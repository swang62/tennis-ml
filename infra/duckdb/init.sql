CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS bronze.match_events (
    match_id           VARCHAR,
    match_date         DATE,
    player_id          VARCHAR,
    opponent_id        VARCHAR,
    tournament         VARCHAR,
    round              VARCHAR,
    surface            VARCHAR,
    player_ranking     INTEGER,
    opponent_ranking   INTEGER,
    wins_last_10       UTINYINT,
    matches_last_10    UTINYINT,
    aces               UTINYINT,
    double_faults      UTINYINT,
    first_serves_made  UTINYINT,
    total_serve_points UTINYINT,
    break_points_won   UTINYINT,
    break_points_total UTINYINT,
    match_won          UTINYINT
);

CREATE TABLE IF NOT EXISTS gold.match_features (
    match_id                 VARCHAR,
    match_date               DATE,
    player_id                VARCHAR,
    opponent_id              VARCHAR,
    tournament               VARCHAR,
    round                    VARCHAR,
    surface                  VARCHAR,
    player_ranking           INTEGER,
    opponent_ranking         INTEGER,
    wins_last_10             UTINYINT,
    matches_last_10          UTINYINT,
    aces                     UTINYINT,
    double_faults            UTINYINT,
    first_serves_made        UTINYINT,
    total_serve_points       UTINYINT,
    break_points_won         UTINYINT,
    break_points_total       UTINYINT,
    match_won                UTINYINT,

    -- Base rates
    win_rate_last_10         DOUBLE,
    ace_rate                 DOUBLE,
    double_fault_rate        DOUBLE,
    first_serve_pct          DOUBLE,
    break_points_converted_pct DOUBLE,

    -- Rolling features (pre-computed via window functions)
    win_rate_5               DOUBLE,
    win_rate_10              DOUBLE,
    win_rate_20              DOUBLE,
    ace_rate_5               DOUBLE,
    ace_rate_10              DOUBLE,
    first_serve_pct_5        DOUBLE,
    first_serve_pct_10       DOUBLE,
    break_pct_5              DOUBLE,
    break_pct_10             DOUBLE,
    avg_opp_rank_10          DOUBLE,
    avg_opp_rank_20          DOUBLE,
    rank_trend_10            DOUBLE,
    rank_trend_20            DOUBLE,
    win_streak               UTINYINT,
    days_since_last_match    INTEGER,
    matches_30d              UTINYINT,
    surface_win_rate_10      DOUBLE,

    -- Context
    is_clay                  UTINYINT,
    is_grass                 UTINYINT,
    is_hard                  UTINYINT
);

CREATE TABLE IF NOT EXISTS gold.player_profiles (
    player_id    VARCHAR PRIMARY KEY,
    display_name VARCHAR,
    summary      VARCHAR,
    handedness   VARCHAR,
    backhand     VARCHAR,
    play_style   VARCHAR,
    height       VARCHAR,
    turned_pro   INTEGER
);
