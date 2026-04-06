-- =========================================================
-- New Feature Engineering (Phase 5) — Style Matchup Features
-- Adds 6 new decay features focused on style matchup signal
--
-- New features (all as f - opp diffs):
--   td_per_sigstr_att_dec_avg_diff   tdatt/sigstratt  (grappler vs striker style)
--   td_land_per_ctrl_dec_avg_diff    tdacc/ctrl_time  (grappling efficiency)
--   ground_land_per_ctrl_dec_avg_diff groundacc/ctrl_time (ground output per ctrl)
--   sub_att_ratio_dec_avg_diff       subatt per fight count
--   opp_decision_rate_dec_avg_diff   (opp_udec+opp_sdec+opp_mdec) per fight
--   rev_per_ctrlopp_dec_avg_diff    rev/opp_ctrl_time (reversal efficiency)
--
-- Note: ctrl_time = raw_ctrl_per_min * fight_time_minutes (per-fight ctrl minutes)
--
-- Run AFTER: ... → new_features2.sql → new_adjperf2.sql
-- ETR: ~2-3 minutes
-- =========================================================

DROP TABLE IF EXISTS new_features3_dec_avg;

CREATE TABLE new_features3_dec_avg AS
WITH new_precomp3 AS (
    SELECT
        p.jfighter,
        p.event_date,
        p.event_jd,
        p.jevent,
        p.jbout,
        p.opp_jfighter,

        -- 1. td_per_sigstr_att = SUM(tdatt)/SUM(sigstratt) prior fights
        --    Measures grappler vs striker style (higher = more takedown-oriented)
        CASE WHEN SUM(p.sigstratt) OVER wu1 = 0 OR SUM(p.sigstratt) OVER wu1 IS NULL THEN NULL
             ELSE 1.0 * SUM(p.tdatt) OVER wu1 / SUM(p.sigstratt) OVER wu1
        END AS prc_tdpsa,

        -- 2. td_land_per_ctrl = SUM(tdacc)/SUM(ctrl_time) prior fights
        --    ctrl_time per fight = raw_ctrl_per_min * fight_time_minutes
        CASE WHEN SUM(p.raw_ctrl_per_min * p.fight_time_minutes) OVER wu1 = 0
              OR SUM(p.raw_ctrl_per_min * p.fight_time_minutes) OVER wu1 IS NULL THEN NULL
             ELSE 1.0 * SUM(p.tdacc) OVER wu1
                      / SUM(p.raw_ctrl_per_min * p.fight_time_minutes) OVER wu1
        END AS prc_tdpc,

        -- 3. ground_land_per_ctrl = SUM(groundacc)/SUM(ctrl_time) prior fights
        CASE WHEN SUM(p.raw_ctrl_per_min * p.fight_time_minutes) OVER wu1 = 0
              OR SUM(p.raw_ctrl_per_min * p.fight_time_minutes) OVER wu1 IS NULL THEN NULL
             ELSE 1.0 * SUM(p.groundacc) OVER wu1
                      / SUM(p.raw_ctrl_per_min * p.fight_time_minutes) OVER wu1
        END AS prc_grpc,

        -- 4. sub_att_ratio = SUM(subatt)/COUNT(*) prior fights (sub attempts per fight)
        CASE WHEN COUNT(*) OVER wu1 = 0 OR COUNT(*) OVER wu1 IS NULL THEN NULL
             ELSE 1.0 * SUM(p.subatt) OVER wu1 / COUNT(*) OVER wu1
        END AS prc_subr,

        -- 5. opp_decision_rate = SUM(opp_udec+opp_sdec+opp_mdec)/COUNT(*) prior fights
        CASE WHEN COUNT(*) OVER wu1 = 0 OR COUNT(*) OVER wu1 IS NULL THEN NULL
             ELSE 1.0 * SUM(p.opp_udec + p.opp_sdec + p.opp_mdec) OVER wu1 / COUNT(*) OVER wu1
        END AS prc_odec,

        -- 6. rev_per_ctrlopp = SUM(rev) / SUM(opp_ctrl/60.0) prior fights
        --    Measures ability to reverse from bottom position
        CASE WHEN SUM(ms_opp.ctrl / 60.0) OVER wu1 = 0
              OR SUM(ms_opp.ctrl / 60.0) OVER wu1 IS NULL THEN NULL
             ELSE 1.0 * SUM(ms.rev) OVER wu1
                      / SUM(ms_opp.ctrl / 60.0) OVER wu1
        END AS prc_rpc

    FROM pre_final_base_mat p
    JOIN ufc_fighter_match_stats ms
      ON ms.jevent = p.jevent AND ms.jbout = p.jbout AND ms.jfighter = p.jfighter
    JOIN ufc_fighter_match_stats ms_opp
      ON ms_opp.jevent = p.jevent AND ms_opp.jbout = p.jbout AND ms_opp.jfighter = p.opp_jfighter
    WHERE p.opp_jfighter IS NOT NULL
    WINDOW
        wu1 AS (PARTITION BY p.jfighter ORDER BY p.event_date, p.jevent, p.jbout
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),

ordered3 AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY jfighter
               ORDER BY event_date, jevent, jbout
           ) AS rn
    FROM new_precomp3
),

ema3 AS (

    -- ---- Seed row (rn = 1): initialize all 12 accumulators ----
    SELECT
        jfighter, rn,
        event_date, event_jd,
        jevent, jbout, opp_jfighter,

        CASE WHEN prc_tdpsa IS NULL THEN 0.0 ELSE prc_tdpsa END AS tdpsa_num,
        CASE WHEN prc_tdpsa IS NULL THEN 0.0 ELSE 1.0       END AS tdpsa_den,
        CASE WHEN prc_tdpc  IS NULL THEN 0.0 ELSE prc_tdpc  END AS tdpc_num,
        CASE WHEN prc_tdpc  IS NULL THEN 0.0 ELSE 1.0       END AS tdpc_den,
        CASE WHEN prc_grpc  IS NULL THEN 0.0 ELSE prc_grpc  END AS grpc_num,
        CASE WHEN prc_grpc  IS NULL THEN 0.0 ELSE 1.0       END AS grpc_den,
        CASE WHEN prc_subr  IS NULL THEN 0.0 ELSE prc_subr  END AS subr_num,
        CASE WHEN prc_subr  IS NULL THEN 0.0 ELSE 1.0       END AS subr_den,
        CASE WHEN prc_odec  IS NULL THEN 0.0 ELSE prc_odec  END AS odec_num,
        CASE WHEN prc_odec  IS NULL THEN 0.0 ELSE 1.0       END AS odec_den,
        CASE WHEN prc_rpc IS NULL THEN 0.0 ELSE prc_rpc END AS rpc_num,
        CASE WHEN prc_rpc IS NULL THEN 0.0 ELSE 1.0     END AS rpc_den

    FROM ordered3
    WHERE rn = 1

    UNION ALL

    -- ---- Recursive step: decay all 12 accumulators, add new fight ----
    SELECT
        o.jfighter, o.rn,
        o.event_date, o.event_jd,
        o.jevent, o.jbout, o.opp_jfighter,

        (e.tdpsa_num * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_tdpsa IS NULL THEN 0.0 ELSE o.prc_tdpsa END,
        (e.tdpsa_den * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_tdpsa IS NULL THEN 0.0 ELSE 1.0       END,
        (e.tdpc_num  * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_tdpc  IS NULL THEN 0.0 ELSE o.prc_tdpc  END,
        (e.tdpc_den  * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_tdpc  IS NULL THEN 0.0 ELSE 1.0       END,
        (e.grpc_num  * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_grpc  IS NULL THEN 0.0 ELSE o.prc_grpc  END,
        (e.grpc_den  * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_grpc  IS NULL THEN 0.0 ELSE 1.0       END,
        (e.subr_num  * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_subr  IS NULL THEN 0.0 ELSE o.prc_subr  END,
        (e.subr_den  * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_subr  IS NULL THEN 0.0 ELSE 1.0       END,
        (e.odec_num  * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_odec  IS NULL THEN 0.0 ELSE o.prc_odec  END,
        (e.odec_den  * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_odec  IS NULL THEN 0.0 ELSE 1.0       END,
        (e.rpc_num * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_rpc IS NULL THEN 0.0 ELSE o.prc_rpc END,
        (e.rpc_den * exp(-0.462 * ((o.event_jd - e.event_jd)/365.25))) + CASE WHEN o.prc_rpc IS NULL THEN 0.0 ELSE 1.0     END

    FROM ordered3 o
    JOIN ema3 e ON e.jfighter = o.jfighter AND e.rn = o.rn - 1
)

-- ---- Final output: divide num/den to get decay-weighted averages ----
SELECT
    jfighter,
    event_date AS DATE,
    jevent,
    jbout,
    opp_jfighter,
    CASE WHEN tdpsa_den = 0 THEN NULL ELSE tdpsa_num / tdpsa_den END AS td_per_sigstr_att_dec_avg,
    CASE WHEN tdpc_den  = 0 THEN NULL ELSE tdpc_num  / tdpc_den  END AS td_land_per_ctrl_dec_avg,
    CASE WHEN grpc_den  = 0 THEN NULL ELSE grpc_num  / grpc_den  END AS ground_land_per_ctrl_dec_avg,
    CASE WHEN subr_den  = 0 THEN NULL ELSE subr_num  / subr_den  END AS sub_att_ratio_dec_avg,
    CASE WHEN odec_den  = 0 THEN NULL ELSE odec_num  / odec_den  END AS opp_decision_rate_dec_avg,
    CASE WHEN rpc_den = 0 THEN NULL ELSE rpc_num / rpc_den END AS rev_per_ctrlopp_dec_avg
FROM ema3;

CREATE INDEX IF NOT EXISTS idx_nf3_fighter_bout
ON new_features3_dec_avg(jevent, jbout, jfighter);

-- =========================================================
-- NOTE: final_features_fast rebuild removed from this file.
-- The complete rebuild is handled by ufc_sql_final_rebuild.sql
-- which runs after ALL feature/adjperf tables are created.
-- =========================================================
