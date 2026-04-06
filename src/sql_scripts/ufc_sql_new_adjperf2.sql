-- =========================================================
-- New Adjperf2 Features (Phase 4)
-- 24 adjperf z-score features
--
-- Per-minute striking (6):
--   adjperf_head_pm_dec_avg, adjperf_body_pm_dec_avg,
--   adjperf_leg_pm_dec_avg,  adjperf_dist_pm_dec_avg,
--   adjperf_clinch_pm_dec_avg, adjperf_ground_pm_dec_avg
--
-- Accuracy percentages (6):
--   adjperf_legacc_dec_avg, adjperf_clinchacc_dec_avg,
--   adjperf_groundacc_dec_avg, adjperf_sigstracc_dec_avg,
--   adjperf_totalstracc_dec_avg, adjperf_tdacc_dec_avg
--
-- Defense (7):
--   adjperf_headdef_dec_avg, adjperf_bodydef_dec_avg,
--   adjperf_legdef_dec_avg, adjperf_clinchdef_dec_avg,
--   adjperf_sigstrdef_dec_avg, adjperf_tddef_dec_avg,
--   adjperf_totalstrdef_dec_avg
--
-- Zone ratios (5):
--   adjperf_bodyratio_dec_avg, adjperf_legratio_dec_avg,
--   adjperf_clinch_ratio_dec_avg, adjperf_ground_ratio_dec_avg,
--   adjperf_dist_ratio_dec_avg
--
-- Methodology: same as adjperf_v2.sql / ufc_sql_new_adjperf.sql
--   raw per-fight stat → z-score (Bayesian K=5, PARTITION BY opp_jfighter)
--   → EMA-decay (λ=0.13, ~5yr half-life)
--   → opp filter: opponent must have >= 2 prior UFC fights
--
-- Run AFTER: ... → new_features2.sql
-- =========================================================

DROP TABLE IF EXISTS adjperf_new2_fast;

CREATE TABLE adjperf_new2_fast AS
WITH RECURSIVE

-- 1. Opponent prior fight count (>= 2 prior fights filter)
opp_history AS (
    SELECT jfighter, jevent, jbout,
           ROW_NUMBER() OVER (PARTITION BY jfighter ORDER BY DATE, jevent, jbout) - 1 AS n_prior
    FROM decayed_precomp_v2_fast
),

-- 2. Per-fight raw stats (qualifying fights only)
fight_data AS (
    SELECT p.DATE, p.jevent, p.jbout, p.jfighter, p.opp_jfighter, p.weightindex,
           julianday(p.DATE) AS jd,

           -- Group 1: Per-minute striking (6)
           CASE WHEN p.fight_time_minutes = 0 OR p.fight_time_minutes IS NULL THEN NULL
                ELSE 1.0 * p.headacc / p.fight_time_minutes END AS raw_hpm,
           CASE WHEN p.fight_time_minutes = 0 OR p.fight_time_minutes IS NULL THEN NULL
                ELSE 1.0 * p.bodyacc / p.fight_time_minutes END AS raw_bpm,
           CASE WHEN p.fight_time_minutes = 0 OR p.fight_time_minutes IS NULL THEN NULL
                ELSE 1.0 * p.legacc / p.fight_time_minutes END AS raw_lpm,
           CASE WHEN p.fight_time_minutes = 0 OR p.fight_time_minutes IS NULL THEN NULL
                ELSE 1.0 * p.distacc / p.fight_time_minutes END AS raw_dpm,
           CASE WHEN p.fight_time_minutes = 0 OR p.fight_time_minutes IS NULL THEN NULL
                ELSE 1.0 * p.clinchacc / p.fight_time_minutes END AS raw_cpm,
           CASE WHEN p.fight_time_minutes = 0 OR p.fight_time_minutes IS NULL THEN NULL
                ELSE 1.0 * p.groundacc / p.fight_time_minutes END AS raw_gpm,

           -- Group 2: Accuracy percentages (6)
           CASE WHEN p.legatt = 0 OR p.legatt IS NULL THEN NULL
                ELSE 1.0 * p.legacc / p.legatt END AS raw_lacc,
           CASE WHEN p.clinchatt = 0 OR p.clinchatt IS NULL THEN NULL
                ELSE 1.0 * p.clinchacc / p.clinchatt END AS raw_cacc,
           CASE WHEN p.groundatt = 0 OR p.groundatt IS NULL THEN NULL
                ELSE 1.0 * p.groundacc / p.groundatt END AS raw_gacc,
           CASE WHEN p.sigstratt = 0 OR p.sigstratt IS NULL THEN NULL
                ELSE 1.0 * p.sigstracc / p.sigstratt END AS raw_sacc,
           CASE WHEN p.totalatt = 0 OR p.totalatt IS NULL THEN NULL
                ELSE 1.0 * p.totalacc / p.totalatt END AS raw_tacc,
           CASE WHEN p.tdatt = 0 OR p.tdatt IS NULL THEN NULL
                ELSE 1.0 * p.tdacc / p.tdatt END AS raw_tdacc2,

           -- Group 3: Defense (7)
           CASE WHEN p.opp_headatt = 0 OR p.opp_headatt IS NULL THEN NULL
                ELSE 1.0 - 1.0 * p.opp_headacc / p.opp_headatt END AS raw_hdef,
           CASE WHEN p.opp_bodyatt = 0 OR p.opp_bodyatt IS NULL THEN NULL
                ELSE 1.0 - 1.0 * p.opp_bodyacc / p.opp_bodyatt END AS raw_bdef,
           CASE WHEN p.opp_legatt = 0 OR p.opp_legatt IS NULL THEN NULL
                ELSE 1.0 - 1.0 * p.opp_legacc / p.opp_legatt END AS raw_ldef,
           CASE WHEN p.opp_clinchatt = 0 OR p.opp_clinchatt IS NULL THEN NULL
                ELSE 1.0 - 1.0 * p.opp_clinchacc / p.opp_clinchatt END AS raw_cdef,
           CASE WHEN p.opp_sigstratt = 0 OR p.opp_sigstratt IS NULL THEN NULL
                ELSE 1.0 - 1.0 * p.opp_sigstracc / p.opp_sigstratt END AS raw_sdef,
           CASE WHEN p.opp_tdatt = 0 OR p.opp_tdatt IS NULL THEN NULL
                ELSE 1.0 - 1.0 * p.opp_tdacc / p.opp_tdatt END AS raw_tdef,
           CASE WHEN p.opp_totalatt = 0 OR p.opp_totalatt IS NULL THEN NULL
                ELSE 1.0 - 1.0 * p.opp_totalacc / p.opp_totalatt END AS raw_tsdef,

           -- Group 4: Zone ratios (5)
           CASE WHEN p.sigstracc = 0 OR p.sigstracc IS NULL THEN NULL
                ELSE 1.0 * p.bodyacc / p.sigstracc END AS raw_bratio,
           CASE WHEN p.sigstracc = 0 OR p.sigstracc IS NULL THEN NULL
                ELSE 1.0 * p.legacc / p.sigstracc END AS raw_lratio,
           CASE WHEN p.sigstracc = 0 OR p.sigstracc IS NULL THEN NULL
                ELSE 1.0 * p.clinchacc / p.sigstracc END AS raw_cratio,
           CASE WHEN p.sigstracc = 0 OR p.sigstracc IS NULL THEN NULL
                ELSE 1.0 * p.groundacc / p.sigstracc END AS raw_gratio,
           CASE WHEN p.sigstracc = 0 OR p.sigstracc IS NULL THEN NULL
                ELSE 1.0 * p.distacc / p.sigstracc END AS raw_dratio

    FROM pre_final_base_mat p
    JOIN opp_history o ON o.jfighter = p.opp_jfighter
                      AND o.jevent   = p.jevent
                      AND o.jbout    = p.jbout
    WHERE p.opp_jfighter IS NOT NULL
      AND o.n_prior >= 1
),

-- 3. Weight-class means (global prior)
wc_mean AS (
    SELECT weightindex,
        AVG(raw_hpm)    AS wc_hpm_mean,
        AVG(raw_bpm)    AS wc_bpm_mean,
        AVG(raw_lpm)    AS wc_lpm_mean,
        AVG(raw_dpm)    AS wc_dpm_mean,
        AVG(raw_cpm)    AS wc_cpm_mean,
        AVG(raw_gpm)    AS wc_gpm_mean,
        AVG(raw_lacc)   AS wc_lacc_mean,
        AVG(raw_cacc)   AS wc_cacc_mean,
        AVG(raw_gacc)   AS wc_gacc_mean,
        AVG(raw_sacc)   AS wc_sacc_mean,
        AVG(raw_tacc)   AS wc_tacc_mean,
        AVG(raw_tdacc2) AS wc_tdacc2_mean,
        AVG(raw_hdef)   AS wc_hdef_mean,
        AVG(raw_bdef)   AS wc_bdef_mean,
        AVG(raw_ldef)   AS wc_ldef_mean,
        AVG(raw_cdef)   AS wc_cdef_mean,
        AVG(raw_sdef)   AS wc_sdef_mean,
        AVG(raw_tdef)   AS wc_tdef_mean,
        AVG(raw_tsdef)  AS wc_tsdef_mean,
        AVG(raw_bratio) AS wc_bratio_mean,
        AVG(raw_lratio) AS wc_lratio_mean,
        AVG(raw_cratio) AS wc_cratio_mean,
        AVG(raw_gratio) AS wc_gratio_mean,
        AVG(raw_dratio) AS wc_dratio_mean
    FROM fight_data
    GROUP BY weightindex
),
wc_aad AS (
    SELECT d.weightindex,
        AVG(ABS(d.raw_hpm    - m.wc_hpm_mean))    AS wc_hpm_aad,
        AVG(ABS(d.raw_bpm    - m.wc_bpm_mean))    AS wc_bpm_aad,
        AVG(ABS(d.raw_lpm    - m.wc_lpm_mean))    AS wc_lpm_aad,
        AVG(ABS(d.raw_dpm    - m.wc_dpm_mean))    AS wc_dpm_aad,
        AVG(ABS(d.raw_cpm    - m.wc_cpm_mean))    AS wc_cpm_aad,
        AVG(ABS(d.raw_gpm    - m.wc_gpm_mean))    AS wc_gpm_aad,
        AVG(ABS(d.raw_lacc   - m.wc_lacc_mean))   AS wc_lacc_aad,
        AVG(ABS(d.raw_cacc   - m.wc_cacc_mean))   AS wc_cacc_aad,
        AVG(ABS(d.raw_gacc   - m.wc_gacc_mean))   AS wc_gacc_aad,
        AVG(ABS(d.raw_sacc   - m.wc_sacc_mean))   AS wc_sacc_aad,
        AVG(ABS(d.raw_tacc   - m.wc_tacc_mean))   AS wc_tacc_aad,
        AVG(ABS(d.raw_tdacc2 - m.wc_tdacc2_mean)) AS wc_tdacc2_aad,
        AVG(ABS(d.raw_hdef   - m.wc_hdef_mean))   AS wc_hdef_aad,
        AVG(ABS(d.raw_bdef   - m.wc_bdef_mean))   AS wc_bdef_aad,
        AVG(ABS(d.raw_ldef   - m.wc_ldef_mean))   AS wc_ldef_aad,
        AVG(ABS(d.raw_cdef   - m.wc_cdef_mean))   AS wc_cdef_aad,
        AVG(ABS(d.raw_sdef   - m.wc_sdef_mean))   AS wc_sdef_aad,
        AVG(ABS(d.raw_tdef   - m.wc_tdef_mean))   AS wc_tdef_aad,
        AVG(ABS(d.raw_tsdef  - m.wc_tsdef_mean))  AS wc_tsdef_aad,
        AVG(ABS(d.raw_bratio - m.wc_bratio_mean)) AS wc_bratio_aad,
        AVG(ABS(d.raw_lratio - m.wc_lratio_mean)) AS wc_lratio_aad,
        AVG(ABS(d.raw_cratio - m.wc_cratio_mean)) AS wc_cratio_aad,
        AVG(ABS(d.raw_gratio - m.wc_gratio_mean)) AS wc_gratio_aad,
        AVG(ABS(d.raw_dratio - m.wc_dratio_mean)) AS wc_dratio_aad
    FROM fight_data d
    JOIN wc_mean m ON d.weightindex = m.weightindex
    GROUP BY d.weightindex
),

-- 4. Assign per-opponent fight index
opp_ordered AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY opp_jfighter ORDER BY DATE, jevent, jbout) AS opp_rn
    FROM fight_data
),

-- 5. Rolling mean per opponent (all prior encounters, equal weight, leakage-free)
rolling_mean AS (
    SELECT *,
        AVG(raw_hpm)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_hpm_mean,
        COUNT(raw_hpm)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_hpm,
        AVG(raw_bpm)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_bpm_mean,
        COUNT(raw_bpm)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_bpm,
        AVG(raw_lpm)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_lpm_mean,
        COUNT(raw_lpm)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_lpm,
        AVG(raw_dpm)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_dpm_mean,
        COUNT(raw_dpm)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_dpm,
        AVG(raw_cpm)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_cpm_mean,
        COUNT(raw_cpm)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_cpm,
        AVG(raw_gpm)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_gpm_mean,
        COUNT(raw_gpm)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_gpm,
        AVG(raw_lacc)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_lacc_mean,
        COUNT(raw_lacc)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_lacc,
        AVG(raw_cacc)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_cacc_mean,
        COUNT(raw_cacc)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_cacc,
        AVG(raw_gacc)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_gacc_mean,
        COUNT(raw_gacc)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_gacc,
        AVG(raw_sacc)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_sacc_mean,
        COUNT(raw_sacc)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_sacc,
        AVG(raw_tacc)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tacc_mean,
        COUNT(raw_tacc)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_tacc,
        AVG(raw_tdacc2)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tdacc2_mean,
        COUNT(raw_tdacc2) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_tdacc2,
        AVG(raw_hdef)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_hdef_mean,
        COUNT(raw_hdef)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_hdef,
        AVG(raw_bdef)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_bdef_mean,
        COUNT(raw_bdef)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_bdef,
        AVG(raw_ldef)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_ldef_mean,
        COUNT(raw_ldef)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_ldef,
        AVG(raw_cdef)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_cdef_mean,
        COUNT(raw_cdef)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_cdef,
        AVG(raw_sdef)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_sdef_mean,
        COUNT(raw_sdef)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_sdef,
        AVG(raw_tdef)   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tdef_mean,
        COUNT(raw_tdef)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_tdef,
        AVG(raw_tsdef)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tsdef_mean,
        COUNT(raw_tsdef) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_tsdef,
        AVG(raw_bratio)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_bratio_mean,
        COUNT(raw_bratio) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_bratio,
        AVG(raw_lratio)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_lratio_mean,
        COUNT(raw_lratio) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_lratio,
        AVG(raw_cratio)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_cratio_mean,
        COUNT(raw_cratio) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_cratio,
        AVG(raw_gratio)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_gratio_mean,
        COUNT(raw_gratio) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_gratio,
        AVG(raw_dratio)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_dratio_mean,
        COUNT(raw_dratio) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_dratio
    FROM opp_ordered
),

-- 6. Rolling AAD per opponent (spread of prior encounters)
rolling_aad AS (
    SELECT *,
        AVG(ABS(raw_hpm    - opp_hpm_mean))    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_hpm_aad,
        AVG(ABS(raw_bpm    - opp_bpm_mean))    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_bpm_aad,
        AVG(ABS(raw_lpm    - opp_lpm_mean))    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_lpm_aad,
        AVG(ABS(raw_dpm    - opp_dpm_mean))    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_dpm_aad,
        AVG(ABS(raw_cpm    - opp_cpm_mean))    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_cpm_aad,
        AVG(ABS(raw_gpm    - opp_gpm_mean))    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_gpm_aad,
        AVG(ABS(raw_lacc   - opp_lacc_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_lacc_aad,
        AVG(ABS(raw_cacc   - opp_cacc_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_cacc_aad,
        AVG(ABS(raw_gacc   - opp_gacc_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_gacc_aad,
        AVG(ABS(raw_sacc   - opp_sacc_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_sacc_aad,
        AVG(ABS(raw_tacc   - opp_tacc_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tacc_aad,
        AVG(ABS(raw_tdacc2 - opp_tdacc2_mean)) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tdacc2_aad,
        AVG(ABS(raw_hdef   - opp_hdef_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_hdef_aad,
        AVG(ABS(raw_bdef   - opp_bdef_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_bdef_aad,
        AVG(ABS(raw_ldef   - opp_ldef_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_ldef_aad,
        AVG(ABS(raw_cdef   - opp_cdef_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_cdef_aad,
        AVG(ABS(raw_sdef   - opp_sdef_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_sdef_aad,
        AVG(ABS(raw_tdef   - opp_tdef_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tdef_aad,
        AVG(ABS(raw_tsdef  - opp_tsdef_mean))  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tsdef_aad,
        AVG(ABS(raw_bratio - opp_bratio_mean)) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_bratio_aad,
        AVG(ABS(raw_lratio - opp_lratio_mean)) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_lratio_aad,
        AVG(ABS(raw_cratio - opp_cratio_mean)) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_cratio_aad,
        AVG(ABS(raw_gratio - opp_gratio_mean)) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_gratio_aad,
        AVG(ABS(raw_dratio - opp_dratio_mean)) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_dratio_aad
    FROM rolling_mean
),

-- 7. Per-fight z-scores (Bayesian K=5, floor=0.05)
z_scores AS (
    SELECT
        r.DATE, r.jevent, r.jbout, r.jfighter, r.weightindex, r.jd,
        ROW_NUMBER() OVER (PARTITION BY r.jfighter ORDER BY r.DATE, r.jevent, r.jbout) AS rn,

        -- z_hpm
        CASE WHEN r.raw_hpm IS NULL THEN NULL ELSE
            (r.raw_hpm - (r.n_hpm/(r.n_hpm+5.0))*COALESCE(r.opp_hpm_mean,m.wc_hpm_mean) - (5.0/(r.n_hpm+5.0))*m.wc_hpm_mean) /
            CASE WHEN ((r.n_hpm/(r.n_hpm+5.0))*COALESCE(r.opp_hpm_aad,a.wc_hpm_aad)+(5.0/(r.n_hpm+5.0))*a.wc_hpm_aad) IS NULL THEN NULL
                 WHEN ((r.n_hpm/(r.n_hpm+5.0))*COALESCE(r.opp_hpm_aad,a.wc_hpm_aad)+(5.0/(r.n_hpm+5.0))*a.wc_hpm_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_hpm/(r.n_hpm+5.0))*COALESCE(r.opp_hpm_aad,a.wc_hpm_aad)+(5.0/(r.n_hpm+5.0))*a.wc_hpm_aad) END
        END AS z_hpm,

        -- z_bpm
        CASE WHEN r.raw_bpm IS NULL THEN NULL ELSE
            (r.raw_bpm - (r.n_bpm/(r.n_bpm+5.0))*COALESCE(r.opp_bpm_mean,m.wc_bpm_mean) - (5.0/(r.n_bpm+5.0))*m.wc_bpm_mean) /
            CASE WHEN ((r.n_bpm/(r.n_bpm+5.0))*COALESCE(r.opp_bpm_aad,a.wc_bpm_aad)+(5.0/(r.n_bpm+5.0))*a.wc_bpm_aad) IS NULL THEN NULL
                 WHEN ((r.n_bpm/(r.n_bpm+5.0))*COALESCE(r.opp_bpm_aad,a.wc_bpm_aad)+(5.0/(r.n_bpm+5.0))*a.wc_bpm_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_bpm/(r.n_bpm+5.0))*COALESCE(r.opp_bpm_aad,a.wc_bpm_aad)+(5.0/(r.n_bpm+5.0))*a.wc_bpm_aad) END
        END AS z_bpm,

        -- z_lpm
        CASE WHEN r.raw_lpm IS NULL THEN NULL ELSE
            (r.raw_lpm - (r.n_lpm/(r.n_lpm+5.0))*COALESCE(r.opp_lpm_mean,m.wc_lpm_mean) - (5.0/(r.n_lpm+5.0))*m.wc_lpm_mean) /
            CASE WHEN ((r.n_lpm/(r.n_lpm+5.0))*COALESCE(r.opp_lpm_aad,a.wc_lpm_aad)+(5.0/(r.n_lpm+5.0))*a.wc_lpm_aad) IS NULL THEN NULL
                 WHEN ((r.n_lpm/(r.n_lpm+5.0))*COALESCE(r.opp_lpm_aad,a.wc_lpm_aad)+(5.0/(r.n_lpm+5.0))*a.wc_lpm_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_lpm/(r.n_lpm+5.0))*COALESCE(r.opp_lpm_aad,a.wc_lpm_aad)+(5.0/(r.n_lpm+5.0))*a.wc_lpm_aad) END
        END AS z_lpm,

        -- z_dpm
        CASE WHEN r.raw_dpm IS NULL THEN NULL ELSE
            (r.raw_dpm - (r.n_dpm/(r.n_dpm+5.0))*COALESCE(r.opp_dpm_mean,m.wc_dpm_mean) - (5.0/(r.n_dpm+5.0))*m.wc_dpm_mean) /
            CASE WHEN ((r.n_dpm/(r.n_dpm+5.0))*COALESCE(r.opp_dpm_aad,a.wc_dpm_aad)+(5.0/(r.n_dpm+5.0))*a.wc_dpm_aad) IS NULL THEN NULL
                 WHEN ((r.n_dpm/(r.n_dpm+5.0))*COALESCE(r.opp_dpm_aad,a.wc_dpm_aad)+(5.0/(r.n_dpm+5.0))*a.wc_dpm_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_dpm/(r.n_dpm+5.0))*COALESCE(r.opp_dpm_aad,a.wc_dpm_aad)+(5.0/(r.n_dpm+5.0))*a.wc_dpm_aad) END
        END AS z_dpm,

        -- z_cpm
        CASE WHEN r.raw_cpm IS NULL THEN NULL ELSE
            (r.raw_cpm - (r.n_cpm/(r.n_cpm+5.0))*COALESCE(r.opp_cpm_mean,m.wc_cpm_mean) - (5.0/(r.n_cpm+5.0))*m.wc_cpm_mean) /
            CASE WHEN ((r.n_cpm/(r.n_cpm+5.0))*COALESCE(r.opp_cpm_aad,a.wc_cpm_aad)+(5.0/(r.n_cpm+5.0))*a.wc_cpm_aad) IS NULL THEN NULL
                 WHEN ((r.n_cpm/(r.n_cpm+5.0))*COALESCE(r.opp_cpm_aad,a.wc_cpm_aad)+(5.0/(r.n_cpm+5.0))*a.wc_cpm_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_cpm/(r.n_cpm+5.0))*COALESCE(r.opp_cpm_aad,a.wc_cpm_aad)+(5.0/(r.n_cpm+5.0))*a.wc_cpm_aad) END
        END AS z_cpm,

        -- z_gpm
        CASE WHEN r.raw_gpm IS NULL THEN NULL ELSE
            (r.raw_gpm - (r.n_gpm/(r.n_gpm+5.0))*COALESCE(r.opp_gpm_mean,m.wc_gpm_mean) - (5.0/(r.n_gpm+5.0))*m.wc_gpm_mean) /
            CASE WHEN ((r.n_gpm/(r.n_gpm+5.0))*COALESCE(r.opp_gpm_aad,a.wc_gpm_aad)+(5.0/(r.n_gpm+5.0))*a.wc_gpm_aad) IS NULL THEN NULL
                 WHEN ((r.n_gpm/(r.n_gpm+5.0))*COALESCE(r.opp_gpm_aad,a.wc_gpm_aad)+(5.0/(r.n_gpm+5.0))*a.wc_gpm_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_gpm/(r.n_gpm+5.0))*COALESCE(r.opp_gpm_aad,a.wc_gpm_aad)+(5.0/(r.n_gpm+5.0))*a.wc_gpm_aad) END
        END AS z_gpm,

        -- z_lacc
        CASE WHEN r.raw_lacc IS NULL THEN NULL ELSE
            (r.raw_lacc - (r.n_lacc/(r.n_lacc+5.0))*COALESCE(r.opp_lacc_mean,m.wc_lacc_mean) - (5.0/(r.n_lacc+5.0))*m.wc_lacc_mean) /
            CASE WHEN ((r.n_lacc/(r.n_lacc+5.0))*COALESCE(r.opp_lacc_aad,a.wc_lacc_aad)+(5.0/(r.n_lacc+5.0))*a.wc_lacc_aad) IS NULL THEN NULL
                 WHEN ((r.n_lacc/(r.n_lacc+5.0))*COALESCE(r.opp_lacc_aad,a.wc_lacc_aad)+(5.0/(r.n_lacc+5.0))*a.wc_lacc_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_lacc/(r.n_lacc+5.0))*COALESCE(r.opp_lacc_aad,a.wc_lacc_aad)+(5.0/(r.n_lacc+5.0))*a.wc_lacc_aad) END
        END AS z_lacc,

        -- z_cacc
        CASE WHEN r.raw_cacc IS NULL THEN NULL ELSE
            (r.raw_cacc - (r.n_cacc/(r.n_cacc+5.0))*COALESCE(r.opp_cacc_mean,m.wc_cacc_mean) - (5.0/(r.n_cacc+5.0))*m.wc_cacc_mean) /
            CASE WHEN ((r.n_cacc/(r.n_cacc+5.0))*COALESCE(r.opp_cacc_aad,a.wc_cacc_aad)+(5.0/(r.n_cacc+5.0))*a.wc_cacc_aad) IS NULL THEN NULL
                 WHEN ((r.n_cacc/(r.n_cacc+5.0))*COALESCE(r.opp_cacc_aad,a.wc_cacc_aad)+(5.0/(r.n_cacc+5.0))*a.wc_cacc_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_cacc/(r.n_cacc+5.0))*COALESCE(r.opp_cacc_aad,a.wc_cacc_aad)+(5.0/(r.n_cacc+5.0))*a.wc_cacc_aad) END
        END AS z_cacc,

        -- z_gacc
        CASE WHEN r.raw_gacc IS NULL THEN NULL ELSE
            (r.raw_gacc - (r.n_gacc/(r.n_gacc+5.0))*COALESCE(r.opp_gacc_mean,m.wc_gacc_mean) - (5.0/(r.n_gacc+5.0))*m.wc_gacc_mean) /
            CASE WHEN ((r.n_gacc/(r.n_gacc+5.0))*COALESCE(r.opp_gacc_aad,a.wc_gacc_aad)+(5.0/(r.n_gacc+5.0))*a.wc_gacc_aad) IS NULL THEN NULL
                 WHEN ((r.n_gacc/(r.n_gacc+5.0))*COALESCE(r.opp_gacc_aad,a.wc_gacc_aad)+(5.0/(r.n_gacc+5.0))*a.wc_gacc_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_gacc/(r.n_gacc+5.0))*COALESCE(r.opp_gacc_aad,a.wc_gacc_aad)+(5.0/(r.n_gacc+5.0))*a.wc_gacc_aad) END
        END AS z_gacc,

        -- z_sacc
        CASE WHEN r.raw_sacc IS NULL THEN NULL ELSE
            (r.raw_sacc - (r.n_sacc/(r.n_sacc+5.0))*COALESCE(r.opp_sacc_mean,m.wc_sacc_mean) - (5.0/(r.n_sacc+5.0))*m.wc_sacc_mean) /
            CASE WHEN ((r.n_sacc/(r.n_sacc+5.0))*COALESCE(r.opp_sacc_aad,a.wc_sacc_aad)+(5.0/(r.n_sacc+5.0))*a.wc_sacc_aad) IS NULL THEN NULL
                 WHEN ((r.n_sacc/(r.n_sacc+5.0))*COALESCE(r.opp_sacc_aad,a.wc_sacc_aad)+(5.0/(r.n_sacc+5.0))*a.wc_sacc_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_sacc/(r.n_sacc+5.0))*COALESCE(r.opp_sacc_aad,a.wc_sacc_aad)+(5.0/(r.n_sacc+5.0))*a.wc_sacc_aad) END
        END AS z_sacc,

        -- z_tacc
        CASE WHEN r.raw_tacc IS NULL THEN NULL ELSE
            (r.raw_tacc - (r.n_tacc/(r.n_tacc+5.0))*COALESCE(r.opp_tacc_mean,m.wc_tacc_mean) - (5.0/(r.n_tacc+5.0))*m.wc_tacc_mean) /
            CASE WHEN ((r.n_tacc/(r.n_tacc+5.0))*COALESCE(r.opp_tacc_aad,a.wc_tacc_aad)+(5.0/(r.n_tacc+5.0))*a.wc_tacc_aad) IS NULL THEN NULL
                 WHEN ((r.n_tacc/(r.n_tacc+5.0))*COALESCE(r.opp_tacc_aad,a.wc_tacc_aad)+(5.0/(r.n_tacc+5.0))*a.wc_tacc_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_tacc/(r.n_tacc+5.0))*COALESCE(r.opp_tacc_aad,a.wc_tacc_aad)+(5.0/(r.n_tacc+5.0))*a.wc_tacc_aad) END
        END AS z_tacc,

        -- z_tdacc2
        CASE WHEN r.raw_tdacc2 IS NULL THEN NULL ELSE
            (r.raw_tdacc2 - (r.n_tdacc2/(r.n_tdacc2+5.0))*COALESCE(r.opp_tdacc2_mean,m.wc_tdacc2_mean) - (5.0/(r.n_tdacc2+5.0))*m.wc_tdacc2_mean) /
            CASE WHEN ((r.n_tdacc2/(r.n_tdacc2+5.0))*COALESCE(r.opp_tdacc2_aad,a.wc_tdacc2_aad)+(5.0/(r.n_tdacc2+5.0))*a.wc_tdacc2_aad) IS NULL THEN NULL
                 WHEN ((r.n_tdacc2/(r.n_tdacc2+5.0))*COALESCE(r.opp_tdacc2_aad,a.wc_tdacc2_aad)+(5.0/(r.n_tdacc2+5.0))*a.wc_tdacc2_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_tdacc2/(r.n_tdacc2+5.0))*COALESCE(r.opp_tdacc2_aad,a.wc_tdacc2_aad)+(5.0/(r.n_tdacc2+5.0))*a.wc_tdacc2_aad) END
        END AS z_tdacc2,

        -- z_hdef
        CASE WHEN r.raw_hdef IS NULL THEN NULL ELSE
            (r.raw_hdef - (r.n_hdef/(r.n_hdef+5.0))*COALESCE(r.opp_hdef_mean,m.wc_hdef_mean) - (5.0/(r.n_hdef+5.0))*m.wc_hdef_mean) /
            CASE WHEN ((r.n_hdef/(r.n_hdef+5.0))*COALESCE(r.opp_hdef_aad,a.wc_hdef_aad)+(5.0/(r.n_hdef+5.0))*a.wc_hdef_aad) IS NULL THEN NULL
                 WHEN ((r.n_hdef/(r.n_hdef+5.0))*COALESCE(r.opp_hdef_aad,a.wc_hdef_aad)+(5.0/(r.n_hdef+5.0))*a.wc_hdef_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_hdef/(r.n_hdef+5.0))*COALESCE(r.opp_hdef_aad,a.wc_hdef_aad)+(5.0/(r.n_hdef+5.0))*a.wc_hdef_aad) END
        END AS z_hdef,

        -- z_bdef
        CASE WHEN r.raw_bdef IS NULL THEN NULL ELSE
            (r.raw_bdef - (r.n_bdef/(r.n_bdef+5.0))*COALESCE(r.opp_bdef_mean,m.wc_bdef_mean) - (5.0/(r.n_bdef+5.0))*m.wc_bdef_mean) /
            CASE WHEN ((r.n_bdef/(r.n_bdef+5.0))*COALESCE(r.opp_bdef_aad,a.wc_bdef_aad)+(5.0/(r.n_bdef+5.0))*a.wc_bdef_aad) IS NULL THEN NULL
                 WHEN ((r.n_bdef/(r.n_bdef+5.0))*COALESCE(r.opp_bdef_aad,a.wc_bdef_aad)+(5.0/(r.n_bdef+5.0))*a.wc_bdef_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_bdef/(r.n_bdef+5.0))*COALESCE(r.opp_bdef_aad,a.wc_bdef_aad)+(5.0/(r.n_bdef+5.0))*a.wc_bdef_aad) END
        END AS z_bdef,

        -- z_ldef
        CASE WHEN r.raw_ldef IS NULL THEN NULL ELSE
            (r.raw_ldef - (r.n_ldef/(r.n_ldef+5.0))*COALESCE(r.opp_ldef_mean,m.wc_ldef_mean) - (5.0/(r.n_ldef+5.0))*m.wc_ldef_mean) /
            CASE WHEN ((r.n_ldef/(r.n_ldef+5.0))*COALESCE(r.opp_ldef_aad,a.wc_ldef_aad)+(5.0/(r.n_ldef+5.0))*a.wc_ldef_aad) IS NULL THEN NULL
                 WHEN ((r.n_ldef/(r.n_ldef+5.0))*COALESCE(r.opp_ldef_aad,a.wc_ldef_aad)+(5.0/(r.n_ldef+5.0))*a.wc_ldef_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_ldef/(r.n_ldef+5.0))*COALESCE(r.opp_ldef_aad,a.wc_ldef_aad)+(5.0/(r.n_ldef+5.0))*a.wc_ldef_aad) END
        END AS z_ldef,

        -- z_cdef
        CASE WHEN r.raw_cdef IS NULL THEN NULL ELSE
            (r.raw_cdef - (r.n_cdef/(r.n_cdef+5.0))*COALESCE(r.opp_cdef_mean,m.wc_cdef_mean) - (5.0/(r.n_cdef+5.0))*m.wc_cdef_mean) /
            CASE WHEN ((r.n_cdef/(r.n_cdef+5.0))*COALESCE(r.opp_cdef_aad,a.wc_cdef_aad)+(5.0/(r.n_cdef+5.0))*a.wc_cdef_aad) IS NULL THEN NULL
                 WHEN ((r.n_cdef/(r.n_cdef+5.0))*COALESCE(r.opp_cdef_aad,a.wc_cdef_aad)+(5.0/(r.n_cdef+5.0))*a.wc_cdef_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_cdef/(r.n_cdef+5.0))*COALESCE(r.opp_cdef_aad,a.wc_cdef_aad)+(5.0/(r.n_cdef+5.0))*a.wc_cdef_aad) END
        END AS z_cdef,

        -- z_sdef
        CASE WHEN r.raw_sdef IS NULL THEN NULL ELSE
            (r.raw_sdef - (r.n_sdef/(r.n_sdef+5.0))*COALESCE(r.opp_sdef_mean,m.wc_sdef_mean) - (5.0/(r.n_sdef+5.0))*m.wc_sdef_mean) /
            CASE WHEN ((r.n_sdef/(r.n_sdef+5.0))*COALESCE(r.opp_sdef_aad,a.wc_sdef_aad)+(5.0/(r.n_sdef+5.0))*a.wc_sdef_aad) IS NULL THEN NULL
                 WHEN ((r.n_sdef/(r.n_sdef+5.0))*COALESCE(r.opp_sdef_aad,a.wc_sdef_aad)+(5.0/(r.n_sdef+5.0))*a.wc_sdef_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_sdef/(r.n_sdef+5.0))*COALESCE(r.opp_sdef_aad,a.wc_sdef_aad)+(5.0/(r.n_sdef+5.0))*a.wc_sdef_aad) END
        END AS z_sdef,

        -- z_tdef
        CASE WHEN r.raw_tdef IS NULL THEN NULL ELSE
            (r.raw_tdef - (r.n_tdef/(r.n_tdef+5.0))*COALESCE(r.opp_tdef_mean,m.wc_tdef_mean) - (5.0/(r.n_tdef+5.0))*m.wc_tdef_mean) /
            CASE WHEN ((r.n_tdef/(r.n_tdef+5.0))*COALESCE(r.opp_tdef_aad,a.wc_tdef_aad)+(5.0/(r.n_tdef+5.0))*a.wc_tdef_aad) IS NULL THEN NULL
                 WHEN ((r.n_tdef/(r.n_tdef+5.0))*COALESCE(r.opp_tdef_aad,a.wc_tdef_aad)+(5.0/(r.n_tdef+5.0))*a.wc_tdef_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_tdef/(r.n_tdef+5.0))*COALESCE(r.opp_tdef_aad,a.wc_tdef_aad)+(5.0/(r.n_tdef+5.0))*a.wc_tdef_aad) END
        END AS z_tdef,

        -- z_tsdef
        CASE WHEN r.raw_tsdef IS NULL THEN NULL ELSE
            (r.raw_tsdef - (r.n_tsdef/(r.n_tsdef+5.0))*COALESCE(r.opp_tsdef_mean,m.wc_tsdef_mean) - (5.0/(r.n_tsdef+5.0))*m.wc_tsdef_mean) /
            CASE WHEN ((r.n_tsdef/(r.n_tsdef+5.0))*COALESCE(r.opp_tsdef_aad,a.wc_tsdef_aad)+(5.0/(r.n_tsdef+5.0))*a.wc_tsdef_aad) IS NULL THEN NULL
                 WHEN ((r.n_tsdef/(r.n_tsdef+5.0))*COALESCE(r.opp_tsdef_aad,a.wc_tsdef_aad)+(5.0/(r.n_tsdef+5.0))*a.wc_tsdef_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_tsdef/(r.n_tsdef+5.0))*COALESCE(r.opp_tsdef_aad,a.wc_tsdef_aad)+(5.0/(r.n_tsdef+5.0))*a.wc_tsdef_aad) END
        END AS z_tsdef,

        -- z_bratio
        CASE WHEN r.raw_bratio IS NULL THEN NULL ELSE
            (r.raw_bratio - (r.n_bratio/(r.n_bratio+5.0))*COALESCE(r.opp_bratio_mean,m.wc_bratio_mean) - (5.0/(r.n_bratio+5.0))*m.wc_bratio_mean) /
            CASE WHEN ((r.n_bratio/(r.n_bratio+5.0))*COALESCE(r.opp_bratio_aad,a.wc_bratio_aad)+(5.0/(r.n_bratio+5.0))*a.wc_bratio_aad) IS NULL THEN NULL
                 WHEN ((r.n_bratio/(r.n_bratio+5.0))*COALESCE(r.opp_bratio_aad,a.wc_bratio_aad)+(5.0/(r.n_bratio+5.0))*a.wc_bratio_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_bratio/(r.n_bratio+5.0))*COALESCE(r.opp_bratio_aad,a.wc_bratio_aad)+(5.0/(r.n_bratio+5.0))*a.wc_bratio_aad) END
        END AS z_bratio,

        -- z_lratio
        CASE WHEN r.raw_lratio IS NULL THEN NULL ELSE
            (r.raw_lratio - (r.n_lratio/(r.n_lratio+5.0))*COALESCE(r.opp_lratio_mean,m.wc_lratio_mean) - (5.0/(r.n_lratio+5.0))*m.wc_lratio_mean) /
            CASE WHEN ((r.n_lratio/(r.n_lratio+5.0))*COALESCE(r.opp_lratio_aad,a.wc_lratio_aad)+(5.0/(r.n_lratio+5.0))*a.wc_lratio_aad) IS NULL THEN NULL
                 WHEN ((r.n_lratio/(r.n_lratio+5.0))*COALESCE(r.opp_lratio_aad,a.wc_lratio_aad)+(5.0/(r.n_lratio+5.0))*a.wc_lratio_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_lratio/(r.n_lratio+5.0))*COALESCE(r.opp_lratio_aad,a.wc_lratio_aad)+(5.0/(r.n_lratio+5.0))*a.wc_lratio_aad) END
        END AS z_lratio,

        -- z_cratio
        CASE WHEN r.raw_cratio IS NULL THEN NULL ELSE
            (r.raw_cratio - (r.n_cratio/(r.n_cratio+5.0))*COALESCE(r.opp_cratio_mean,m.wc_cratio_mean) - (5.0/(r.n_cratio+5.0))*m.wc_cratio_mean) /
            CASE WHEN ((r.n_cratio/(r.n_cratio+5.0))*COALESCE(r.opp_cratio_aad,a.wc_cratio_aad)+(5.0/(r.n_cratio+5.0))*a.wc_cratio_aad) IS NULL THEN NULL
                 WHEN ((r.n_cratio/(r.n_cratio+5.0))*COALESCE(r.opp_cratio_aad,a.wc_cratio_aad)+(5.0/(r.n_cratio+5.0))*a.wc_cratio_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_cratio/(r.n_cratio+5.0))*COALESCE(r.opp_cratio_aad,a.wc_cratio_aad)+(5.0/(r.n_cratio+5.0))*a.wc_cratio_aad) END
        END AS z_cratio,

        -- z_gratio
        CASE WHEN r.raw_gratio IS NULL THEN NULL ELSE
            (r.raw_gratio - (r.n_gratio/(r.n_gratio+5.0))*COALESCE(r.opp_gratio_mean,m.wc_gratio_mean) - (5.0/(r.n_gratio+5.0))*m.wc_gratio_mean) /
            CASE WHEN ((r.n_gratio/(r.n_gratio+5.0))*COALESCE(r.opp_gratio_aad,a.wc_gratio_aad)+(5.0/(r.n_gratio+5.0))*a.wc_gratio_aad) IS NULL THEN NULL
                 WHEN ((r.n_gratio/(r.n_gratio+5.0))*COALESCE(r.opp_gratio_aad,a.wc_gratio_aad)+(5.0/(r.n_gratio+5.0))*a.wc_gratio_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_gratio/(r.n_gratio+5.0))*COALESCE(r.opp_gratio_aad,a.wc_gratio_aad)+(5.0/(r.n_gratio+5.0))*a.wc_gratio_aad) END
        END AS z_gratio,

        -- z_dratio
        CASE WHEN r.raw_dratio IS NULL THEN NULL ELSE
            (r.raw_dratio - (r.n_dratio/(r.n_dratio+5.0))*COALESCE(r.opp_dratio_mean,m.wc_dratio_mean) - (5.0/(r.n_dratio+5.0))*m.wc_dratio_mean) /
            CASE WHEN ((r.n_dratio/(r.n_dratio+5.0))*COALESCE(r.opp_dratio_aad,a.wc_dratio_aad)+(5.0/(r.n_dratio+5.0))*a.wc_dratio_aad) IS NULL THEN NULL
                 WHEN ((r.n_dratio/(r.n_dratio+5.0))*COALESCE(r.opp_dratio_aad,a.wc_dratio_aad)+(5.0/(r.n_dratio+5.0))*a.wc_dratio_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_dratio/(r.n_dratio+5.0))*COALESCE(r.opp_dratio_aad,a.wc_dratio_aad)+(5.0/(r.n_dratio+5.0))*a.wc_dratio_aad) END
        END AS z_dratio

    FROM rolling_aad r
    JOIN wc_mean m ON r.weightindex = m.weightindex
    JOIN wc_aad  a ON r.weightindex = a.weightindex
),

-- 8. Pre-compute shrink factor and dynamic clip bound per row
z_params AS (
    SELECT *,
        CAST(rn AS REAL) / (rn + 10.0)       AS _sh,
        14.0 * CAST(rn AS REAL) / (rn + 7.0) AS _cl
    FROM z_scores
),

-- 9. Apply shrinkage then dynamic clip
z_shrunk AS (
    SELECT DATE, jevent, jbout, jfighter, weightindex, jd, rn,
        CASE WHEN z_hpm     IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_hpm     * _sh)) END AS z_hpm,
        CASE WHEN z_bpm     IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_bpm     * _sh)) END AS z_bpm,
        CASE WHEN z_lpm     IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_lpm     * _sh)) END AS z_lpm,
        CASE WHEN z_dpm     IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_dpm     * _sh)) END AS z_dpm,
        CASE WHEN z_cpm     IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_cpm     * _sh)) END AS z_cpm,
        CASE WHEN z_gpm     IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_gpm     * _sh)) END AS z_gpm,
        CASE WHEN z_lacc    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_lacc    * _sh)) END AS z_lacc,
        CASE WHEN z_cacc    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_cacc    * _sh)) END AS z_cacc,
        CASE WHEN z_gacc    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_gacc    * _sh)) END AS z_gacc,
        CASE WHEN z_sacc    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_sacc    * _sh)) END AS z_sacc,
        CASE WHEN z_tacc    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_tacc    * _sh)) END AS z_tacc,
        CASE WHEN z_tdacc2  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_tdacc2  * _sh)) END AS z_tdacc2,
        CASE WHEN z_hdef    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_hdef    * _sh)) END AS z_hdef,
        CASE WHEN z_bdef    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_bdef    * _sh)) END AS z_bdef,
        CASE WHEN z_ldef    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_ldef    * _sh)) END AS z_ldef,
        CASE WHEN z_cdef    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_cdef    * _sh)) END AS z_cdef,
        CASE WHEN z_sdef    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_sdef    * _sh)) END AS z_sdef,
        CASE WHEN z_tdef    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_tdef    * _sh)) END AS z_tdef,
        CASE WHEN z_tsdef   IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_tsdef   * _sh)) END AS z_tsdef,
        CASE WHEN z_bratio  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_bratio  * _sh)) END AS z_bratio,
        CASE WHEN z_lratio  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_lratio  * _sh)) END AS z_lratio,
        CASE WHEN z_cratio  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_cratio  * _sh)) END AS z_cratio,
        CASE WHEN z_gratio  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_gratio  * _sh)) END AS z_gratio,
        CASE WHEN z_dratio  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_dratio  * _sh)) END AS z_dratio
    FROM z_params
),

-- 10. Seed: first fight per fighter
ema_seed AS (
    SELECT *,
        CASE WHEN z_hpm    IS NULL THEN 0.0 ELSE z_hpm    END AS hpm_n,
        CASE WHEN z_hpm    IS NULL THEN 0.0 ELSE 1.0      END AS hpm_d,
        CASE WHEN z_bpm    IS NULL THEN 0.0 ELSE z_bpm    END AS bpm_n,
        CASE WHEN z_bpm    IS NULL THEN 0.0 ELSE 1.0      END AS bpm_d,
        CASE WHEN z_lpm    IS NULL THEN 0.0 ELSE z_lpm    END AS lpm_n,
        CASE WHEN z_lpm    IS NULL THEN 0.0 ELSE 1.0      END AS lpm_d,
        CASE WHEN z_dpm    IS NULL THEN 0.0 ELSE z_dpm    END AS dpm_n,
        CASE WHEN z_dpm    IS NULL THEN 0.0 ELSE 1.0      END AS dpm_d,
        CASE WHEN z_cpm    IS NULL THEN 0.0 ELSE z_cpm    END AS cpm_n,
        CASE WHEN z_cpm    IS NULL THEN 0.0 ELSE 1.0      END AS cpm_d,
        CASE WHEN z_gpm    IS NULL THEN 0.0 ELSE z_gpm    END AS gpm_n,
        CASE WHEN z_gpm    IS NULL THEN 0.0 ELSE 1.0      END AS gpm_d,
        CASE WHEN z_lacc   IS NULL THEN 0.0 ELSE z_lacc   END AS lac_n,
        CASE WHEN z_lacc   IS NULL THEN 0.0 ELSE 1.0      END AS lac_d,
        CASE WHEN z_cacc   IS NULL THEN 0.0 ELSE z_cacc   END AS cac_n,
        CASE WHEN z_cacc   IS NULL THEN 0.0 ELSE 1.0      END AS cac_d,
        CASE WHEN z_gacc   IS NULL THEN 0.0 ELSE z_gacc   END AS gac_n,
        CASE WHEN z_gacc   IS NULL THEN 0.0 ELSE 1.0      END AS gac_d,
        CASE WHEN z_sacc   IS NULL THEN 0.0 ELSE z_sacc   END AS sac_n,
        CASE WHEN z_sacc   IS NULL THEN 0.0 ELSE 1.0      END AS sac_d,
        CASE WHEN z_tacc   IS NULL THEN 0.0 ELSE z_tacc   END AS tac_n,
        CASE WHEN z_tacc   IS NULL THEN 0.0 ELSE 1.0      END AS tac_d,
        CASE WHEN z_tdacc2 IS NULL THEN 0.0 ELSE z_tdacc2 END AS tda_n,
        CASE WHEN z_tdacc2 IS NULL THEN 0.0 ELSE 1.0      END AS tda_d,
        CASE WHEN z_hdef   IS NULL THEN 0.0 ELSE z_hdef   END AS hdf_n,
        CASE WHEN z_hdef   IS NULL THEN 0.0 ELSE 1.0      END AS hdf_d,
        CASE WHEN z_bdef   IS NULL THEN 0.0 ELSE z_bdef   END AS bdf_n,
        CASE WHEN z_bdef   IS NULL THEN 0.0 ELSE 1.0      END AS bdf_d,
        CASE WHEN z_ldef   IS NULL THEN 0.0 ELSE z_ldef   END AS ldf_n,
        CASE WHEN z_ldef   IS NULL THEN 0.0 ELSE 1.0      END AS ldf_d,
        CASE WHEN z_cdef   IS NULL THEN 0.0 ELSE z_cdef   END AS cdf_n,
        CASE WHEN z_cdef   IS NULL THEN 0.0 ELSE 1.0      END AS cdf_d,
        CASE WHEN z_sdef   IS NULL THEN 0.0 ELSE z_sdef   END AS sdf_n,
        CASE WHEN z_sdef   IS NULL THEN 0.0 ELSE 1.0      END AS sdf_d,
        CASE WHEN z_tdef   IS NULL THEN 0.0 ELSE z_tdef   END AS tdf_n,
        CASE WHEN z_tdef   IS NULL THEN 0.0 ELSE 1.0      END AS tdf_d,
        CASE WHEN z_tsdef  IS NULL THEN 0.0 ELSE z_tsdef  END AS tsf_n,
        CASE WHEN z_tsdef  IS NULL THEN 0.0 ELSE 1.0      END AS tsf_d,
        CASE WHEN z_bratio IS NULL THEN 0.0 ELSE z_bratio END AS brt_n,
        CASE WHEN z_bratio IS NULL THEN 0.0 ELSE 1.0      END AS brt_d,
        CASE WHEN z_lratio IS NULL THEN 0.0 ELSE z_lratio END AS lrt_n,
        CASE WHEN z_lratio IS NULL THEN 0.0 ELSE 1.0      END AS lrt_d,
        CASE WHEN z_cratio IS NULL THEN 0.0 ELSE z_cratio END AS crt_n,
        CASE WHEN z_cratio IS NULL THEN 0.0 ELSE 1.0      END AS crt_d,
        CASE WHEN z_gratio IS NULL THEN 0.0 ELSE z_gratio END AS grt_n,
        CASE WHEN z_gratio IS NULL THEN 0.0 ELSE 1.0      END AS grt_d,
        CASE WHEN z_dratio IS NULL THEN 0.0 ELSE z_dratio END AS drt_n,
        CASE WHEN z_dratio IS NULL THEN 0.0 ELSE 1.0      END AS drt_d
    FROM z_shrunk WHERE rn = 1
),

-- 11. Recursive EMA-decay of per-fight z-scores (lambda=0.13 per year, ~5yr half-life)
ema_z AS (
    SELECT * FROM ema_seed
    UNION ALL
    SELECT
        s.DATE, s.jevent, s.jbout, s.jfighter, s.weightindex, s.jd, s.rn,
        s.z_hpm, s.z_bpm, s.z_lpm, s.z_dpm, s.z_cpm, s.z_gpm,
        s.z_lacc, s.z_cacc, s.z_gacc, s.z_sacc, s.z_tacc, s.z_tdacc2,
        s.z_hdef, s.z_bdef, s.z_ldef, s.z_cdef, s.z_sdef, s.z_tdef, s.z_tsdef,
        s.z_bratio, s.z_lratio, s.z_cratio, s.z_gratio, s.z_dratio,
        -- hpm
        (e.hpm_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_hpm IS NULL THEN 0.0 ELSE s.z_hpm END AS hpm_n,
        (e.hpm_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_hpm IS NULL THEN 0.0 ELSE 1.0    END AS hpm_d,
        -- bpm
        (e.bpm_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_bpm IS NULL THEN 0.0 ELSE s.z_bpm END AS bpm_n,
        (e.bpm_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_bpm IS NULL THEN 0.0 ELSE 1.0    END AS bpm_d,
        -- lpm
        (e.lpm_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_lpm IS NULL THEN 0.0 ELSE s.z_lpm END AS lpm_n,
        (e.lpm_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_lpm IS NULL THEN 0.0 ELSE 1.0    END AS lpm_d,
        -- dpm
        (e.dpm_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_dpm IS NULL THEN 0.0 ELSE s.z_dpm END AS dpm_n,
        (e.dpm_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_dpm IS NULL THEN 0.0 ELSE 1.0    END AS dpm_d,
        -- cpm
        (e.cpm_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_cpm IS NULL THEN 0.0 ELSE s.z_cpm END AS cpm_n,
        (e.cpm_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_cpm IS NULL THEN 0.0 ELSE 1.0    END AS cpm_d,
        -- gpm
        (e.gpm_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_gpm IS NULL THEN 0.0 ELSE s.z_gpm END AS gpm_n,
        (e.gpm_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_gpm IS NULL THEN 0.0 ELSE 1.0    END AS gpm_d,
        -- lacc
        (e.lac_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_lacc IS NULL THEN 0.0 ELSE s.z_lacc END AS lac_n,
        (e.lac_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_lacc IS NULL THEN 0.0 ELSE 1.0     END AS lac_d,
        -- cacc
        (e.cac_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_cacc IS NULL THEN 0.0 ELSE s.z_cacc END AS cac_n,
        (e.cac_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_cacc IS NULL THEN 0.0 ELSE 1.0     END AS cac_d,
        -- gacc
        (e.gac_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_gacc IS NULL THEN 0.0 ELSE s.z_gacc END AS gac_n,
        (e.gac_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_gacc IS NULL THEN 0.0 ELSE 1.0     END AS gac_d,
        -- sacc
        (e.sac_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_sacc IS NULL THEN 0.0 ELSE s.z_sacc END AS sac_n,
        (e.sac_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_sacc IS NULL THEN 0.0 ELSE 1.0     END AS sac_d,
        -- tacc
        (e.tac_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tacc IS NULL THEN 0.0 ELSE s.z_tacc END AS tac_n,
        (e.tac_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tacc IS NULL THEN 0.0 ELSE 1.0     END AS tac_d,
        -- tdacc2
        (e.tda_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tdacc2 IS NULL THEN 0.0 ELSE s.z_tdacc2 END AS tda_n,
        (e.tda_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tdacc2 IS NULL THEN 0.0 ELSE 1.0       END AS tda_d,
        -- hdef
        (e.hdf_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_hdef IS NULL THEN 0.0 ELSE s.z_hdef END AS hdf_n,
        (e.hdf_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_hdef IS NULL THEN 0.0 ELSE 1.0     END AS hdf_d,
        -- bdef
        (e.bdf_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_bdef IS NULL THEN 0.0 ELSE s.z_bdef END AS bdf_n,
        (e.bdf_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_bdef IS NULL THEN 0.0 ELSE 1.0     END AS bdf_d,
        -- ldef
        (e.ldf_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_ldef IS NULL THEN 0.0 ELSE s.z_ldef END AS ldf_n,
        (e.ldf_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_ldef IS NULL THEN 0.0 ELSE 1.0     END AS ldf_d,
        -- cdef
        (e.cdf_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_cdef IS NULL THEN 0.0 ELSE s.z_cdef END AS cdf_n,
        (e.cdf_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_cdef IS NULL THEN 0.0 ELSE 1.0     END AS cdf_d,
        -- sdef
        (e.sdf_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_sdef IS NULL THEN 0.0 ELSE s.z_sdef END AS sdf_n,
        (e.sdf_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_sdef IS NULL THEN 0.0 ELSE 1.0     END AS sdf_d,
        -- tdef
        (e.tdf_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tdef IS NULL THEN 0.0 ELSE s.z_tdef END AS tdf_n,
        (e.tdf_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tdef IS NULL THEN 0.0 ELSE 1.0     END AS tdf_d,
        -- tsdef
        (e.tsf_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tsdef IS NULL THEN 0.0 ELSE s.z_tsdef END AS tsf_n,
        (e.tsf_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tsdef IS NULL THEN 0.0 ELSE 1.0      END AS tsf_d,
        -- bratio
        (e.brt_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_bratio IS NULL THEN 0.0 ELSE s.z_bratio END AS brt_n,
        (e.brt_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_bratio IS NULL THEN 0.0 ELSE 1.0       END AS brt_d,
        -- lratio
        (e.lrt_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_lratio IS NULL THEN 0.0 ELSE s.z_lratio END AS lrt_n,
        (e.lrt_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_lratio IS NULL THEN 0.0 ELSE 1.0       END AS lrt_d,
        -- cratio
        (e.crt_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_cratio IS NULL THEN 0.0 ELSE s.z_cratio END AS crt_n,
        (e.crt_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_cratio IS NULL THEN 0.0 ELSE 1.0       END AS crt_d,
        -- gratio
        (e.grt_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_gratio IS NULL THEN 0.0 ELSE s.z_gratio END AS grt_n,
        (e.grt_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_gratio IS NULL THEN 0.0 ELSE 1.0       END AS grt_d,
        -- dratio
        (e.drt_n * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_dratio IS NULL THEN 0.0 ELSE s.z_dratio END AS drt_n,
        (e.drt_d * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_dratio IS NULL THEN 0.0 ELSE 1.0       END AS drt_d
    FROM z_shrunk s
    JOIN ema_z e ON e.jfighter = s.jfighter AND e.rn = s.rn - 1
),

-- 12. LAG all EMA-z accumulators by 1 fight (no current-fight leakage)
lagged_ema_z AS (
    SELECT
        DATE, jevent, jbout, jfighter, weightindex,
        LAG(hpm_n, 1, 0.0) OVER w AS hpm_n,  LAG(hpm_d, 1, 0.0) OVER w AS hpm_d,
        LAG(bpm_n, 1, 0.0) OVER w AS bpm_n,  LAG(bpm_d, 1, 0.0) OVER w AS bpm_d,
        LAG(lpm_n, 1, 0.0) OVER w AS lpm_n,  LAG(lpm_d, 1, 0.0) OVER w AS lpm_d,
        LAG(dpm_n, 1, 0.0) OVER w AS dpm_n,  LAG(dpm_d, 1, 0.0) OVER w AS dpm_d,
        LAG(cpm_n, 1, 0.0) OVER w AS cpm_n,  LAG(cpm_d, 1, 0.0) OVER w AS cpm_d,
        LAG(gpm_n, 1, 0.0) OVER w AS gpm_n,  LAG(gpm_d, 1, 0.0) OVER w AS gpm_d,
        LAG(lac_n, 1, 0.0) OVER w AS lac_n,  LAG(lac_d, 1, 0.0) OVER w AS lac_d,
        LAG(cac_n, 1, 0.0) OVER w AS cac_n,  LAG(cac_d, 1, 0.0) OVER w AS cac_d,
        LAG(gac_n, 1, 0.0) OVER w AS gac_n,  LAG(gac_d, 1, 0.0) OVER w AS gac_d,
        LAG(sac_n, 1, 0.0) OVER w AS sac_n,  LAG(sac_d, 1, 0.0) OVER w AS sac_d,
        LAG(tac_n, 1, 0.0) OVER w AS tac_n,  LAG(tac_d, 1, 0.0) OVER w AS tac_d,
        LAG(tda_n, 1, 0.0) OVER w AS tda_n,  LAG(tda_d, 1, 0.0) OVER w AS tda_d,
        LAG(hdf_n, 1, 0.0) OVER w AS hdf_n,  LAG(hdf_d, 1, 0.0) OVER w AS hdf_d,
        LAG(bdf_n, 1, 0.0) OVER w AS bdf_n,  LAG(bdf_d, 1, 0.0) OVER w AS bdf_d,
        LAG(ldf_n, 1, 0.0) OVER w AS ldf_n,  LAG(ldf_d, 1, 0.0) OVER w AS ldf_d,
        LAG(cdf_n, 1, 0.0) OVER w AS cdf_n,  LAG(cdf_d, 1, 0.0) OVER w AS cdf_d,
        LAG(sdf_n, 1, 0.0) OVER w AS sdf_n,  LAG(sdf_d, 1, 0.0) OVER w AS sdf_d,
        LAG(tdf_n, 1, 0.0) OVER w AS tdf_n,  LAG(tdf_d, 1, 0.0) OVER w AS tdf_d,
        LAG(tsf_n, 1, 0.0) OVER w AS tsf_n,  LAG(tsf_d, 1, 0.0) OVER w AS tsf_d,
        LAG(brt_n, 1, 0.0) OVER w AS brt_n,  LAG(brt_d, 1, 0.0) OVER w AS brt_d,
        LAG(lrt_n, 1, 0.0) OVER w AS lrt_n,  LAG(lrt_d, 1, 0.0) OVER w AS lrt_d,
        LAG(crt_n, 1, 0.0) OVER w AS crt_n,  LAG(crt_d, 1, 0.0) OVER w AS crt_d,
        LAG(grt_n, 1, 0.0) OVER w AS grt_n,  LAG(grt_d, 1, 0.0) OVER w AS grt_d,
        LAG(drt_n, 1, 0.0) OVER w AS drt_n,  LAG(drt_d, 1, 0.0) OVER w AS drt_d
    FROM ema_z
    WINDOW w AS (PARTITION BY jfighter ORDER BY rn)
)

-- 13. Final output: decayed average z-scores
SELECT
    DATE, jevent, jbout, jfighter, weightindex,
    -- Group 1: Per-minute striking
    CASE WHEN hpm_d = 0 THEN NULL ELSE hpm_n / hpm_d END AS adjperf_head_pm_dec_avg,
    CASE WHEN bpm_d = 0 THEN NULL ELSE bpm_n / bpm_d END AS adjperf_body_pm_dec_avg,
    CASE WHEN lpm_d = 0 THEN NULL ELSE lpm_n / lpm_d END AS adjperf_leg_pm_dec_avg,
    CASE WHEN dpm_d = 0 THEN NULL ELSE dpm_n / dpm_d END AS adjperf_dist_pm_dec_avg,
    CASE WHEN cpm_d = 0 THEN NULL ELSE cpm_n / cpm_d END AS adjperf_clinch_pm_dec_avg,
    CASE WHEN gpm_d = 0 THEN NULL ELSE gpm_n / gpm_d END AS adjperf_ground_pm_dec_avg,
    -- Group 2: Accuracy percentages
    CASE WHEN lac_d = 0 THEN NULL ELSE lac_n / lac_d END AS adjperf_legacc_dec_avg,
    CASE WHEN cac_d = 0 THEN NULL ELSE cac_n / cac_d END AS adjperf_clinchacc_dec_avg,
    CASE WHEN gac_d = 0 THEN NULL ELSE gac_n / gac_d END AS adjperf_groundacc_dec_avg,
    CASE WHEN sac_d = 0 THEN NULL ELSE sac_n / sac_d END AS adjperf_sigstracc_dec_avg,
    CASE WHEN tac_d = 0 THEN NULL ELSE tac_n / tac_d END AS adjperf_totalstracc_dec_avg,
    CASE WHEN tda_d = 0 THEN NULL ELSE tda_n / tda_d END AS adjperf_tdacc_dec_avg,
    -- Group 3: Defense
    CASE WHEN hdf_d = 0 THEN NULL ELSE hdf_n / hdf_d END AS adjperf_headdef_dec_avg,
    CASE WHEN bdf_d = 0 THEN NULL ELSE bdf_n / bdf_d END AS adjperf_bodydef_dec_avg,
    CASE WHEN ldf_d = 0 THEN NULL ELSE ldf_n / ldf_d END AS adjperf_legdef_dec_avg,
    CASE WHEN cdf_d = 0 THEN NULL ELSE cdf_n / cdf_d END AS adjperf_clinchdef_dec_avg,
    CASE WHEN sdf_d = 0 THEN NULL ELSE sdf_n / sdf_d END AS adjperf_sigstrdef_dec_avg,
    CASE WHEN tdf_d = 0 THEN NULL ELSE tdf_n / tdf_d END AS adjperf_tddef_dec_avg,
    CASE WHEN tsf_d = 0 THEN NULL ELSE tsf_n / tsf_d END AS adjperf_totalstrdef_dec_avg,
    -- Group 4: Zone ratios
    CASE WHEN brt_d = 0 THEN NULL ELSE brt_n / brt_d END AS adjperf_bodyratio_dec_avg,
    CASE WHEN lrt_d = 0 THEN NULL ELSE lrt_n / lrt_d END AS adjperf_legratio_dec_avg,
    CASE WHEN crt_d = 0 THEN NULL ELSE crt_n / crt_d END AS adjperf_clinch_ratio_dec_avg,
    CASE WHEN grt_d = 0 THEN NULL ELSE grt_n / grt_d END AS adjperf_ground_ratio_dec_avg,
    CASE WHEN drt_d = 0 THEN NULL ELSE drt_n / drt_d END AS adjperf_dist_ratio_dec_avg
FROM lagged_ema_z;

CREATE INDEX IF NOT EXISTS idx_adjperf_new2_fast_bout
ON adjperf_new2_fast(jevent, jbout, jfighter);
