-- =========================================================
-- New Adjperf3 Features (Phase 6)
-- 9 adjperf z-score features
--
-- New features:
--   adjperf_win_dec_avg                  (win rate)
--   adjperf_td_per_sigstr_att_dec_avg    (TD att / sig str att)
--   adjperf_ctrl_ratio_dec_avg           (ctrl per min)
--   adjperf_ko_ratio_dec_avg             (KO rate)
--   adjperf_sub_att_ratio_dec_avg        (sub attempts)
--   adjperf_rev_per_ctrlopp_dec_avg      (NULL - no reversal data)
--   adjperf_sub_def_dec_avg              (sub defense)
--   adjperf_td_land_per_ctrl_dec_avg     (TD land / ctrl time)
--   adjperf_rev_ratio_dec_avg            (NULL - no reversal data)
--
-- Methodology: same as adjperf_new_fast
--   raw per-fight stat -> z-score (Bayesian K=5, PARTITION BY opp_jfighter)
--   -> EMA-decay (lambda=0.13, ~5yr half-life)
--   -> opp filter: opponent must have >= 2 prior UFC fights
--
-- Run AFTER: ... -> new_features3.sql
-- =========================================================

DROP TABLE IF EXISTS adjperf_new3_fast;

CREATE TABLE adjperf_new3_fast AS
WITH RECURSIVE

-- 1. Opponent prior fight count (>= 2 prior fights filter)
opp_history AS (
    SELECT jfighter, jevent, jbout,
           ROW_NUMBER() OVER (PARTITION BY jfighter ORDER BY DATE, jevent, jbout) - 1 AS n_prior
    FROM decayed_precomp_v2_fast
),

-- 2. Per-fight stats (qualifying fights only)
fight_data AS (
    SELECT p.DATE, p.jevent, p.jbout, p.jfighter, p.opp_jfighter, p.weightindex,
           julianday(p.DATE) AS jd,

           -- win (binary)
           p.win AS raw_win,

           -- TD att / sig str att
           CASE WHEN p.sigstratt = 0 OR p.sigstratt IS NULL THEN NULL
                ELSE 1.0 * p.tdatt / p.sigstratt END AS raw_tdpsa,

           -- ctrl per min (already computed)
           p.raw_ctrl_per_min AS raw_ctrl,

           -- KO (binary)
           p.ko AS raw_ko,

           -- sub attempts
           p.subatt AS raw_subr,

           -- reversal per ctrl opp (no data)
           NULL AS raw_revc,

           -- sub defense = 1 - subwd
           CASE WHEN p.subwd IS NULL THEN NULL
                ELSE 1.0 - p.subwd END AS raw_sdef,

           -- TD land / ctrl time
           CASE WHEN (p.raw_ctrl_per_min * p.fight_time_minutes) = 0
                  OR (p.raw_ctrl_per_min * p.fight_time_minutes) IS NULL
                  OR p.tdacc IS NULL THEN NULL
                ELSE 1.0 * p.tdacc / (p.raw_ctrl_per_min * p.fight_time_minutes) END AS raw_tdpc,

           -- reversal ratio (no data)
           NULL AS raw_revr

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
        AVG(raw_win)   AS wc_win_mean,
        AVG(raw_tdpsa) AS wc_tdpsa_mean,
        AVG(raw_ctrl)  AS wc_ctrl_mean,
        AVG(raw_ko)    AS wc_ko_mean,
        AVG(raw_subr)  AS wc_subr_mean,
        AVG(raw_revc)  AS wc_revc_mean,
        AVG(raw_sdef)  AS wc_sdef_mean,
        AVG(raw_tdpc)  AS wc_tdpc_mean,
        AVG(raw_revr)  AS wc_revr_mean
    FROM fight_data
    GROUP BY weightindex
),
wc_aad AS (
    SELECT d.weightindex,
        AVG(ABS(d.raw_win   - m.wc_win_mean))   AS wc_win_aad,
        AVG(ABS(d.raw_tdpsa - m.wc_tdpsa_mean)) AS wc_tdpsa_aad,
        AVG(ABS(d.raw_ctrl  - m.wc_ctrl_mean))  AS wc_ctrl_aad,
        AVG(ABS(d.raw_ko    - m.wc_ko_mean))    AS wc_ko_aad,
        AVG(ABS(d.raw_subr  - m.wc_subr_mean))  AS wc_subr_aad,
        AVG(ABS(d.raw_revc  - m.wc_revc_mean))  AS wc_revc_aad,
        AVG(ABS(d.raw_sdef  - m.wc_sdef_mean))  AS wc_sdef_aad,
        AVG(ABS(d.raw_tdpc  - m.wc_tdpc_mean))  AS wc_tdpc_aad,
        AVG(ABS(d.raw_revr  - m.wc_revr_mean))  AS wc_revr_aad
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
        AVG(raw_win)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_win_mean,
        COUNT(raw_win)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_win,
        AVG(raw_tdpsa)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tdpsa_mean,
        COUNT(raw_tdpsa)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_tdpsa,
        AVG(raw_ctrl)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_ctrl_mean,
        COUNT(raw_ctrl)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_ctrl,
        AVG(raw_ko)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_ko_mean,
        COUNT(raw_ko)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_ko,
        AVG(raw_subr)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_subr_mean,
        COUNT(raw_subr)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_subr,
        AVG(raw_revc)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_revc_mean,
        COUNT(raw_revc)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_revc,
        AVG(raw_sdef)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_sdef_mean,
        COUNT(raw_sdef)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_sdef,
        AVG(raw_tdpc)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tdpc_mean,
        COUNT(raw_tdpc)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_tdpc,
        AVG(raw_revr)    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_revr_mean,
        COUNT(raw_revr)  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_revr
    FROM opp_ordered
),

-- 6. Rolling AAD per opponent (spread of prior encounters)
rolling_aad AS (
    SELECT *,
        AVG(ABS(raw_win   - opp_win_mean))   OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_win_aad,
        AVG(ABS(raw_tdpsa - opp_tdpsa_mean)) OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tdpsa_aad,
        AVG(ABS(raw_ctrl  - opp_ctrl_mean))  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_ctrl_aad,
        AVG(ABS(raw_ko    - opp_ko_mean))    OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_ko_aad,
        AVG(ABS(raw_subr  - opp_subr_mean))  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_subr_aad,
        AVG(ABS(raw_revc  - opp_revc_mean))  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_revc_aad,
        AVG(ABS(raw_sdef  - opp_sdef_mean))  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_sdef_aad,
        AVG(ABS(raw_tdpc  - opp_tdpc_mean))  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_tdpc_aad,
        AVG(ABS(raw_revr  - opp_revr_mean))  OVER (PARTITION BY opp_jfighter ORDER BY opp_rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS opp_revr_aad
    FROM rolling_mean
),

-- 7. Per-fight z-scores (Bayesian K=5, floor=0.05)
z_scores AS (
    SELECT
        r.DATE, r.jevent, r.jbout, r.jfighter, r.weightindex, r.jd,
        ROW_NUMBER() OVER (PARTITION BY r.jfighter ORDER BY r.DATE, r.jevent, r.jbout) AS rn,

        -- z_win
        CASE WHEN r.raw_win IS NULL THEN NULL ELSE
            (r.raw_win - (r.n_win/(r.n_win+5.0))*COALESCE(r.opp_win_mean,m.wc_win_mean) - (5.0/(r.n_win+5.0))*m.wc_win_mean) /
            CASE WHEN ((r.n_win/(r.n_win+5.0))*COALESCE(r.opp_win_aad,a.wc_win_aad)+(5.0/(r.n_win+5.0))*a.wc_win_aad) IS NULL THEN NULL
                 WHEN ((r.n_win/(r.n_win+5.0))*COALESCE(r.opp_win_aad,a.wc_win_aad)+(5.0/(r.n_win+5.0))*a.wc_win_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_win/(r.n_win+5.0))*COALESCE(r.opp_win_aad,a.wc_win_aad)+(5.0/(r.n_win+5.0))*a.wc_win_aad) END
        END AS z_win,

        -- z_tdpsa
        CASE WHEN r.raw_tdpsa IS NULL THEN NULL ELSE
            (r.raw_tdpsa - (r.n_tdpsa/(r.n_tdpsa+5.0))*COALESCE(r.opp_tdpsa_mean,m.wc_tdpsa_mean) - (5.0/(r.n_tdpsa+5.0))*m.wc_tdpsa_mean) /
            CASE WHEN ((r.n_tdpsa/(r.n_tdpsa+5.0))*COALESCE(r.opp_tdpsa_aad,a.wc_tdpsa_aad)+(5.0/(r.n_tdpsa+5.0))*a.wc_tdpsa_aad) IS NULL THEN NULL
                 WHEN ((r.n_tdpsa/(r.n_tdpsa+5.0))*COALESCE(r.opp_tdpsa_aad,a.wc_tdpsa_aad)+(5.0/(r.n_tdpsa+5.0))*a.wc_tdpsa_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_tdpsa/(r.n_tdpsa+5.0))*COALESCE(r.opp_tdpsa_aad,a.wc_tdpsa_aad)+(5.0/(r.n_tdpsa+5.0))*a.wc_tdpsa_aad) END
        END AS z_tdpsa,

        -- z_ctrl
        CASE WHEN r.raw_ctrl IS NULL THEN NULL ELSE
            (r.raw_ctrl - (r.n_ctrl/(r.n_ctrl+5.0))*COALESCE(r.opp_ctrl_mean,m.wc_ctrl_mean) - (5.0/(r.n_ctrl+5.0))*m.wc_ctrl_mean) /
            CASE WHEN ((r.n_ctrl/(r.n_ctrl+5.0))*COALESCE(r.opp_ctrl_aad,a.wc_ctrl_aad)+(5.0/(r.n_ctrl+5.0))*a.wc_ctrl_aad) IS NULL THEN NULL
                 WHEN ((r.n_ctrl/(r.n_ctrl+5.0))*COALESCE(r.opp_ctrl_aad,a.wc_ctrl_aad)+(5.0/(r.n_ctrl+5.0))*a.wc_ctrl_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_ctrl/(r.n_ctrl+5.0))*COALESCE(r.opp_ctrl_aad,a.wc_ctrl_aad)+(5.0/(r.n_ctrl+5.0))*a.wc_ctrl_aad) END
        END AS z_ctrl,

        -- z_ko
        CASE WHEN r.raw_ko IS NULL THEN NULL ELSE
            (r.raw_ko - (r.n_ko/(r.n_ko+5.0))*COALESCE(r.opp_ko_mean,m.wc_ko_mean) - (5.0/(r.n_ko+5.0))*m.wc_ko_mean) /
            CASE WHEN ((r.n_ko/(r.n_ko+5.0))*COALESCE(r.opp_ko_aad,a.wc_ko_aad)+(5.0/(r.n_ko+5.0))*a.wc_ko_aad) IS NULL THEN NULL
                 WHEN ((r.n_ko/(r.n_ko+5.0))*COALESCE(r.opp_ko_aad,a.wc_ko_aad)+(5.0/(r.n_ko+5.0))*a.wc_ko_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_ko/(r.n_ko+5.0))*COALESCE(r.opp_ko_aad,a.wc_ko_aad)+(5.0/(r.n_ko+5.0))*a.wc_ko_aad) END
        END AS z_ko,

        -- z_subr
        CASE WHEN r.raw_subr IS NULL THEN NULL ELSE
            (r.raw_subr - (r.n_subr/(r.n_subr+5.0))*COALESCE(r.opp_subr_mean,m.wc_subr_mean) - (5.0/(r.n_subr+5.0))*m.wc_subr_mean) /
            CASE WHEN ((r.n_subr/(r.n_subr+5.0))*COALESCE(r.opp_subr_aad,a.wc_subr_aad)+(5.0/(r.n_subr+5.0))*a.wc_subr_aad) IS NULL THEN NULL
                 WHEN ((r.n_subr/(r.n_subr+5.0))*COALESCE(r.opp_subr_aad,a.wc_subr_aad)+(5.0/(r.n_subr+5.0))*a.wc_subr_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_subr/(r.n_subr+5.0))*COALESCE(r.opp_subr_aad,a.wc_subr_aad)+(5.0/(r.n_subr+5.0))*a.wc_subr_aad) END
        END AS z_subr,

        -- z_revc (NULL - no reversal data)
        CASE WHEN r.raw_revc IS NULL THEN NULL ELSE
            (r.raw_revc - (r.n_revc/(r.n_revc+5.0))*COALESCE(r.opp_revc_mean,m.wc_revc_mean) - (5.0/(r.n_revc+5.0))*m.wc_revc_mean) /
            CASE WHEN ((r.n_revc/(r.n_revc+5.0))*COALESCE(r.opp_revc_aad,a.wc_revc_aad)+(5.0/(r.n_revc+5.0))*a.wc_revc_aad) IS NULL THEN NULL
                 WHEN ((r.n_revc/(r.n_revc+5.0))*COALESCE(r.opp_revc_aad,a.wc_revc_aad)+(5.0/(r.n_revc+5.0))*a.wc_revc_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_revc/(r.n_revc+5.0))*COALESCE(r.opp_revc_aad,a.wc_revc_aad)+(5.0/(r.n_revc+5.0))*a.wc_revc_aad) END
        END AS z_revc,

        -- z_sdef
        CASE WHEN r.raw_sdef IS NULL THEN NULL ELSE
            (r.raw_sdef - (r.n_sdef/(r.n_sdef+5.0))*COALESCE(r.opp_sdef_mean,m.wc_sdef_mean) - (5.0/(r.n_sdef+5.0))*m.wc_sdef_mean) /
            CASE WHEN ((r.n_sdef/(r.n_sdef+5.0))*COALESCE(r.opp_sdef_aad,a.wc_sdef_aad)+(5.0/(r.n_sdef+5.0))*a.wc_sdef_aad) IS NULL THEN NULL
                 WHEN ((r.n_sdef/(r.n_sdef+5.0))*COALESCE(r.opp_sdef_aad,a.wc_sdef_aad)+(5.0/(r.n_sdef+5.0))*a.wc_sdef_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_sdef/(r.n_sdef+5.0))*COALESCE(r.opp_sdef_aad,a.wc_sdef_aad)+(5.0/(r.n_sdef+5.0))*a.wc_sdef_aad) END
        END AS z_sdef,

        -- z_tdpc
        CASE WHEN r.raw_tdpc IS NULL THEN NULL ELSE
            (r.raw_tdpc - (r.n_tdpc/(r.n_tdpc+5.0))*COALESCE(r.opp_tdpc_mean,m.wc_tdpc_mean) - (5.0/(r.n_tdpc+5.0))*m.wc_tdpc_mean) /
            CASE WHEN ((r.n_tdpc/(r.n_tdpc+5.0))*COALESCE(r.opp_tdpc_aad,a.wc_tdpc_aad)+(5.0/(r.n_tdpc+5.0))*a.wc_tdpc_aad) IS NULL THEN NULL
                 WHEN ((r.n_tdpc/(r.n_tdpc+5.0))*COALESCE(r.opp_tdpc_aad,a.wc_tdpc_aad)+(5.0/(r.n_tdpc+5.0))*a.wc_tdpc_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_tdpc/(r.n_tdpc+5.0))*COALESCE(r.opp_tdpc_aad,a.wc_tdpc_aad)+(5.0/(r.n_tdpc+5.0))*a.wc_tdpc_aad) END
        END AS z_tdpc,

        -- z_revr (NULL - no reversal data)
        CASE WHEN r.raw_revr IS NULL THEN NULL ELSE
            (r.raw_revr - (r.n_revr/(r.n_revr+5.0))*COALESCE(r.opp_revr_mean,m.wc_revr_mean) - (5.0/(r.n_revr+5.0))*m.wc_revr_mean) /
            CASE WHEN ((r.n_revr/(r.n_revr+5.0))*COALESCE(r.opp_revr_aad,a.wc_revr_aad)+(5.0/(r.n_revr+5.0))*a.wc_revr_aad) IS NULL THEN NULL
                 WHEN ((r.n_revr/(r.n_revr+5.0))*COALESCE(r.opp_revr_aad,a.wc_revr_aad)+(5.0/(r.n_revr+5.0))*a.wc_revr_aad) < 0.05 THEN 0.05
                 ELSE  ((r.n_revr/(r.n_revr+5.0))*COALESCE(r.opp_revr_aad,a.wc_revr_aad)+(5.0/(r.n_revr+5.0))*a.wc_revr_aad) END
        END AS z_revr

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
        CASE WHEN z_win   IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_win   * _sh)) END AS z_win,
        CASE WHEN z_tdpsa IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_tdpsa * _sh)) END AS z_tdpsa,
        CASE WHEN z_ctrl  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_ctrl  * _sh)) END AS z_ctrl,
        CASE WHEN z_ko    IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_ko    * _sh)) END AS z_ko,
        CASE WHEN z_subr  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_subr  * _sh)) END AS z_subr,
        CASE WHEN z_revc  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_revc  * _sh)) END AS z_revc,
        CASE WHEN z_sdef  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_sdef  * _sh)) END AS z_sdef,
        CASE WHEN z_tdpc  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_tdpc  * _sh)) END AS z_tdpc,
        CASE WHEN z_revr  IS NULL THEN NULL ELSE MIN(_cl, MAX(-_cl, z_revr  * _sh)) END AS z_revr
    FROM z_params
),

-- 10. Seed: first fight per fighter
ema_seed AS (
    SELECT *,
        CASE WHEN z_win   IS NULL THEN 0.0 ELSE z_win   END AS win_num,
        CASE WHEN z_win   IS NULL THEN 0.0 ELSE 1.0     END AS win_den,
        CASE WHEN z_tdpsa IS NULL THEN 0.0 ELSE z_tdpsa END AS tdpsa_num,
        CASE WHEN z_tdpsa IS NULL THEN 0.0 ELSE 1.0     END AS tdpsa_den,
        CASE WHEN z_ctrl  IS NULL THEN 0.0 ELSE z_ctrl  END AS ctrl_num,
        CASE WHEN z_ctrl  IS NULL THEN 0.0 ELSE 1.0     END AS ctrl_den,
        CASE WHEN z_ko    IS NULL THEN 0.0 ELSE z_ko    END AS ko_num,
        CASE WHEN z_ko    IS NULL THEN 0.0 ELSE 1.0     END AS ko_den,
        CASE WHEN z_subr  IS NULL THEN 0.0 ELSE z_subr  END AS subr_num,
        CASE WHEN z_subr  IS NULL THEN 0.0 ELSE 1.0     END AS subr_den,
        CASE WHEN z_revc  IS NULL THEN 0.0 ELSE z_revc  END AS revc_num,
        CASE WHEN z_revc  IS NULL THEN 0.0 ELSE 1.0     END AS revc_den,
        CASE WHEN z_sdef  IS NULL THEN 0.0 ELSE z_sdef  END AS sdef_num,
        CASE WHEN z_sdef  IS NULL THEN 0.0 ELSE 1.0     END AS sdef_den,
        CASE WHEN z_tdpc  IS NULL THEN 0.0 ELSE z_tdpc  END AS tdpc_num,
        CASE WHEN z_tdpc  IS NULL THEN 0.0 ELSE 1.0     END AS tdpc_den,
        CASE WHEN z_revr  IS NULL THEN 0.0 ELSE z_revr  END AS revr_num,
        CASE WHEN z_revr  IS NULL THEN 0.0 ELSE 1.0     END AS revr_den
    FROM z_shrunk WHERE rn = 1
),

-- 11. Recursive EMA-decay of per-fight z-scores (lambda=0.13 per year, ~5yr half-life)
ema_z AS (
    SELECT * FROM ema_seed
    UNION ALL
    SELECT
        s.DATE, s.jevent, s.jbout, s.jfighter, s.weightindex, s.jd, s.rn,
        s.z_win, s.z_tdpsa, s.z_ctrl, s.z_ko, s.z_subr, s.z_revc, s.z_sdef, s.z_tdpc, s.z_revr,
        (e.win_num   * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_win   IS NULL THEN 0.0 ELSE s.z_win   END AS win_num,
        (e.win_den   * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_win   IS NULL THEN 0.0 ELSE 1.0      END AS win_den,
        (e.tdpsa_num * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tdpsa IS NULL THEN 0.0 ELSE s.z_tdpsa END AS tdpsa_num,
        (e.tdpsa_den * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tdpsa IS NULL THEN 0.0 ELSE 1.0      END AS tdpsa_den,
        (e.ctrl_num  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_ctrl  IS NULL THEN 0.0 ELSE s.z_ctrl  END AS ctrl_num,
        (e.ctrl_den  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_ctrl  IS NULL THEN 0.0 ELSE 1.0      END AS ctrl_den,
        (e.ko_num    * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_ko    IS NULL THEN 0.0 ELSE s.z_ko    END AS ko_num,
        (e.ko_den    * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_ko    IS NULL THEN 0.0 ELSE 1.0      END AS ko_den,
        (e.subr_num  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_subr  IS NULL THEN 0.0 ELSE s.z_subr  END AS subr_num,
        (e.subr_den  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_subr  IS NULL THEN 0.0 ELSE 1.0      END AS subr_den,
        (e.revc_num  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_revc  IS NULL THEN 0.0 ELSE s.z_revc  END AS revc_num,
        (e.revc_den  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_revc  IS NULL THEN 0.0 ELSE 1.0      END AS revc_den,
        (e.sdef_num  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_sdef  IS NULL THEN 0.0 ELSE s.z_sdef  END AS sdef_num,
        (e.sdef_den  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_sdef  IS NULL THEN 0.0 ELSE 1.0      END AS sdef_den,
        (e.tdpc_num  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tdpc  IS NULL THEN 0.0 ELSE s.z_tdpc  END AS tdpc_num,
        (e.tdpc_den  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_tdpc  IS NULL THEN 0.0 ELSE 1.0      END AS tdpc_den,
        (e.revr_num  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_revr  IS NULL THEN 0.0 ELSE s.z_revr  END AS revr_num,
        (e.revr_den  * exp(-0.13 * (s.jd - e.jd) / 365.25)) + CASE WHEN s.z_revr  IS NULL THEN 0.0 ELSE 1.0      END AS revr_den
    FROM z_shrunk s
    JOIN ema_z e ON e.jfighter = s.jfighter AND e.rn = s.rn - 1
),

-- 12. LAG all EMA-z accumulators by 1 fight (no current-fight leakage)
lagged_ema_z AS (
    SELECT
        DATE, jevent, jbout, jfighter, weightindex,
        LAG(win_num,   1, 0.0) OVER w AS win_num,
        LAG(win_den,   1, 0.0) OVER w AS win_den,
        LAG(tdpsa_num, 1, 0.0) OVER w AS tdpsa_num,
        LAG(tdpsa_den, 1, 0.0) OVER w AS tdpsa_den,
        LAG(ctrl_num,  1, 0.0) OVER w AS ctrl_num,
        LAG(ctrl_den,  1, 0.0) OVER w AS ctrl_den,
        LAG(ko_num,    1, 0.0) OVER w AS ko_num,
        LAG(ko_den,    1, 0.0) OVER w AS ko_den,
        LAG(subr_num,  1, 0.0) OVER w AS subr_num,
        LAG(subr_den,  1, 0.0) OVER w AS subr_den,
        LAG(revc_num,  1, 0.0) OVER w AS revc_num,
        LAG(revc_den,  1, 0.0) OVER w AS revc_den,
        LAG(sdef_num,  1, 0.0) OVER w AS sdef_num,
        LAG(sdef_den,  1, 0.0) OVER w AS sdef_den,
        LAG(tdpc_num,  1, 0.0) OVER w AS tdpc_num,
        LAG(tdpc_den,  1, 0.0) OVER w AS tdpc_den,
        LAG(revr_num,  1, 0.0) OVER w AS revr_num,
        LAG(revr_den,  1, 0.0) OVER w AS revr_den
    FROM ema_z
    WINDOW w AS (PARTITION BY jfighter ORDER BY rn)
)

SELECT
    DATE, jevent, jbout, jfighter, weightindex,
    CASE WHEN win_den   = 0 THEN NULL ELSE win_num   / win_den   END AS adjperf_win_dec_avg,
    CASE WHEN tdpsa_den = 0 THEN NULL ELSE tdpsa_num / tdpsa_den END AS adjperf_td_per_sigstr_att_dec_avg,
    CASE WHEN ctrl_den  = 0 THEN NULL ELSE ctrl_num  / ctrl_den  END AS adjperf_ctrl_ratio_dec_avg,
    CASE WHEN ko_den    = 0 THEN NULL ELSE ko_num    / ko_den    END AS adjperf_ko_ratio_dec_avg,
    CASE WHEN subr_den  = 0 THEN NULL ELSE subr_num  / subr_den  END AS adjperf_sub_att_ratio_dec_avg,
    CASE WHEN revc_den  = 0 THEN NULL ELSE revc_num  / revc_den  END AS adjperf_rev_per_ctrlopp_dec_avg,
    CASE WHEN sdef_den  = 0 THEN NULL ELSE sdef_num  / sdef_den  END AS adjperf_sub_def_dec_avg,
    CASE WHEN tdpc_den  = 0 THEN NULL ELSE tdpc_num  / tdpc_den  END AS adjperf_td_land_per_ctrl_dec_avg,
    CASE WHEN revr_den  = 0 THEN NULL ELSE revr_num  / revr_den  END AS adjperf_rev_ratio_dec_avg
FROM lagged_ema_z;

CREATE INDEX IF NOT EXISTS idx_adjperf_new3_fast_bout
ON adjperf_new3_fast(jevent, jbout, jfighter);
