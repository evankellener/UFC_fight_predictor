"""
MMA-AI Pipeline Configuration — all hyperparameters in one place.
Two presets: v7 (Sep 2025) and jan26 (Jan 2026).
"""

from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "sqlite_db" / "sqlite_scrapper.db")
ERA_CUTOFF = "2014-04-01"

# ── Presets ────────────────────────────────────────────────────────────────

V7_CONFIG = {
    "name": "v7",
    "decay_lambda": 0.13,          # ~5 year half-life
    "start_date": "2014-04-01",
    "shuffle": True,
    "calibrate": True,
    "train_size": 0.75,
    "val_size": 0.15,
    "test_size": 0.10,
}

JAN26_CONFIG = {
    "name": "jan26",
    "decay_lambda": 0.347,         # ~2 year half-life
    "start_date": "2016-01-01",
    "shuffle": False,              # temporal CV, no shuffle
    "calibrate": True,
    "train_size": 0.75,
    "val_size": 0.15,
    "test_size": 0.10,
}

# ── Poisson-Gamma pseudo-minutes (τ) ──────────────────────────────────────

PG_TAU_GLOBAL = {
    # Full-fight count stats
    "sig_str_land": 0.7,  "sig_str_att": 0.7,
    "head_land": 0.8,     "head_att": 0.8,
    "body_land": 2.5,     "body_att": 2.5,
    "leg_land": 2.1,      "leg_att": 2.1,
    "dist_land": 0.7,     "dist_att": 0.7,
    "clinch_land": 2.5,   "clinch_att": 2.5,
    "ground_land": 2.5,   "ground_att": 2.5,
    "td_land": 7.0,       "td_att": 7.0,
    "sub_att": 12.0,
    "kd": 20.0,
    "rev": 42.0,
}

PG_TAU_RD1 = {
    # Round 1 specific
    "sig_str_land_rd1": 0.7,  "sig_str_att_rd1": 0.7,
    "head_land_rd1": 0.7,     "head_att_rd1": 0.7,
    "body_land_rd1": 2.5,     "body_att_rd1": 2.5,
    "leg_land_rd1": 1.7,      "leg_att_rd1": 1.7,
    "td_land_rd1": 9.0,       "td_att_rd1": 9.0,
    "sub_att_rd1": 15.0,
    "kd_rd1": 12.0,
    "rev_rd1": 60.0,
}

# Per-weight-class overrides (weightindex → stat → τ)
PG_TAU_WC_OVERRIDES = {
    1: {"rev": 22.0},                    # Flyweight
    7: {"head_land_rd1": 0.5},           # Light Heavyweight
    8: {"td_land": 5.0, "td_att": 5.0,
        "td_land_rd1": 4.0, "td_att_rd1": 4.0},  # Heavyweight
}

# ── Beta-Binomial pseudo-counts (τ) ──────────────────────────────────────

BB_TAU_GLOBAL = {
    "ko": 23,
    "win": 25,
    "decision": 20,
    "sub_land": 9,
    "ctrl": 2,           # modeled as time-share (seconds)
}

BB_TAU_RD1 = {
    "ko_rd1": 17,
    "win_rd1": 15,
    "decision_rd1": 16,
    "sub_land_rd1": 7,
    "ctrl_rd1": 1,
}

BB_TAU_WC_OVERRIDES = {
    4: {"sub_land": 3},                  # Featherweight
    7: {"ctrl": 1.5},                    # Light Heavyweight
    8: {"ctrl": 1.5},                    # Heavyweight
}

# ── AdjPerf K values ─────────────────────────────────────────────────────

ADJPERF_K = {
    "default": {"K_mean": 4.0, "K_mad": 4.0},
    "striking": {"K_mean": 8.0, "K_mad": 12.0},   # sig_str, head, body, leg, dist, clinch, ground
    "grappling": {"K_mean": 5.0, "K_mad": 8.0},    # td, sub, ctrl, rev
    "rare": {"K_mean": 4.0, "K_mad": 4.0},          # kd, ko
}

# Map stat prefixes to families for K selection
STAT_FAMILY = {
    "sig_str": "striking", "head": "striking", "body": "striking",
    "leg": "striking", "dist": "striking", "clinch": "striking",
    "ground": "striking", "total_str": "striking",
    "td": "grappling", "sub": "grappling", "ctrl": "grappling", "rev": "grappling",
    "kd": "rare", "ko": "rare",
}

ADJPERF_CLIP = 7.0
MAD_FLOOR_PERCENTILE = 5  # 5th percentile of per-opponent MADs

# ── AutoGluon ─────────────────────────────────────────────────────────────

AG_HYPERPARAMETERS = {
    "CAT": {},
    "GBM": [{"extra_trees": True}, {}],
    "NN_TORCH": {},
}

AG_SETTINGS = {
    "num_stack_levels": 2,
    "num_bag_folds": 4,
    "num_bag_sets": 2,
    "use_bag_holdout": True,
    "time_limit": 900,
    "keep_only_best": False,
}

# ── Weight class index mapping ────────────────────────────────────────────
WEIGHTCLASS_NAMES = {
    0: "Catchweight",
    1: "Flyweight",
    2: "Bantamweight",
    3: "Featherweight",
    4: "Lightweight",
    5: "Welterweight",
    6: "Middleweight",
    7: "Light Heavyweight",
    8: "Heavyweight",
    9: "W_Strawweight",
    10: "W_Flyweight",
    11: "W_Bantamweight",
    12: "W_Featherweight",
}
