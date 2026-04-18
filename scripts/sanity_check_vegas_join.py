"""Sanity-check the model-vs-vegas join. Three questions:

1. Orientation: does p_model really mean P(fighter_a wins)? Verify on known bouts.
2. Coverage bias: are matched bouts disproportionately main-card / favorites?
3. ROI plausibility: top 10 winning bets — are they suspiciously long-odds longshots?
"""
import json, sqlite3, numpy as np, pandas as pd
from pathlib import Path

d = json.load(open("app/models/blend/backtest_predictions.json"))
pred = pd.DataFrame(d["predictions"])
pred["bout_date"] = pd.to_datetime(pred["bout_date"])
pred["p_model"] = 0.5*pred["p_lr"] + 0.5*pred["p_xgb"]
pred["y_true"]  = (pred["actual_winner"] == pred["fighter_a"]).astype(int)

con = sqlite3.connect("data/sqlite_db/slim_scrapper.db")
odds = pd.read_sql(
    "SELECT DATE as bout_date, jfighter, opp_jfighter, avg_odds_f1, avg_odds_f2 "
    "FROM ufc_fight_odds", con, parse_dates=["bout_date"])
con.close()

def am_to_implied(a): return np.where(a<0, -a/(-a+100), 100/(a+100))
def am_to_decimal(a): return np.where(a<0, 1+100/(-a), 1+a/100)

odds["p_raw_jf"]  = am_to_implied(odds["avg_odds_f1"])
odds["p_raw_opp"] = am_to_implied(odds["avg_odds_f2"])
s = odds["p_raw_jf"] + odds["p_raw_opp"]
odds["p_v_jf"]  = odds["p_raw_jf"]/s
odds["p_v_opp"] = odds["p_raw_opp"]/s
odds["dec_jf"]  = am_to_decimal(odds["avg_odds_f1"])
odds["dec_opp"] = am_to_decimal(odds["avg_odds_f2"])

pred["pair"] = [tuple(sorted([a,b])) for a,b in zip(pred["fighter_a"], pred["fighter_b"])]
odds["pair"] = [tuple(sorted([a,b])) for a,b in zip(odds["jfighter"], odds["opp_jfighter"])]

m = pred.merge(
    odds[["bout_date","pair","jfighter","opp_jfighter","p_v_jf","p_v_opp",
          "dec_jf","dec_opp","avg_odds_f1","avg_odds_f2"]],
    on=["bout_date","pair"], how="left")
same = m["fighter_a"] == m["jfighter"]
m["p_vegas_a"]  = np.where(same, m["p_v_jf"],  m["p_v_opp"])
m["dec_odds_a"] = np.where(same, m["dec_jf"],  m["dec_opp"])
m["dec_odds_b"] = np.where(same, m["dec_opp"], m["dec_jf"])
m["am_odds_a"]  = np.where(same, m["avg_odds_f1"], m["avg_odds_f2"])
m["am_odds_b"]  = np.where(same, m["avg_odds_f2"], m["avg_odds_f1"])

M = m[m["p_vegas_a"].notna()].copy()
print(f"Matched: {len(M)}\n")

# ── Q1: Orientation check ─────────────────────────────────────────────
print("="*90)
print("Q1: ORIENTATION — does p_model/p_vegas align with fighter_a?")
print("="*90)
# Pick 10 cases where Vegas was very lopsided (easy to verify)
heavy = M[(M["p_vegas_a"] > 0.75) | (M["p_vegas_a"] < 0.25)].copy()
heavy["vegas_pick_a"] = heavy["p_vegas_a"] >= 0.5
heavy["vegas_right"]  = heavy["vegas_pick_a"] == heavy["y_true"].astype(bool)
print("On Vegas-lopsided bouts (p_vegas_a > .75 or < .25):")
print(f"  n = {len(heavy)}")
print(f"  Vegas accuracy when heavy: {heavy['vegas_right'].mean():.3f}")
print(f"  (If orientation broken, this would be close to ~0.25, not ~0.75)")
print()
# Spot-check: 5 most lopsided matched bouts
heavy["abs_edge"] = (heavy["p_vegas_a"] - 0.5).abs()
print("Top 5 most lopsided bouts:")
cols = ["bout_date","fighter_a","fighter_b","am_odds_a","am_odds_b",
        "p_vegas_a","p_model","y_true","actual_winner"]
print(heavy.sort_values("abs_edge", ascending=False).head(5)[cols].to_string(index=False))

# ── Q2: Coverage bias ─────────────────────────────────────────────────
print("\n" + "="*90)
print("Q2: COVERAGE BIAS — are matched bouts easier than unmatched?")
print("="*90)
m["has_odds"] = m["p_vegas_a"].notna()
grp = m.groupby("has_odds").agg(
    n=("y_true","size"),
    model_acc=("p_model", lambda s: ((s>=0.5)==m.loc[s.index,"y_true"].astype(bool)).mean()),
    model_confidence=("p_model", lambda s: (s-0.5).abs().mean()),
)
print(grp.to_string())
print("If 'has_odds=True' has higher model_acc, that's survivorship bias —")
print("we're comparing on the easier bouts.")

# ── Q3: ROI drivers ───────────────────────────────────────────────────
print("\n" + "="*90)
print("Q3: ROI DRIVERS — which bouts actually made the money?")
print("="*90)
M["pick_a"] = M["p_model"] >= 0.5
M["odds_pick"] = np.where(M["pick_a"], M["dec_odds_a"], M["dec_odds_b"])
M["am_pick"]   = np.where(M["pick_a"], M["am_odds_a"], M["am_odds_b"])
M["won"] = np.where(M["pick_a"], M["y_true"]==1, M["y_true"]==0)
M["pnl"] = np.where(M["won"], M["odds_pick"]-1.0, -1.0)

print(f"Total bets: {len(M)}   won: {M['won'].sum()}  "
      f"win rate: {M['won'].mean():.3f}   ROI: {M['pnl'].mean()*100:+.2f}%")
print(f"\nP&L buckets (by American odds of the pick):")
M["bucket"] = pd.cut(M["am_pick"], bins=[-10000,-300,-200,-150,100,150,200,300,10000],
                     labels=["<=-300","-300..-200","-200..-150","-150..+100",
                             "+100..+150","+150..+200","+200..+300",">+300"])
b = M.groupby("bucket", observed=True).agg(
    n=("pnl","size"), wins=("won","sum"),
    win_rate=("won","mean"), roi_pct=("pnl", lambda s: s.mean()*100),
    total_pnl=("pnl","sum"))
print(b.to_string())

print("\nTop 10 biggest winners (largest PnL per bet):")
print(M.nlargest(10, "pnl")[["bout_date","fighter_a","fighter_b","pick_a",
                             "am_pick","p_model","p_vegas_a","won","pnl"]]
      .to_string(index=False))
print("\nTop 10 biggest losers (largest -1.0 bets are just -1.0 each, so show by odds):")
lost = M[~M["won"]].copy()
print(lost.nlargest(10, "am_pick")[["bout_date","fighter_a","fighter_b","pick_a",
                                    "am_pick","p_model","p_vegas_a"]]
      .to_string(index=False))
