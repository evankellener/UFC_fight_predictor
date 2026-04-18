"""Scrape odds for bouts missing from data/tmp/odds_table.csv.

Identifies every prediction without a match, groups by date, fetches
historical snapshots (day-of + next-day) from The-Odds-API, matches by
fuzzy name similarity, appends new rows to odds_table.csv (long format,
same schema as notebook Cell 6).
"""
import json, re, os, time, requests
import numpy as np, pandas as pd
from pathlib import Path
from datetime import timedelta
from difflib import SequenceMatcher

REPO_MAIN = Path("/Users/evankellener/Desktop/UFC_fight_predictor")
ODDS_CSV  = REPO_MAIN / "data/tmp/odds_table.csv"
PRED_PATH = Path("app/models/blend/backtest_predictions.json")

# Load .env
for line in open(REPO_MAIN/".env"):
    if "=" in line and not line.strip().startswith("#"):
        k,v = line.strip().split("=",1)
        os.environ[k] = v.strip().strip('"').strip("'")
API_KEY = os.environ.get("ODDS_API_KEY")
assert API_KEY, "ODDS_API_KEY missing"

SPORT = "mma_mixed_martial_arts"
HIST_URL = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds"
MAIN_BOOKS = ["draftkings","fanduel","betmgm","bet365","bovada"]

NAME_ALIASES = {
    "patriciofreire":   "patriciopitbull",
    "arianelipski":     "arianedasilva",
    "teciatorres":      "teciapennington",
}
def tokens(n):
    """Sorted-token form handling BOTH 'ZhangWeili' and 'Weili Zhang' → same output."""
    s = str(n).strip()
    # split camelCase: insert space before uppercase-then-lowercase
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    parts = []
    for tok in re.split(r"\s+", s):
        tok = re.sub(r"\W+","", tok).lower()
        if tok and tok not in ("jr","sr","ii","iii","iv","v","da","de","del","van","von","dos"):
            parts.append(tok)
    return "+".join(sorted(parts))
def norm(n):
    x = re.sub(r"\W+", "", str(n)).lower()
    return NAME_ALIASES.get(x, x)
def sim(a,b): return SequenceMatcher(None,a,b).ratio()
def tok_sim(a, b):
    """Token-set similarity: same characters regardless of order."""
    ta, tb = set(tokens(a).split("+")), set(tokens(b).split("+"))
    if not ta or not tb: return 0.0
    return len(ta & tb) / max(len(ta), len(tb))

# ── Find unmatched bouts ──────────────────────────────────────────────
d = json.load(open(PRED_PATH))
pred = pd.DataFrame(d["predictions"])
pred["bout_date"] = pd.to_datetime(pred["bout_date"])
pred["jf_a"] = pred["fighter_a"].apply(norm)
pred["jf_b"] = pred["fighter_b"].apply(norm)
pred["pair"] = [tuple(sorted([a,b])) for a,b in zip(pred["jf_a"], pred["jf_b"])]

raw = pd.read_csv(ODDS_CSV, parse_dates=["DATE"])
raw["jf"] = raw["FIGHTER"].apply(norm)
pairs = raw.groupby(["DATE","BOUT"]).agg(jfs=("jf",list)).reset_index()
pairs["pair"] = pairs["jfs"].apply(lambda xs: tuple(sorted(xs)) if len(xs)==2 else None)

m = pred.merge(pairs[["DATE","pair","BOUT"]].rename(columns={"DATE":"bout_date"}),
               on=["bout_date","pair"], how="left")
u = m[m["BOUT"].isna()][["bout_date","fighter_a","fighter_b","fold_num"]]
dates = sorted(u["bout_date"].dt.strftime("%Y-%m-%d").unique())
print(f"Unmatched bouts: {len(u)} across {len(dates)} dates")
print(f"Dates to fetch: {dates}")

# ── Fetch snapshots ──────────────────────────────────────────────────
def fetch(ts_iso):
    r = requests.get(HIST_URL, params={
        "apiKey": API_KEY, "regions":"us", "markets":"h2h",
        "oddsFormat":"american", "dateFormat":"iso", "date": ts_iso,
    })
    if r.status_code != 200:
        print(f"  {ts_iso}: HTTP {r.status_code}")
        return []
    rem = r.headers.get("x-requests-remaining","?")
    data = r.json().get("data",[])
    print(f"  {ts_iso[:10]} T{ts_iso[11:19]}: {len(data)} events  (remaining: {rem})")
    return data

all_events = []
for d_str in dates:
    base = pd.Timestamp(d_str)
    # Try a few times of day — sometimes snapshots before commence_time lack markets
    for delta_days, hour in [(0, 0), (0, 12), (1, 0), (1, 12)]:
        ts = (base + pd.Timedelta(days=delta_days, hours=hour)).strftime("%Y-%m-%dT%H:00:00Z")
        all_events.extend(fetch(ts))
        time.sleep(0.25)

seen = {e["id"]: e for e in all_events}
print(f"\nUnique events fetched: {len(seen)}")

# ── Extract avg odds per event ────────────────────────────────────────
records = []
for e in seen.values():
    ct = pd.to_datetime(e["commence_time"], utc=True)
    home_n = norm(e.get("home_team",""))
    away_n = norm(e.get("away_team",""))
    h_prices, a_prices = [], []
    for bk in e.get("bookmakers",[]):
        if bk["key"] not in MAIN_BOOKS: continue
        for mkt in bk.get("markets",[]):
            if mkt["key"] != "h2h": continue
            for oc in mkt["outcomes"]:
                n = norm(oc["name"])
                if n == home_n: h_prices.append(oc["price"])
                elif n == away_n: a_prices.append(oc["price"])
    if h_prices and a_prices:
        records.append(dict(
            commence_dt=ct, home=home_n, away=away_n,
            home_name=e.get("home_team",""), away_name=e.get("away_team",""),
            avg_odds_home=np.mean(h_prices), avg_odds_away=np.mean(a_prices)))
ev_df = pd.DataFrame(records)
print(f"Events with h2h odds from main books: {len(ev_df)}")

# ── Match unmatched predictions to events ─────────────────────────────
def am_to_prob(o):
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)

new_rows = []
matched = 0
for _, f in u.iterrows():
    fa, fb = f["fighter_a"], f["fighter_b"]
    fdate = f["bout_date"]
    best, best_score = None, 0
    for _, ev in ev_df.iterrows():
        if abs((ev["commence_dt"].tz_localize(None) - fdate).days) > 3: continue
        # Try both string-sim AND token-sim; take max
        s1 = max(sim(norm(fa), ev["home"]), sim(norm(fa), ev["away"]),
                 tok_sim(fa, ev["home_name"]), tok_sim(fa, ev["away_name"]))
        s2 = max(sim(norm(fb), ev["home"]), sim(norm(fb), ev["away"]),
                 tok_sim(fb, ev["home_name"]), tok_sim(fb, ev["away_name"]))
        score = s1+s2
        if score > best_score and s1 >= 0.7 and s2 >= 0.7:
            best_score = score; best = ev
    if best is None:
        print(f"  NO MATCH: {fdate.date()}  {fa} vs {fb}")
        continue
    matched += 1
    # orient using whichever similarity is higher
    s_a_home = max(sim(norm(fa), best["home"]), tok_sim(fa, best["home_name"]))
    s_a_away = max(sim(norm(fa), best["away"]), tok_sim(fa, best["away_name"]))
    is_home = s_a_home > s_a_away
    o1 = best["avg_odds_home"] if is_home else best["avg_odds_away"]
    o2 = best["avg_odds_away"] if is_home else best["avg_odds_home"]
    p1, p2 = am_to_prob(o1), am_to_prob(o2)
    total = p1+p2
    bout_str = f"{f['fighter_a']} vs {f['fighter_b']}"   # keep prediction ordering
    new_rows.append(dict(DATE=fdate, BOUT=bout_str,
                         FIGHTER=f["fighter_a"], prob_norm=p1/total, odds=o1))
    new_rows.append(dict(DATE=fdate, BOUT=bout_str,
                         FIGHTER=f["fighter_b"], prob_norm=p2/total, odds=o2))
    print(f"  MATCHED: {fdate.date()}  {f['fighter_a']} ({o1:+.0f}) vs {f['fighter_b']} ({o2:+.0f})")

print(f"\nMatched: {matched}/{len(u)}")

# ── Append to CSV (dedupe on DATE+FIGHTER+BOUT) ───────────────────────
if new_rows:
    new_df = pd.DataFrame(new_rows)
    new_df["DATE"] = pd.to_datetime(new_df["DATE"])
    old = pd.read_csv(ODDS_CSV, parse_dates=["DATE"])
    combined = pd.concat([old, new_df], ignore_index=True).drop_duplicates(
        subset=["DATE","FIGHTER","BOUT"], keep="last")
    combined.to_csv(ODDS_CSV, index=False)
    print(f"Appended {len(new_df)} rows → {ODDS_CSV}")
    print(f"CSV now has {len(combined)} rows ({len(combined)//2} bouts)")
else:
    print("No new rows to append")
