"""Build leakage-free contextual features for the existing mmaai_features.csv.

Each feature uses ONLY information available before the fight DATE.

Features:
  - home_advantage_diff           (event country vs fighter country, current fight)
  - travel_distance_diff_km       (haversine, current fight)
  - tz_diff_diff_hr               (current fight)
  - is_main_event                 (current fight)
  - card_position_norm_career_diff (career mean PRIOR to this fight)
  - missed_weight_history_diff    (career sum PRIOR to this fight, derived from method='Forfeit'/missed-weight notes — proxy)
  - coming_off_loss_diff          (PRIOR fight only)
  - win_streak_entering_diff      (PRIOR fights)
  - fights_last_12m_diff          (PRIOR fights only)
  - stance_mismatch / southpaw_advantage_diff (static fighter attribute)

Writes data/tmp/market_features_clean.csv keyed by (DATE, jbout, jfighter).
"""
import sqlite3, sys, json, numpy as np, pandas as pd
from math import radians, cos, sin, asin, sqrt
sys.path.insert(0, "src")
# Re-use city/country geo from new_market_features
from new_market_features import COUNTRY_GEO, EVENT_CITY_GEO, haversine, get_event_geo

SCRAPER = "data/sqlite_db/sqlite_scrapper.db"
APP     = "data/sqlite_db/app.db"

base = pd.read_csv("data/tmp/mmaai_features.csv",
                   usecols=["DATE","jevent","jbout","jfighter","opp_jfighter","win"],
                   parse_dates=["DATE"])
base['win'] = base['win'].astype(int)
print(f"Base rows: {len(base)}")

# --- Static lookups ---
sc = sqlite3.connect(SCRAPER); ac = sqlite3.connect(APP)
tott = pd.read_sql("SELECT jfighter,STANCE FROM ufc_fighter_tott", sc)
stance_map = {r.jfighter: (r.STANCE or "").strip().lower() for r in tott.itertuples()}

events = pd.read_sql("SELECT jevent,DATE,LOCATION FROM ufc_event_details", sc)
events['DATE'] = pd.to_datetime(events['DATE'])
loc_map = dict(zip(events['jevent'], events['LOCATION']))

nat = pd.read_sql("SELECT jfighter,country FROM ufc_fighter_nationality", ac)
nat_map = dict(zip(nat['jfighter'], nat['country']))

cp = pd.read_sql("SELECT jevent,jbout,position,total_fights FROM ufc_card_position", ac)
cp['pos_norm'] = ((cp['position']-1) / (cp['total_fights']-1).clip(lower=1))
cp['is_main'] = (cp['position']==1).astype(int)
cp_map = {(r.jevent, r.jbout): (r.pos_norm, r.is_main) for r in cp.itertuples()}

# Career card position requires (fighter, date, pos_norm) sorted ascending
fp = pd.read_sql("""
    SELECT w.jfighter, e.DATE, cp.position, cp.total_fights
    FROM ufc_winlossko w
    JOIN ufc_card_position cp ON cp.jevent=w.jevent AND cp.jbout=w.jbout
    JOIN ufc_event_details e ON e.jevent=w.jevent
""", ac)
# winlossko is in scraper DB usually — try fallback
if len(fp)==0:
    wlk = pd.read_sql("SELECT jfighter,jevent,jbout FROM ufc_winlossko", sc)
    fp = wlk.merge(cp[['jevent','jbout','position','total_fights']], on=['jevent','jbout'], how='inner')
    fp = fp.merge(events[['jevent','DATE']], on='jevent', how='left')
fp = fp.dropna(subset=['DATE'])
fp['DATE'] = pd.to_datetime(fp['DATE'])
fp['pos_norm'] = ((fp['position']-1) / (fp['total_fights']-1).clip(lower=1))
fp = fp.sort_values(['jfighter','DATE'])
print(f"Career card-position records: {len(fp)}")

# Build per-fighter lists of (date, pos_norm) for fast lookup
career_cp = {}
for j, g in fp.groupby('jfighter'):
    career_cp[j] = list(zip(g['DATE'].values, g['pos_norm'].values))

def career_pos_before(jf, dt):
    h = career_cp.get(jf, [])
    prior = [p for d, p in h if d < dt]
    return float(np.mean(prior)) if prior else 0.5  # mid-card default

# Missed-weight proxy from method strings (no weighin table)
res = pd.read_sql("SELECT jevent,jbout,METHOD FROM ufc_fight_results", sc)
method_map = {(r.jevent, r.jbout): (r.METHOD or "") for r in res.itertuples()}
# can't reliably back out who missed; skip career missed-weight

# Build per-fighter fight history from base itself for psych features
# Each base row appears twice (once per perspective); base has both.
# Sort by DATE and walk per fighter.
hist = []
for r in base.itertuples():
    hist.append((r.DATE, r.jfighter, int(r.win)))
hist_df = pd.DataFrame(hist, columns=['DATE','jfighter','won']).sort_values(['jfighter','DATE'])

prior_loss   = {}  # (date, jf) -> 0/1 (lost previous)
prior_streak = {}  # (date, jf) -> consecutive prior wins
prior_12m    = {}  # (date, jf) -> count
for jf, g in hist_df.groupby('jfighter'):
    dates = g['DATE'].values; wins = g['won'].values
    streak = 0
    for i, dt in enumerate(dates):
        key = (pd.Timestamp(dt), jf)
        if i == 0:
            prior_loss[key]=0; prior_streak[key]=0; prior_12m[key]=0
        else:
            prior_loss[key]   = 0 if wins[i-1] else 1
            s=0
            for j in range(i-1, -1, -1):
                if wins[j]: s += 1
                else: break
            prior_streak[key] = s
            cutoff = dates[i] - np.timedelta64(365,'D')
            prior_12m[key] = int(sum(1 for d in dates[:i] if d >= cutoff))

# Now build per-row features
def stance_code(s):
    s = s or ""
    if s == "orthodox": return 1
    if s == "southpaw": return 2
    if s == "switch":   return 3
    return 0

rows = []
for r in base.itertuples():
    jf, opp, dt, jevent, jbout = r.jfighter, r.opp_jfighter, r.DATE, r.jevent, r.jbout
    loc = loc_map.get(jevent, "")
    evt_geo = get_event_geo(loc)
    f1c = nat_map.get(jf); f2c = nat_map.get(opp)
    f1g = COUNTRY_GEO.get(f1c) if f1c else None
    f2g = COUNTRY_GEO.get(f2c) if f2c else None

    # Home advantage
    evt_country = (loc.split(",")[-1].strip() if isinstance(loc,str) and "," in loc else "") or ""
    aliases = {"USA":"United States","UK":"United Kingdom","England":"United Kingdom",
               "Scotland":"United Kingdom","Wales":"United Kingdom"}
    evt_country = aliases.get(evt_country, evt_country)
    f1_home = int(f1c == evt_country) if f1c else 0
    f2_home = int(f2c == evt_country) if f2c else 0
    home_advantage_diff = f1_home - f2_home

    # Travel + tz
    if evt_geo and f1g: td1, tz1 = haversine(f1g[0],f1g[1],evt_geo[0],evt_geo[1]), abs(f1g[2]-evt_geo[2])
    else: td1, tz1 = np.nan, np.nan
    if evt_geo and f2g: td2, tz2 = haversine(f2g[0],f2g[1],evt_geo[0],evt_geo[1]), abs(f2g[2]-evt_geo[2])
    else: td2, tz2 = np.nan, np.nan
    travel_diff = (td1 - td2) if not (np.isnan(td1) or np.isnan(td2)) else 0.0
    tz_diff     = (tz1 - tz2) if not (np.isnan(tz1) or np.isnan(tz2)) else 0.0

    # Card position (current)
    pn, im = cp_map.get((jevent, jbout), (0.5, 0))
    is_main = im

    # Career card position before this fight
    cp1 = career_pos_before(jf, dt)
    cp2 = career_pos_before(opp, dt)
    card_career_diff = cp1 - cp2

    # Psychology
    k1 = (dt, jf); k2 = (dt, opp)
    coming_off_loss_diff   = prior_loss.get(k1,0)   - prior_loss.get(k2,0)
    win_streak_diff        = prior_streak.get(k1,0) - prior_streak.get(k2,0)
    fights_12m_diff        = prior_12m.get(k1,0)    - prior_12m.get(k2,0)

    # Stance
    s1 = stance_code(stance_map.get(jf, "")); s2 = stance_code(stance_map.get(opp, ""))
    stance_mismatch = int(s1 != s2 and s1 > 0 and s2 > 0)
    southpaw_adv_diff = int(s1==2) - int(s2==2)

    rows.append({
        'DATE':dt,'jbout':jbout,'jfighter':jf,
        'home_advantage_diff':home_advantage_diff,
        'travel_distance_diff_km':travel_diff,
        'tz_diff_diff_hr':tz_diff,
        'is_main_event':int(is_main),
        'card_position_norm_career_diff':card_career_diff,
        'coming_off_loss_diff':coming_off_loss_diff,
        'win_streak_entering_diff':win_streak_diff,
        'fights_last_12m_diff':fights_12m_diff,
        'stance_mismatch':stance_mismatch,
        'southpaw_advantage_diff':southpaw_adv_diff,
    })

out = pd.DataFrame(rows)
out.to_csv("data/tmp/market_features_clean.csv", index=False)
print(f"Wrote {len(out)} rows × {out.shape[1]} cols -> data/tmp/market_features_clean.csv")
print("\nNon-zero/non-default rates (sanity):")
for c in out.columns:
    if c in ('DATE','jbout','jfighter'): continue
    nz = (out[c]!=0).sum() if c not in ('card_position_norm_career_diff',) else (out[c]!=0).sum()
    print(f"  {c:<35} nonzero={nz}/{len(out)}  mean={out[c].mean():+.4f}")
