# props_api.py
import os, sys, re, json, time
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import requests
import pandas as pd
import numpy as np

# ------------------ Config ------------------
import os
API_KEY = os.getenv('ODDS_API_KEY')
SPORT = "mma_mixed_martial_arts"
REGIONS = "us,uk,eu"          # widen to improve hit rate
MARKETS = "h2h" # add more later if you want
ODDS_FORMAT = "american"
DATE_FORMAT = "iso"

MAIN_BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "bet365", "bovada"]
FUZZY_THRESHOLD = 0.8
LOOKBACK_DAYS = 30         # how far back to scan your CSV
HTTP_TIMEOUT = 12
RETRIES = 3
SLEEP_BETWEEN = 0.5

HIST_URL = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds"

# ------------------ Helpers ------------------
def _normalize(s: str) -> str:
    return re.sub(r"\W+", "", (s or "").lower())

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _get_json(url, params, label):
    for i in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                print(f"[{label}] HTTP {r.status_code}")
                print(r.text[:800])
                time.sleep(SLEEP_BETWEEN)
                continue
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"[{label}] {type(e).__name__}: {e}")
            time.sleep(SLEEP_BETWEEN)
    print(f"[{label}] failed after {RETRIES} tries")
    return None

def _fetch_snapshot(ts_iso: str):
    """Historical odds snapshot for method_of_victory at a given timestamp."""
    payload = _get_json(
        HIST_URL,
        {
            "apiKey": API_KEY,
            "regions": REGIONS,
            "markets": MARKETS,
            "oddsFormat": ODDS_FORMAT,
            "dateFormat": DATE_FORMAT,
            "date": ts_iso,
        },
        f"historical_odds@{ts_iso}",
    )
    if payload is None:
        return []
    # Historical odds returns an object with data list
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    # Some plans may return a bare list
    if isinstance(payload, list):
        return payload
    return []

def _parse_mov_for_fighter(market, fighter_norm):
    """
    Extract KO, SUB, DEC prices for fighter from one MOV market dict.
    Outcome examples:
      "Fighter Name by KO/TKO/DQ"
      "Fighter Name by Decision"
      "Fighter Name by Submission"
    """
    ko = sub = dec = None
    for o in market.get("outcomes", []) or []:
        name = str(o.get("name", ""))
        price = o.get("price")
        if price is None:
            continue
        nrm = _normalize(name)

        # require fighter mention
        if not (nrm.startswith(fighter_norm) or fighter_norm in nrm):
            continue

        if any(k in nrm for k in ["ko", "tko", "dq"]):
            ko = price if ko is None else ko
        elif "sub" in nrm or "submission" in nrm:
            sub = price if sub is None else sub
        elif any(k in nrm for k in ["dec", "decision", "points"]):
            dec = price if dec is None else dec
    return ko, sub, dec

def _find_best_event(fn_norm, on_norm, row_date, ev_list):
    """Pick the best event match within plus or minus 1 day of row_date."""
    candidates = []
    for ev in ev_list:
        ev_date = ev["commence_dt"].date()
        if abs((ev_date - row_date).days) > 1:
            continue

        home, away = ev["home"], ev["away"]

        if {fn_norm, on_norm} == {home, away}:
            return ev

        sim1 = _similar(fn_norm, home) + _similar(on_norm, away)
        sim2 = _similar(fn_norm, away) + _similar(on_norm, home)
        best = max(sim1, sim2) / 2.0
        if best >= FUZZY_THRESHOLD:
            candidates.append((best, ev))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    return None

def _clamp_extreme(df):
    cols = [c for c in df.columns if c.endswith("_ko") or c.endswith("_sub") or c.endswith("_dec")]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] > 2000, col] = np.nan
        df.loc[df[col] < -5000, col] = np.nan
    return df

def _default_paths():
    here = os.getcwd()
    candidates = [
        os.path.join(here, "final.csv"),
        os.path.join(here, "../data/tmp/final.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            in_path = os.path.abspath(p)
            out_path = os.path.join(os.path.dirname(in_path), "final_with_mov_props.csv")
            return in_path, out_path
    print("Could not find final.csv in cwd or ../data/tmp/. Provide it next to this script.")
    sys.exit(1)

# ------------------ Main scrape ------------------
def scrape_and_filter_props(input_csv_path, output_csv_path):
    if API_KEY == "YOUR_API_KEY":
        print("Set ODDS_API_KEY in your shell or edit API_KEY in this file.")
        sys.exit(1)

    print("=== MOV props scrape ===")
    print(f"Input:  {input_csv_path}")
    print(f"Output: {output_csv_path}")

    df = pd.read_csv(input_csv_path, parse_dates=["DATE"])
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True)

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=LOOKBACK_DAYS)
    df = df[df["DATE"] >= cutoff].copy()

    # prepare names for matching
    # adjust column name below if your dataframe uses a different opponent column
    opp_col = "opp_FIGHTER" if "opp_FIGHTER" in df.columns else "OPPONENT"
    if opp_col not in df.columns:
        raise ValueError("Could not find opponent column. Expected opp_FIGHTER or OPPONENT.")

    df["f_norm"] = df["FIGHTER"].apply(_normalize)
    df["o_norm"] = df[opp_col].apply(_normalize)
    df["date_str"] = df["DATE"].dt.strftime("%Y-%m-%d")

    print(f"Rows after filter: {len(df)} from {df['DATE'].min().date()} to {df['DATE'].max().date()}")

    # fetch historical snapshots per day at midnight UTC for day and day+1
    print("\nFetching historical snapshots for method_of_victory")
    raw = []
    uniq_days = sorted(df["date_str"].unique())
    for d in uniq_days:
        base = datetime.fromisoformat(d)
        for delta in (0, 1):
            ts = (base + timedelta(days=delta)).strftime("%Y-%m-%dT00:00:00Z")
            print(f"  snapshot {ts}")
            data = _fetch_snapshot(ts)
            if data:
                raw.extend(data)

    # build event list
    seen = {}
    for e in raw:
        seen[e["id"]] = e
    ev_list = []
    for e in seen.values():
        ct = pd.to_datetime(e["commence_time"], utc=True)
        ev_list.append(
            {
                "id": e["id"],
                "commence_dt": ct,
                "home": _normalize(e.get("home_team")),
                "away": _normalize(e.get("away_team")),
                "bookmakers": e.get("bookmakers", []),
            }
        )
    print(f"Unique events with potential MOV markets: {len(ev_list)}")

    # create output columns
    for bk in MAIN_BOOKMAKERS:
        df[f"{bk}_ko"]  = np.nan
        df[f"{bk}_sub"] = np.nan
        df[f"{bk}_dec"] = np.nan

    # match and extract
    print("\nMatching fights and extracting MOV prices")
    matched = 0
    for idx, row in df.iterrows():
        fn, on = row["f_norm"], row["o_norm"]
        row_date = row["DATE"].date()
        ev = _find_best_event(fn, on, row_date, ev_list)
        if not ev:
            continue
        matched += 1

        for bm in ev["bookmakers"]:
            key = bm.get("key")
            if key not in MAIN_BOOKMAKERS:
                continue
            market = next((m for m in bm.get("markets", []) if m.get("key") == "method_of_victory"), None)
            if not market:
                continue

            ko_p, sub_p, dec_p = _parse_mov_for_fighter(market, fn)
            if ko_p is not None:
                df.at[idx, f"{key}_ko"] = ko_p
            if sub_p is not None:
                df.at[idx, f"{key}_sub"] = sub_p
            if dec_p is not None:
                df.at[idx, f"{key}_dec"] = dec_p

    print(f"Matched rows: {matched}")

    # cleanup
    df.drop(columns=["f_norm", "o_norm", "date_str"], inplace=True)
    df = _clamp_extreme(df)

    # save
    df.to_csv(output_csv_path, index=False)
    print(f"Saved: {output_csv_path}")

    # quick summary
    mov_cols = [c for c in df.columns if c.endswith("_ko") or c.endswith("_sub") or c.endswith("_dec")]
    have_any = df[mov_cols].notna().any(axis=1).sum()
    print("\nSummary")
    print(f"Rows total: {len(df)}")
    print(f"Rows with any MOV price: {have_any}")
    for bk in MAIN_BOOKMAKERS:
        cnt = df[[f"{bk}_ko", f"{bk}_sub", f"{bk}_dec"]].notna().any(axis=1).sum()
        print(f"  {bk}: {cnt}")

# ------------------ Entry ------------------
if __name__ == "__main__":
    in_path, out_path = _default_paths()
    scrape_and_filter_props(in_path, out_path)
