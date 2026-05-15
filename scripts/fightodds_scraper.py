"""
fightodds.io scraper — pulls historical UFC odds via GraphQL API.
Runs GQL calls inside Playwright browser to satisfy Cloudflare.

Outputs:
  data/tmp/fo_odds_raw.csv     — all books, open+close, all fights 2015+
  data/tmp/fo_closing.csv      — devigged closing lines (matches odds_table.csv schema)
  data/tmp/fo_opening.csv      — devigged opening lines
"""
import asyncio, json, time, csv, os
from pathlib import Path
from collections import defaultdict
from playwright.async_api import async_playwright

GQL_URL = "https://api.fightodds.io/gql"
OUT_DIR = Path(__file__).parent.parent / "data" / "tmp"
RAW_FILE = OUT_DIR / "fo_odds_raw.csv"
CLOSE_FILE = OUT_DIR / "fo_closing.csv"
OPEN_FILE = OUT_DIR / "fo_opening.csv"
CACHE_FILE = OUT_DIR / "fo_events_cache.json"
PROGRESS_FILE = OUT_DIR / "fo_scrape_progress.json"

PREFERRED_BOOKS = ["Pinnacle", "Bookmaker", "Bovada", "BetOnline", "MyBookie", "BetUS"]


# ── GQL via Playwright ────────────────────────────────────────────────────────

async def gql_fetch(page, query, variables=None):
    """Execute a GQL call from inside the browser context."""
    payload = json.dumps({"query": query, "variables": variables or {}})
    result = await page.evaluate(f"""
        async () => {{
            const r = await fetch("{GQL_URL}", {{
                method: "POST",
                headers: {{"Content-Type": "application/json",
                           "Referer": "https://fightodds.io/",
                           "Origin": "https://fightodds.io"}},
                body: {payload!r}
            }});
            return r.json();
        }}
    """)
    if result.get("errors"):
        print(f"  GQL errors: {result['errors'][:2]}")
        return None
    return result.get("data")


# ── Queries ───────────────────────────────────────────────────────────────────

ALL_EVENTS_QUERY = """
query AllUFCEvents($after: String, $first: Int) {
  allEvents(
    promotion_ShortName: "UFC"
    date_Gte: "2015-01-01"
    isCancelled: false
    first: $first
    after: $after
    orderBy: "date"
  ) {
    pageInfo { hasNextPage endCursor }
    edges { node { id name pk slug date } }
  }
}
"""

ODDS_QUERY = """
query EventOdds($pk: Int!) {
  eventOfferTable(pk: $pk) {
    slug
    date
    fightOffers {
      edges {
        node {
          fighter1 { firstName lastName }
          fighter2 { firstName lastName }
          isCancelled
          straightOffers {
            edges {
              node {
                sportsbook { shortName slug }
                outcome1 { fighter { firstName lastName } odds oddsOpen }
                outcome2 { fighter { firstName lastName } odds oddsOpen }
              }
            }
          }
        }
      }
    }
  }
}
"""


# ── Event enumeration ─────────────────────────────────────────────────────────

async def get_all_ufc_events(page):
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            events = json.load(f)
        print(f"Loaded {len(events)} events from cache ({events[0]['date']} → {events[-1]['date']})")
        return events

    print("Enumerating all UFC events...")
    events = []
    cursor = ""
    page_num = 0
    while True:
        page_num += 1
        data = await gql_fetch(page, ALL_EVENTS_QUERY, {"after": cursor, "first": 50})
        if not data:
            break
        conn = data["allEvents"]
        batch = [e["node"] for e in conn["edges"]]
        events.extend(batch)
        print(f"  Page {page_num}: +{len(batch)} events (total {len(events)})")
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]
        await asyncio.sleep(0.3)

    with open(CACHE_FILE, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Cached {len(events)} events → {CACHE_FILE}")
    return events


# ── Odds parsing ──────────────────────────────────────────────────────────────

def norm_name(first, last):
    return "".join(f"{first or ''}{last or ''}".split())


def norm_bout(f1, f2):
    a = norm_name(f1[0], f1[1])
    b = norm_name(f2[0], f2[1])
    return f"{a}vs.{b}"


def parse_event(ev_meta, offer_table):
    rows = []
    if not offer_table or not offer_table.get("fightOffers"):
        return rows

    date = ev_meta.get("date", "")
    event_name = ev_meta.get("name", "")
    slug = ev_meta.get("slug", "")

    for fe in offer_table["fightOffers"]["edges"]:
        fight = fe["node"]
        if fight.get("isCancelled"):
            continue

        f1 = fight["fighter1"]
        f2 = fight["fighter2"]
        f1_tup = (f1["firstName"] or "", f1["lastName"] or "")
        f2_tup = (f2["firstName"] or "", f2["lastName"] or "")
        bout = norm_bout(f1_tup, f2_tup)
        f1_display = " ".join(f1_tup).strip()
        f2_display = " ".join(f2_tup).strip()

        for oe in fight["straightOffers"]["edges"]:
            offer = oe["node"]
            sb = offer["sportsbook"]["shortName"]
            o1 = offer.get("outcome1") or {}
            o2 = offer.get("outcome2") or {}

            def fighter_name_from(outcome, fallback):
                f = (outcome or {}).get("fighter") or {}
                fn = (f.get("firstName") or "", f.get("lastName") or "")
                return fn if any(fn) else fallback

            fn1 = fighter_name_from(o1, f1_tup)
            fn2 = fighter_name_from(o2, f2_tup)

            for fighter_tup, outcome, is_f1 in [(fn1, o1, True), (fn2, o2, False)]:
                rows.append({
                    "date": date,
                    "event": event_name,
                    "event_slug": slug,
                    "bout": bout,
                    "fighter": norm_name(*fighter_tup),
                    "fighter_display": " ".join(fighter_tup).strip(),
                    "book": sb,
                    "close_odds": outcome.get("odds"),
                    "open_odds": outcome.get("oddsOpen"),
                })
    return rows


# ── Devigging ─────────────────────────────────────────────────────────────────

def american_to_prob(odds):
    if odds is None or odds == 0:
        return None
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else (-o) / (-o + 100.0)


def devig(p1, p2):
    if p1 is None or p2 is None:
        return None, None
    total = p1 + p2
    return (p1 / total, p2 / total) if total > 0 else (None, None)


def build_devigged_files(all_rows):
    fight_book = defaultdict(lambda: defaultdict(dict))
    for row in all_rows:
        key = (row["date"], row["bout"])
        fight_book[key][row["book"]][row["fighter"]] = row

    close_rows, open_rows = [], []

    for (date, bout), books in fight_book.items():
        for line_type in ["close", "open"]:
            odds_key = "close_odds" if line_type == "close" else "open_odds"

            valid_books = {}
            for book, fighters in books.items():
                if len(fighters) != 2:
                    continue
                names = list(fighters.keys())
                f1k, f2k = names[0], names[1]
                o1_raw = fighters[f1k].get(odds_key)
                o2_raw = fighters[f2k].get(odds_key)
                if o1_raw and o2_raw:
                    valid_books[book] = {
                        "f1_key": f1k, "f2_key": f2k,
                        "o1": int(o1_raw), "o2": int(o2_raw),
                        "f1_display": fighters[f1k]["fighter_display"],
                        "f2_display": fighters[f2k]["fighter_display"],
                        "date": fighters[f1k]["date"],
                        "event": fighters[f1k]["event"],
                    }

            if not valid_books:
                continue

            chosen = next((b for b in PREFERRED_BOOKS if b in valid_books),
                          list(valid_books.keys())[0])
            entry = valid_books[chosen]
            p1, p2 = devig(american_to_prob(entry["o1"]), american_to_prob(entry["o2"]))
            if p1 is None:
                continue

            base = {
                "DATE": entry["date"],
                "EVENT": entry["event"],
                "BOUT": f"{entry['f1_display']} vs {entry['f2_display']}",
                "book": chosen,
                "n_books": len(valid_books),
            }
            target = close_rows if line_type == "close" else open_rows
            for fighter_key, prob, odds_raw, display in [
                (entry["f1_key"], p1, entry["o1"], entry["f1_display"]),
                (entry["f2_key"], p2, entry["o2"], entry["f2_display"]),
            ]:
                target.append({**base,
                                "FIGHTER": fighter_key,
                                "FIGHTER_DISPLAY": display,
                                "prob_norm": round(prob, 6),
                                "odds": odds_raw})

    fields = ["DATE", "EVENT", "BOUT", "FIGHTER", "FIGHTER_DISPLAY",
              "prob_norm", "odds", "book", "n_books"]
    for path, rows in [(CLOSE_FILE, close_rows), (OPEN_FILE, open_rows)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  {path}: {len(rows)} rows")


# ── Main ──────────────────────────────────────────────────────────────────────

RAW_FIELDS = ["date", "event", "event_slug", "bout", "fighter", "fighter_display",
              "book", "close_odds", "open_odds"]


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ))
        page = await ctx.new_page()

        # Load fightodds.io so Cloudflare challenge is solved
        print("Loading fightodds.io to establish session...")
        await page.goto("https://fightodds.io/mma-events/999/ufc-194-aldo-vs-mcgregor/odds",
                        wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(4000)
        print("  Session established")

        print("\n=== Step 1: Enumerate all UFC events ===")
        events = await get_all_ufc_events(page)
        if not events:
            print("ERROR: no events found"); return
        print(f"Total: {len(events)} events ({events[0]['date']} → {events[-1]['date']})")

        # Load progress
        done_slugs = set()
        all_rows = []
        if RAW_FILE.exists():
            with open(RAW_FILE) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)
                    done_slugs.add(row.get("event_slug", ""))
            print(f"Resuming: {len(done_slugs)} events done, {len(all_rows)} rows loaded")

        print("\n=== Step 2: Fetch odds per event ===")
        new_count = 0
        for i, ev in enumerate(events):
            slug = ev["slug"]
            if slug in done_slugs:
                continue

            print(f"  [{i+1}/{len(events)}] {ev['date']} {ev['name']} (pk={ev['pk']})", end=" ", flush=True)
            offer_table = await gql_fetch(page, ODDS_QUERY, {"pk": ev["pk"]})
            if offer_table:
                offer_table = offer_table.get("eventOfferTable")

            rows = parse_event(ev, offer_table) if offer_table else []
            for row in rows:
                row["event_slug"] = slug
            all_rows.extend(rows)
            done_slugs.add(slug)
            new_count += 1

            fights_with_odds = len({r["bout"] for r in rows if r.get("close_odds")}) if rows else 0
            print(f"— {fights_with_odds} fights, {len(rows)} offer rows")

            # Save every 20 events
            if new_count % 20 == 0:
                _write_raw(all_rows, RAW_FIELDS)
                print(f"  [checkpoint] {len(all_rows)} rows saved")

            await asyncio.sleep(0.5)

        _write_raw(all_rows, RAW_FIELDS)
        print(f"\nRaw file: {RAW_FILE} ({len(all_rows)} rows)")

        await browser.close()

    print("\n=== Step 3: Build devigged files ===")
    build_devigged_files(all_rows)
    print("Done.")


def _write_raw(rows, fields):
    with open(RAW_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    asyncio.run(main())
