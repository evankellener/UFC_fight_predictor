# fight_news_pipeline.py
import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from datetime import datetime, timezone
import re
import math
import spacy

nlp = spacy.load("en_core_web_sm")  # install via: python -m spacy download en_core_web_sm

SOURCE_CRED = {
    "mmajunkie": 1.0,
    "mmafighting": 1.0,
    "espn": 1.0,
    "sherdog": 0.9,
    "local": 0.6
}

INJURY_KEYWORDS = {
    "injury": 1, "injured": 1, "surgery": 3, "withdraw": 3, "illness": 2,
    "hamstring": 2, "knee": 2, "elbow": 2, "broken": 3, "torn": 3, "DVT": 3,
    "health battle": 2, "health issues": 2, "medical": 1, "sick": 1, "hospital": 2,
    "health problems": 2, "health scare": 2, "health crisis": 3
}
SHORT_NOTICE_PHRASES = [
    "short notice", "late replacement", "stepped in", "replacement", "called up", "pulled from card"
]
CAMP_POSITIVE = ["good camp", "prepared", "in great shape", "peaking", "confident", "sharp"]
CAMP_NEGATIVE = ["struggling in camp", "had trouble", "weight cut issues", "cutting issues", "missed weight", "camp issues"]

def fetch_article(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "ufo-bot/1.0"})
        r.raise_for_status()
    except Exception as e:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    # simplistic extraction
    title = soup.find("title").get_text(strip=True) if soup.find("title") else ""
    # try common article containers
    article_text = ""
    for tag in soup.find_all(["p"]):
        article_text += tag.get_text(" ", strip=True) + " "
    # attempt to find published date
    date = None
    for meta in soup.find_all("meta"):
        if meta.get("property") in ("article:published_time", "og:published_time") or meta.get("name") in ("pubdate", "publishdate", "date"):
            try:
                date = dateparser.parse(meta.get("content"))
                break
            except:
                pass
    return {"url": url, "title": title, "text": article_text.strip(), "published": date}

def score_injury(text, published, source_key="mmajunkie"):
    now = datetime.now(timezone.utc)
    days = max((now - (published or now)).days, 0)
    decay = math.exp(-0.05 * days)
    hits = 0
    score_raw = 0.0
    for k,w in INJURY_KEYWORDS.items():
        cnt = len(re.findall(r'\b' + re.escape(k) + r'\b', text, flags=re.I))
        if cnt:
            score_raw += cnt * w
            hits += cnt
    score = score_raw * decay * SOURCE_CRED.get(source_key, 0.7)
    # map to 0-10
    scaled = min(10, round(score * 1.5))
    return scaled, hits

def detect_short_notice(text, title):
    hay = title + " " + text[:600]
    for phrase in SHORT_NOTICE_PHRASES:
        if re.search(r'\b' + re.escape(phrase) + r'\b', hay, flags=re.I):
            # try to find a date near the phrase, else return detected True
            return True
    return False

def camp_signal(text, published, source_key="mmajunkie"):
    now = datetime.now(timezone.utc)
    days = max((now - (published or now)).days, 0)
    decay = math.exp(-0.05 * days)
    pos = 0
    neg = 0
    for p in CAMP_POSITIVE:
        pos += len(re.findall(re.escape(p), text, flags=re.I))
    for n in CAMP_NEGATIVE:
        neg += len(re.findall(re.escape(n), text, flags=re.I))
    # sentiment fallback: use spaCy to get polarity by simple heuristic (neg words)
    doc = nlp(text[:800])
    # naive sentiment: count neg tokens
    neg_tokens = sum(1 for tok in doc if tok.dep_ == 'neg')
    score = (pos - neg - neg_tokens) * decay * SOURCE_CRED.get(source_key, 0.7)
    # map score to 1-10 with neutral ~5
    mapped = int(max(1, min(10, 5 + round(score))))
    return mapped, pos, neg

def aggregate_for_fighter(fighter_name, urls, event_date):
    articles = []
    for u in urls:
        art = fetch_article(u)
        if art:
            articles.append(art)
    # simple aggregation
    injury_scores = []
    camp_scores = []
    short_notice_flag = 0
    short_notice_dates = []
    earliest_bout_announce = None
    for a in articles:
        pub = a["published"]
        s_inj, hits = score_injury(a["text"], pub)
        injury_scores.append((s_inj, hits, pub))
        sn = detect_short_notice(a["text"], a["title"])
        if sn:
            short_notice_flag = 1
            if pub:
                short_notice_dates.append(pub)
        cscore, pos, neg = camp_signal(a["text"], pub)
        camp_scores.append((cscore, pos, neg, pub))
        # find earliest article that mentions the matchup as announced (naive)
        if re.search(r'\b' + re.escape(fighter_name) + r'\b', a["text"], flags=re.I):
            if earliest_bout_announce is None or (pub and pub < earliest_bout_announce):
                earliest_bout_announce = pub
    # final injury risk aggregation: weighted average with recency
    if injury_scores:
        inj_vals = [v[0] for v in injury_scores]
        injury_risk = min(10, max(0, round(sum(inj_vals) / len(inj_vals))))
    else:
        injury_risk = 0
    # camp_status: average mapped
    if camp_scores:
        camp_vals = [v[0] for v in camp_scores]
        camp_status = int(round(sum(camp_vals) / len(camp_vals)))
    else:
        camp_status = 5
    # short notice duration
    if short_notice_flag and short_notice_dates:
        ann = min(short_notice_dates)
        duration = max(0, (event_date - ann.date()).days)
    elif earliest_bout_announce:
        duration = max(0, (event_date - earliest_bout_announce.date()).days)
        short_notice_flag = 0
    else:
        duration = None
    return {
        "fighter": fighter_name,
        "short_notice": int(short_notice_flag),
        "short_notice_duration_days": duration if duration is not None else 999,
        "injury_risk_0to10": injury_risk,
        "camp_status_1to10": camp_status,
        "n_articles": len(articles)
    }

# Example usage:
if __name__ == "__main__":
    # example urls - replace with your discovery results
    urls = [
        "https://bloodyelbow.com/2025/09/25/dominick-reyes-was-told-by-doctors-if-you-wake-up-today-then-good-on-you-during-health-battle/?utm_source=chatgpt.com",
        # add real urls discovered via feedparser/serpapi
    ]
    event_date = datetime(2025, 9, 27).date()
    print(aggregate_for_fighter("Dominick Reyes", urls, event_date))
