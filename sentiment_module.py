"""
sentiment_module.py
===================
News sentiment module for the stock prediction web app.

News sources:
  1. Alpha Vantage News Sentiment  - financial news specific to the ticker
  2. GDELT Doc API                 - general news with macroeconomic impact
  3. yfinance                      - fallback if other sources fail

Sentiment model: FinBERT (ProsusAI/finbert)

Logic:
  - Does NOT train models
  - Does NOT modify historical datasets
  - Acts AFTER obtaining the ML model signal
  - Adjusts the signal based on current news context

Installation:
    pip install transformers torch yfinance requests beautifulsoup4

Usage:
    from sentiment_module import get_sentiment_signal
    result = get_sentiment_signal("AAPL", "BUY")
"""

import warnings
warnings.filterwarnings("ignore")

import re
from math import exp
from datetime import datetime, timezone, timedelta


# CONFIGURATION

ALPHA_VANTAGE_KEY = "TUNTTOFCPHT0VTXR"
MAX_NEWS_AV    = 200   # max news from Alpha Vantage
MAX_NEWS_GDELT = 100   # max news from GDELT
MAX_NEWS_YF    = 20    # max news from yfinance (fallback)
MAX_DAYS_OLD   = 7     # ignore news older than 7 days
MIN_RELEVANCE  = 0.7   # minimum relevance threshold
TIME_DECAY     = 0.3   # temporal decay rate (per day)


# FINBERT SINGLETON
# Load the model once and reuse it for all news items.

_finbert_pipeline = None

def _load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,
            truncation=True,
            max_length=512,
        )
    return _finbert_pipeline


# HELPERS

def _days_ago(ts) -> float:
    """Returns the number of days elapsed since ts (Unix timestamp or datetime)."""
    now = datetime.now(timezone.utc)
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        return 0.0
    return max(0.0, (now - dt).total_seconds() / 86400)


def _time_weight(days: float) -> float:
    """
    Temporal weight based on exponential decay.
    A news item from today (days=0) has weight 1.0.
    A news item from 7 days ago has weight exp(-0.3*7) = 0.12.
    """
    return exp(-TIME_DECAY * days)


def _clean_text(text: str) -> str:
    """Remove special characters and truncate to 512 chars for FinBERT."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:512]


def _relevance_score(title: str, body: str, company_name: str, ticker: str) -> float:
    """
    Calculates the relevance of a news item for the given stock [0.0, 1.0].

    Conditions:
      +1.0 if the base ticker appears in the text (e.g. 'nvda' for 'NVDA')
      +0.4 for each significant word from the company name (max +0.8)
      -0.4 if the news explicitly mentions another known ticker
    """
    combined = (title + " " + body).lower()
    score = 0.0

    # 1. Base ticker (e.g. aapl, tsla, itx)
    ticker_base = ticker.upper().split(".")[0].lower()
    if ticker_base in combined:
        score += 1.0

    # 2. Significant words from the company name (more than 3 characters)
    clean_name = re.sub(
        r'\b(inc|corp|ltd|plc|sa|nv|ag|se|co|group|holdings|holding|the)\b',
        '', company_name.lower()
    )
    words = [w.strip() for w in clean_name.split() if len(w.strip()) > 3]
    matches = sum(1 for w in words if w in combined)
    score += min(matches * 0.4, 0.8)

    # 3. Penalize if another known ticker is explicitly mentioned in the title
    other_tickers = re.findall(r'\b([A-Z]{2,5})\b', title)
    known_tickers = {
        "DELL", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "AAPL", "QCOM", "AMD", "INTC", "IBM",
        "NFLX", "PYPL", "CRM", "ADBE", "ORCL", "CSCO",
    }
    for t in other_tickers:
        if t != ticker_base.upper() and t not in company_name.upper():
            if t in known_tickers:
                score -= 0.4

    return max(0.0, min(score, 1.0))


def _analyze_sentiment(text: str):
    """
    Analyze the sentiment of a text using FinBERT.
    Returns a float between -1 and 1:
      positive -> +confidence score (e.g. +0.95)
      negative -> -confidence score (e.g. -0.97)
      neutral  ->  0.0
    Returns None if there is an error.
    """
    if not text or not text.strip():
        return None
    try:
        pipe = _load_finbert()
        result = pipe(_clean_text(text), truncation=True)[0]
        label = result["label"].lower()
        score = result["score"]
        if label == "positive":
            return score
        elif label == "negative":
            return -score
        else:
            return 0.0
    except Exception:
        return None


# SOURCE 1: ALPHA VANTAGE NEWS SENTIMENT
# API specific for financial news with ticker filtering.
# Includes a relevance_score for each mentioned ticker.

def _fetch_alpha_vantage(ticker: str, company_name: str) -> list:
    """
    Downloads financial news from Alpha Vantage.
    Filters by ticker and minimum relevance.
    Returns list of dicts with title, text, url, days_ago, relevance, source.
    """
    import requests

    ticker_base = ticker.upper().split(".")[0]
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT"
        f"&tickers={ticker_base}"
        f"&limit={MAX_NEWS_AV}"
        f"&apikey={ALPHA_VANTAGE_KEY}"
    )

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[AV] Error: {e}")
        return []

    if "feed" not in data:
        print(f"[AV] No results. Response keys: {list(data.keys())}")
        return []

    articles = []
    for item in data["feed"]:
        try:
            title   = item.get("title", "")
            summary = item.get("summary", "")
            url_art = item.get("url", "")

            # Publication date
            time_str = item.get("time_published", "")
            if time_str:
                dt = datetime.strptime(time_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                days = _days_ago(dt)
            else:
                days = 0.0

            if days > MAX_DAYS_OLD:
                continue

            # Ticker-specific relevance (Alpha Vantage calculates this per ticker)
            ticker_sentiment = None
            rel_score = 0.0
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker_base.upper():
                    try:
                        ticker_sentiment = float(ts.get("ticker_sentiment_score", 0))
                        rel_score = float(ts.get("relevance_score", 0))
                    except Exception:
                        pass
                    break

            # If Alpha Vantage does not find the ticker in the news, skip it
            if ticker_sentiment is None:
                continue

            # Combined relevance: max between AV's and ours
            our_rel = _relevance_score(title, summary, company_name, ticker)
            relevance = max(rel_score, our_rel)

            if relevance < MIN_RELEVANCE:
                continue

            articles.append({
                "title":     title,
                "text":      summary if summary else title,
                "url":       url_art,
                "days_ago":  days,
                "relevance": round(relevance, 3),
                "source":    "alphavantage",
            })

        except Exception:
            continue

    print(f"[AV] {len(articles)} relevant news obtained")
    return articles


# SOURCE 2: GDELT DOC API
# Public API without key. Updated every 15 minutes.
# Search by company name + ticker. Returns titles only.

def _fetch_gdelt(ticker: str, company_name: str) -> list:
    """
    Downloads news from GDELT via public REST API.
    Filters by company name and ticker.
    Returns list of dicts with title, text, url, days_ago, relevance, source.
    """
    import requests

    ticker_base = ticker.upper().split(".")[0]

    # Build the query with the most significant words from the name
    clean_name = re.sub(
        r'\b(inc|corp|ltd|plc|sa|nv|ag|se|co|group|holdings|holding|the)\b',
        '', company_name.lower()
    ).strip()
    words = [w for w in clean_name.split() if len(w) > 3][:2]
    query_terms = " ".join(words) if words else ticker_base

    # Date range: last MAX_DAYS_OLD days
    start_dt = datetime.now(timezone.utc) - timedelta(days=MAX_DAYS_OLD)
    start_str = start_dt.strftime("%Y%m%d%H%M%S")

    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={requests.utils.quote(query_terms)}%20{ticker_base}"
        f"&mode=artlist"
        f"&maxrecords={MAX_NEWS_GDELT}"
        f"&startdatetime={start_str}"
        f"&format=json"
        f"&sourcelang=english"
    )

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[GDELT] Error: {e}")
        return []

    articles_raw = data.get("articles", [])
    if not articles_raw:
        print(f"[GDELT] No results")
        return []

    articles = []
    for item in articles_raw:
        try:
            title   = item.get("title", "")
            url_art = item.get("url", "")

            date_str = item.get("seendate", "")
            if date_str:
                try:
                    dt = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                    days = _days_ago(dt)
                except Exception:
                    days = 0.0
            else:
                days = 0.0

            if days > MAX_DAYS_OLD:
                continue

            # GDELT does not calculate relevance per ticker, use ours
            relevance = _relevance_score(title, "", company_name, ticker)
            if relevance < MIN_RELEVANCE:
                continue

            articles.append({
                "title":     title,
                "text":      title,   # GDELT only provides title by default
                "url":       url_art,
                "days_ago":  days,
                "relevance": round(relevance, 3),
                "source":    "gdelt",
            })

        except Exception:
            continue

    print(f"[GDELT] {len(articles)} relevant news obtained")
    return articles


# SOURCE 3: YFINANCE (fallback)
# Used only if other sources return fewer than 5 news items.

def _fetch_yfinance(ticker: str, company_name: str) -> list:
    """
    Fallback: news from yfinance if other sources fail or return too few items.
    Returns list of dicts with title, text, url, days_ago, relevance, source.
    """
    try:
        import yfinance as yf
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news or []
    except Exception:
        return []

    articles = []

    for item in news[:MAX_NEWS_YF]:
        try:
            # Support for new yfinance structure (content dict)
            if isinstance(item, dict) and "content" in item:
                content = item["content"]
                title   = content.get("title", "")
                pub_str = content.get("pubDate", "")
                link    = (
                    content.get("canonicalUrl", {}).get("url", "") or
                    content.get("clickThroughUrl", {}).get("url", "")
                )
                if pub_str:
                    try:
                        dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                        days = _days_ago(dt)
                    except Exception:
                        days = 0.0
                else:
                    days = 0.0
            else:
                title  = item.get("title", "")
                link   = item.get("link", "")
                pub_ts = item.get("providerPublishTime", None)
                days   = _days_ago(pub_ts) if pub_ts else 0.0

            if not title or days > MAX_DAYS_OLD:
                continue

            relevance = _relevance_score(title, "", company_name, ticker)
            if relevance < MIN_RELEVANCE:
                continue

            articles.append({
                "title":     title,
                "text":      title,
                "url":       link,
                "days_ago":  days,
                "relevance": round(relevance, 3),
                "source":    "yfinance",
            })

        except Exception:
            continue

    print(f"[YF] {len(articles)} relevant news obtained")
    return articles


# MAIN PIPELINE

def get_sentiment_signal(ticker: str, base_signal: str = "INSUFFICIENT") -> dict:
    """
    Analyze recent news from multiple sources and adjusts the base ML signal.

    Flow:
      1. Get official company name via yfinance
      2. Download news (AV -> GDELT -> yfinance fallback)
      3. Deduplicate by title
      4. Analyze sentiment of each news item with FinBERT
      5. Weight by relevance and time
      6. Aggregate into a global score [-1, 1]
      7. Adjust the base signal conservatively

    Args:
        ticker:       stock ticker (e.g. 'AAPL', 'ITX.MC')
        base_signal:  signal from ML models ('BUY'/'DO NOT BUY'/'INSUFFICIENT')

    Returns:
        dict with: adjusted_signal, sentiment_score, sentiment_label, strength,
                   positive_ratio, negative_ratio, neutral_ratio, news_count,
                   sources_used, by_source, news
    """

    def _default():
        return {
            "adjusted_signal": base_signal,
            "sentiment_score": 0.0,
            "sentiment_label": "NEUTRAL",
            "strength":        "MILD",
            "positive_ratio":  0.0,
            "negative_ratio":  0.0,
            "neutral_ratio":   1.0,
            "news_count":      0,
            "sources_used":    [],
            "news":            [],
        }

    # Get company name
    company_name = ticker
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        company_name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        pass

    print(f"\n[SENTIMENT] Analysing news for {ticker} ({company_name})...")

    # Collect news from all sources
    all_articles = []
    sources_used = []

    av_articles = _fetch_alpha_vantage(ticker, company_name)
    if av_articles:
        all_articles.extend(av_articles)
        sources_used.append("Alpha Vantage")

    gdelt_articles = _fetch_gdelt(ticker, company_name)
    if gdelt_articles:
        all_articles.extend(gdelt_articles)
        sources_used.append("GDELT")

    # yfinance only if other sources returned fewer than 5 items
    if len(all_articles) < 5:
        yf_articles = _fetch_yfinance(ticker, company_name)
        if yf_articles:
            all_articles.extend(yf_articles)
            sources_used.append("yfinance")

    if not all_articles:
        print(f"[SENTIMENT] No news found for {ticker}")
        return _default()

    # Deduplicate by title (first 80 characters)
    seen_titles = set()
    unique_articles = []
    for art in all_articles:
        title_key = art["title"].lower().strip()[:80]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(art)

    print(f"[SENTIMENT] {len(unique_articles)} unique news from {sources_used}")

    # Analyze sentiment with FinBERT
    processed = []
    for art in unique_articles:
        try:
            sentiment = _analyze_sentiment(art["text"])
            if sentiment is None:
                continue

            tw          = _time_weight(art["days_ago"])
            final_score = sentiment * art["relevance"] * tw

            processed.append({
                "title":       art["title"],
                "sentiment":   round(sentiment, 3),
                "relevance":   art["relevance"],
                "time_weight": round(tw, 3),
                "final_score": round(final_score, 3),
                "source":      art["source"],
                "days_ago":    round(art["days_ago"], 1),
            })

        except Exception:
            continue

    if not processed:
        return _default()

    # Global weighted aggregation
    scores  = [n["final_score"] for n in processed]
    weights = [abs(n["relevance"] * n["time_weight"]) for n in processed]
    total_w = sum(weights)

    sentiment_score = (sum(scores) / total_w) if total_w > 0 else 0.0
    sentiment_score = max(-1.0, min(1.0, round(sentiment_score, 3)))

    n_total = len(processed)
    n_pos   = sum(1 for n in processed if n["sentiment"] > 0.05)
    n_neg   = sum(1 for n in processed if n["sentiment"] < -0.05)
    n_neu   = n_total - n_pos - n_neg

    pos_ratio = round(n_pos / n_total, 3)
    neg_ratio = round(n_neg / n_total, 3)
    neu_ratio = round(n_neu / n_total, 3)

    # Global label
    if sentiment_score > 0.05:
        label = "POSITIVE"
    elif sentiment_score < -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    # Intensity
    abs_score = abs(sentiment_score)
    if abs_score > 0.30:
        strength = "STRONG"
    elif abs_score > 0.15:
        strength = "MODERATE"
    else:
        strength = "MILD"

    # Statistics by source
    by_source = {}
    for n in processed:
        src = n["source"]
        if src not in by_source:
            by_source[src] = {"count": 0, "scores": []}
        by_source[src]["count"] += 1
        by_source[src]["scores"].append(n["sentiment"])
    for src in by_source:
        sc = by_source[src]["scores"]
        by_source[src]["avg_sentiment"] = round(sum(sc) / len(sc), 3)

    # Signal adjustment (conservative)
    # Logic: sentiment adjusts the base signal but does not contradict it directly
    # If sentiment is strongly contrary to the signal, add caution
    # If sentiment confirms the signal, keep it
    s = sentiment_score
    base_up = base_signal.upper()

    if "DO NOT BUY" in base_up:
        if s > 0.30:
            adjusted_signal = "DO NOT BUY (WITH CAUTION)"
        else:
            adjusted_signal = "DO NOT BUY"
    elif "BUY" in base_up:
        if s < -0.30:
            adjusted_signal = "BUY (WITH CAUTION)"
        else:
            adjusted_signal = "BUY"
    else:  # INSUFFICIENT
        if s > 0.30:
            adjusted_signal = "SLIGHTLY POSITIVE"
        elif s < -0.30:
            adjusted_signal = "SLIGHTLY NEGATIVE"
        else:
            adjusted_signal = "INSUFFICIENT"

    print(
        f"[SENTIMENT] Score:{sentiment_score:+.3f} | {label} ({strength}) | "
        f"{n_pos}+ {n_neg}- {n_neu}= | Signal: {adjusted_signal}"
    )

    return {
        "adjusted_signal":  adjusted_signal,
        "sentiment_score":  sentiment_score,
        "sentiment_label":  label,
        "strength":         strength,
        "positive_ratio":   pos_ratio,
        "negative_ratio":   neg_ratio,
        "neutral_ratio":    neu_ratio,
        "news_count":       n_total,
        "sources_used":     sources_used,
        "by_source":        by_source,
        "news":             processed,
    }


# COMMAND LINE TEST
# Usage: python sentiment_module.py AAPL BUY

if __name__ == "__main__":
    import sys

    ticker      = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    base_signal = sys.argv[2] if len(sys.argv) > 2 else "INSUFFICIENT"

    _load_finbert()
    result = get_sentiment_signal(ticker, base_signal)

    print(f"\n{'='*60}")
    print(f"  SENTIMENT - {ticker}")
    print(f"{'='*60}")
    print(f"  News analysed  : {result['news_count']}")
    print(f"  Sources used   : {', '.join(result['sources_used'])}")
    print(f"  Global score   : {result['sentiment_score']:+.3f}")
    print(f"  Sentiment      : {result['sentiment_label']} ({result['strength']})")
    print(f"  Positive       : {result['positive_ratio']*100:.0f}%")
    print(f"  Negative       : {result['negative_ratio']*100:.0f}%")
    print(f"  Neutral        : {result['neutral_ratio']*100:.0f}%")
    print(f"{'-'*60}")
    if "by_source" in result:
        print(f"  By source:")
        for src, stats in result["by_source"].items():
            print(f"    {src:<15}: {stats['count']} news | "
                  f"avg sentiment: {stats['avg_sentiment']:+.3f}")
    print(f"{'-'*60}")
    print(f"  Base signal    : {base_signal}")
    print(f"  Adjusted signal: {result['adjusted_signal']}")
    print(f"{'='*60}")

    print(f"\nTop 10 news by score:")
    sorted_news = sorted(result["news"], key=lambda x: abs(x["final_score"]), reverse=True)
    for n in sorted_news[:10]:
        if n["sentiment"] > 0.05:
            icon = "[+]"
        elif n["sentiment"] < -0.05:
            icon = "[-]"
        else:
            icon = "[ ]"
        print(f"  {icon} [{n['sentiment']:+.2f}] ({n['source']}) {n['title'][:65]}")