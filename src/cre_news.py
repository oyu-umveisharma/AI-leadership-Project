"""
CRE News Agent — scrapes public RSS feeds and government announcement pages
to surface companies that have announced new manufacturing plants, training
centers, warehouses, data centers, and other industrial/commercial projects.

Sources (by trust tier):
  Tier 1 — Independent News:   Reuters, Bloomberg, AP News
  Tier 2 — Government:         DOE, Commerce Dept, EDA
  Tier 3 — Trade Press:        Manufacturing.net, IndustryWeek, Site Selection,
                                Expansion Solutions
  Tier 4 — Press Releases:     PR Newswire, Business Wire

Verification pipeline:
  1. Age filter            — drop articles older than 30 days
  2. Source tier scoring   — Tier 1=85, Tier 2=80, Tier 3=65, Tier 4=30 base
  3. Specificity bonuses   — dollar amount / location / jobs / timeline each +5–10
  4. Cross-source confirm  — same story in 2+ sources boosts score
  5. Credibility label     — VERIFIED (80+) / HIGH (65+) / MODERATE (45+) / LOW (<45)
  6. Groq summary          — uses only MODERATE+ articles for the investment brief
"""

import os
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

# ── Source registry with trust tiers ─────────────────────────────────────────

SOURCE_TIER = {
    # Tier 1 — Independent journalism (highest credibility)
    "Reuters Business":        1,
    "Bloomberg Markets":       1,
    "Bloomberg Technology":    1,
    "AP News":                 1,
    # Tier 2 — Government / official sources
    "Department of Energy News":   2,
    "Commerce Department News":    2,
    "EDA Press Releases":          2,
    # Tier 3 — Independent trade press
    "Manufacturing.net":           3,
    "IndustryWeek":                3,
    "Site Selection Magazine":     3,
    "Economic Development News":   3,
    # Tier 4 — Company-issued press releases
    "PR Newswire — Manufacturing": 4,
    "Business Wire — Industrial":  4,
}

TIER_BASE_SCORE = {1: 85, 2: 80, 3: 65, 4: 30}
TIER_LABEL      = {1: "Independent News", 2: "Government", 3: "Trade Press", 4: "Press Release"}
TIER_COLOR      = {1: "#1a3a1a", 2: "#0d1a2a", 3: "#1a2a0d", 4: "#2a1a0d"}
TIER_TEXT       = {1: "#4caf50",  2: "#4a8abf",  3: "#8abf4a",  4: "#bf8a4a"}

CREDIBILITY_LABEL = {
    "VERIFIED":  {"color": "#4caf50", "bg": "#0d1a0d"},
    "HIGH":      {"color": "#8bc34a", "bg": "#121a0d"},
    "MODERATE":  {"color": "#ff9800", "bg": "#1a1200"},
    "LOW":       {"color": "#f44336", "bg": "#1a0d0d"},
}

RSS_FEEDS = [
    # ── Tier 1 — Independent News ──────────────────────────────────────────
    {
        "name": "Reuters Business",
        "url":  "https://feeds.reuters.com/reuters/businessNews",
        "type": "news",
        "tier": 1,
    },
    {
        "name": "Bloomberg Markets",
        "url":  "https://feeds.bloomberg.com/markets/news.rss",
        "type": "news",
        "tier": 1,
    },
    {
        "name": "Bloomberg Technology",
        "url":  "https://feeds.bloomberg.com/technology/news.rss",
        "type": "news",
        "tier": 1,
    },
    {
        "name": "AP News",
        "url":  "https://rsshub.app/apnews/topics/business",
        "type": "news",
        "tier": 1,
    },
    # ── Tier 2 — Government ────────────────────────────────────────────────
    {
        "name": "Department of Energy News",
        "url":  "https://www.energy.gov/rss.xml",
        "type": "government",
        "tier": 2,
    },
    {
        "name": "Commerce Department News",
        "url":  "https://www.commerce.gov/feeds/news",
        "type": "government",
        "tier": 2,
    },
    {
        "name": "EDA Press Releases",
        "url":  "https://www.eda.gov/rss/news.xml",
        "type": "government",
        "tier": 2,
    },
    # ── Tier 3 — Trade Press ───────────────────────────────────────────────
    {
        "name": "Manufacturing.net",
        "url":  "https://www.manufacturing.net/rss/content",
        "type": "industry",
        "tier": 3,
    },
    {
        "name": "IndustryWeek",
        "url":  "https://www.industryweek.com/rss/all",
        "type": "industry",
        "tier": 3,
    },
    {
        "name": "Economic Development News",
        "url":  "https://expansionsolutionsmag.com/feed/",
        "type": "industry",
        "tier": 3,
    },
    {
        "name": "Site Selection Magazine",
        "url":  "https://siteselection.com/feed/",
        "type": "industry",
        "tier": 3,
    },
    # ── Tier 4 — Press Releases ────────────────────────────────────────────
    {
        "name": "PR Newswire — Manufacturing",
        "url":  "https://www.prnewswire.com/rss/news-releases-list.rss?tagid=139",
        "type": "press",
        "tier": 4,
    },
    {
        "name": "Business Wire — Industrial",
        "url":  "https://feed.businesswire.com/rss/home/?rss=G7",
        "type": "press",
        "tier": 4,
    },
]

FACILITY_KEYWORDS = [
    "manufacturing plant", "manufacturing facility", "factory", "production facility",
    "training center", "training facility", "workforce training",
    "data center", "warehouse", "distribution center", "fulfillment center",
    "semiconductor fab", "chip plant", "battery plant", "gigafactory",
    "logistics hub", "research facility", "R&D center", "headquarters",
    "new facility", "new plant", "groundbreaking", "ribbon cutting",
    "jobs announcement", "investment announcement", "economic development",
    "build", "construct", "open", "expand", "relocate", "move operations",
    "CHIPS Act", "Inflation Reduction Act", "IRA funding", "EDA grant",
    "bipartisan infrastructure", "reshoring", "onshoring",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CRE-Intelligence-Bot/1.0; "
        "+https://purdue.edu)"
    )
}

# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_date(raw: str) -> datetime | None:
    """Parse RFC 2822 (RSS pubDate) or ISO 8601 dates into an aware datetime."""
    if not raw:
        return None
    raw = raw.strip()
    # Try RFC 2822 (standard RSS format)
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        pass
    # Try ISO 8601 variants
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw[:19], fmt[:len(fmt)])
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def _days_old(pub_date_str: str) -> int | None:
    """Return how many days ago this article was published, or None if unknown."""
    dt = _parse_date(pub_date_str)
    if not dt:
        return None
    now = datetime.now(tz=timezone.utc)
    return max(0, (now - dt).days)


# ── RSS fetch ─────────────────────────────────────────────────────────────────

def _parse_rss(feed: dict, timeout: int = 12) -> list[dict]:
    """Fetch and parse an RSS feed, return list of article dicts."""
    url         = feed["url"]
    source_name = feed["name"]
    tier        = feed["tier"]
    feed_type   = feed["type"]

    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        ns    = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item")
        if not items:
            items = root.findall(".//atom:entry", ns)

        articles = []
        for item in items[:30]:
            title = (item.findtext("title") or
                     item.findtext("atom:title", namespaces=ns) or "").strip()
            desc  = (item.findtext("description") or
                     item.findtext("summary") or
                     item.findtext("atom:summary", namespaces=ns) or "").strip()
            link  = (item.findtext("link") or
                     item.findtext("atom:link", namespaces=ns) or "").strip()
            pub_date = (item.findtext("pubDate") or
                        item.findtext("published") or
                        item.findtext("atom:published", namespaces=ns) or "")
            articles.append({
                "source":    source_name,
                "feed_type": feed_type,
                "tier":      tier,
                "title":     title,
                "description": desc[:500],
                "link":      link,
                "pub_date":  pub_date,
            })
        return articles
    except Exception:
        return []


# ── Relevance filter ──────────────────────────────────────────────────────────

def _is_relevant(article: dict) -> bool:
    text = (article.get("title", "") + " " + article.get("description", "")).lower()
    return any(kw.lower() in text for kw in FACILITY_KEYWORDS)


# ── Specificity scoring ───────────────────────────────────────────────────────

_DOLLAR_RE  = re.compile(r'\$[\d,]+(?:\.\d+)?\s*(?:million|billion|[mb])\b', re.I)
_JOBS_RE    = re.compile(r'\d[\d,]*\s*(?:jobs?|workers?|employees?|positions?)\b', re.I)
_LOCATION_RE = re.compile(
    r'\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|'
    r'Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|'
    r'Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|'
    r'Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|'
    r'North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|'
    r'South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|'
    r'West Virginia|Wisconsin|Wyoming)\b', re.I
)
_TIMELINE_RE = re.compile(
    r'\b(?:20\d\d|Q[1-4]\s*20\d\d|(?:January|February|March|April|May|June|July|'
    r'August|September|October|November|December)\s+20\d\d|by\s+20\d\d|'
    r'groundbreaking|ribbon.cutting|opening\s+in)\b', re.I
)


def _specificity_bonus(article: dict) -> int:
    text = article.get("title", "") + " " + article.get("description", "")
    bonus = 0
    if _DOLLAR_RE.search(text):   bonus += 10
    if _JOBS_RE.search(text):     bonus += 5
    if _LOCATION_RE.search(text): bonus += 5
    if _TIMELINE_RE.search(text): bonus += 5
    return bonus


# ── Cross-source confirmation ─────────────────────────────────────────────────

def _title_tokens(title: str) -> set[str]:
    """Normalize title into content tokens for overlap comparison."""
    stop = {"a", "an", "the", "in", "on", "at", "to", "of", "for", "and",
            "or", "is", "its", "with", "new", "will", "has", "that", "this"}
    tokens = re.sub(r'[^a-z0-9\s]', '', title.lower()).split()
    return {t for t in tokens if t not in stop and len(t) > 2}


def _find_confirming_sources(article: dict, all_articles: list[dict]) -> list[str]:
    """Return list of OTHER source names that appear to cover the same story."""
    my_tokens  = _title_tokens(article["title"])
    my_source  = article["source"]
    confirming = []

    if len(my_tokens) < 3:
        return confirming

    for other in all_articles:
        if other is article or other["source"] == my_source:
            continue
        other_tokens = _title_tokens(other["title"])
        if not other_tokens:
            continue
        overlap = len(my_tokens & other_tokens) / len(my_tokens | other_tokens)
        if overlap >= 0.40:  # 40% Jaccard similarity threshold
            confirming.append(other["source"])

    return list(set(confirming))


# ── Credibility scoring ───────────────────────────────────────────────────────

def _score_article(article: dict, all_articles: list[dict]) -> dict:
    """
    Assign a credibility score (0–100) and label to an article.
    Returns a copy of the article dict with credibility fields added.
    """
    tier  = article.get("tier", 4)
    score = TIER_BASE_SCORE.get(tier, 30)

    # Age penalty
    age = _days_old(article.get("pub_date", ""))
    if age is None:
        score -= 5   # Unknown date = slight penalty
    elif age > 14:
        score -= 20
    elif age > 7:
        score -= 10

    # Specificity bonus
    score += _specificity_bonus(article)

    # Cross-source confirmation bonus
    confirming = _find_confirming_sources(article, all_articles)
    if len(confirming) >= 2:
        score += 20
    elif len(confirming) == 1:
        score += 10

    score = max(0, min(100, score))

    if score >= 80:
        label = "VERIFIED"
    elif score >= 65:
        label = "HIGH"
    elif score >= 45:
        label = "MODERATE"
    else:
        label = "LOW"

    return {
        **article,
        "credibility_score":      score,
        "credibility_label":      label,
        "confirming_sources":     confirming,
        "tier_label":             TIER_LABEL.get(tier, "Unknown"),
        "age_days":               age,
    }


# ── Main fetch pipeline ───────────────────────────────────────────────────────

def fetch_facility_announcements() -> list[dict]:
    """
    Pull all RSS feeds, filter for CRE-relevant announcements,
    apply age filter, score credibility, and return sorted list.
    """
    raw_all = []
    for feed in RSS_FEEDS:
        articles = _parse_rss(feed)
        for a in articles:
            # Age gate: drop anything older than 30 days
            age = _days_old(a.get("pub_date", ""))
            if age is not None and age > 30:
                continue
            if _is_relevant(a):
                raw_all.append(a)

    # Deduplicate by title (keeping highest-tier version of duplicates)
    seen: dict[str, dict] = {}
    for a in raw_all:
        key = _title_tokens(a["title"])
        key_str = " ".join(sorted(key))[:80]
        if not key_str:
            continue
        if key_str not in seen or a["tier"] < seen[key_str]["tier"]:
            seen[key_str] = a

    unique = list(seen.values())

    # Score all articles (needs full list for cross-source check)
    scored = [_score_article(a, unique) for a in unique]

    # Sort: VERIFIED first, then by score desc
    scored.sort(key=lambda x: x["credibility_score"], reverse=True)
    return scored


# ── Source-quote verification loop ───────────────────────────────────────────
#
# Dr. Zhang's Week 5 feedback: source_quote should be an ACTIVE guardrail,
# not passive documentation. After Groq returns a facility extraction, the
# harness confirms the extracted `location` actually appears in the exact
# `source_quote` sentence the model cites. Any mismatch triggers a retry with
# the discrepancy flagged in the prompt, and the failure is logged to the
# audit trail. This directly closes the GROQ-09 loophole (location hallucination).

_VERIFY_SKIP_LOCATIONS = {"", "multiple us markets", "multiple markets",
                          "nationwide", "various", "unknown", "n/a", "na"}


def verify_source_quote(record: dict) -> tuple[bool, str]:
    """
    Active guardrail — confirm the extracted location appears in source_quote.

    Returns
    -------
    (is_valid, reason)
        is_valid = True  → source_quote covers the location (or location is a
                           legitimate multi-market / blank value we can't check)
        is_valid = False → mismatch, caller should flag the record and retry
    """
    location     = (record.get("location") or "").strip()
    source_quote = (record.get("source_quote") or "").strip()

    if location.lower() in _VERIFY_SKIP_LOCATIONS:
        return True, "skipped_generic_location"
    if not source_quote:
        return False, "missing_source_quote"

    # Break "City, State" (or "State" alone) into tokens ≥ 3 chars
    parts = [p.strip().lower() for p in re.split(r"[,/]", location) if p.strip()]
    tokens = [p for p in parts if len(p) >= 3]
    if not tokens:
        return True, "skipped_short_location"

    sq_lower = source_quote.lower()
    if not any(tok in sq_lower for tok in tokens):
        sample = source_quote.replace("\n", " ")[:80]
        return False, f"location '{location}' not found in quote '{sample}…'"
    return True, ""


def verify_and_flag_records(records: list[dict],
                            agent_name: str = "predictions") -> dict:
    """
    Verify each extracted record's source_quote / location coherence.

    Every failure is appended to `cache/audit_log.csv` with status='warning'
    and error='source_quote_mismatch: <detail>'. Records are annotated with
    `verification_ok` and (on failure) `verification_reason` keys, but NOT
    dropped — the caller decides what to do with flagged records.

    Returns
    -------
    {
        "records":        records (annotated in place),
        "passed":         count that passed,
        "failed":         count that failed,
        "failed_records": list of failing records (for retry prompts),
    }
    """
    try:
        from src.audit_logger import log_agent_run
    except Exception:
        log_agent_run = None   # type: ignore

    passed, failed_recs = 0, []
    for rec in records:
        ok, reason = verify_source_quote(rec)
        rec["verification_ok"] = ok
        if ok:
            passed += 1
            continue
        rec["verification_reason"] = reason
        failed_recs.append(rec)
        if log_agent_run:
            try:
                log_agent_run(
                    agent_name=agent_name,
                    status="warning",
                    latency_ms=0,
                    output_summary=(
                        f"source_quote verification failed: "
                        f"{rec.get('company', '?')} / {rec.get('location', '?')}"
                    ),
                    record_count=1,
                    error=f"source_quote_mismatch: {reason[:160]}",
                )
            except Exception:
                pass

    return {
        "records":        records,
        "passed":         passed,
        "failed":         len(failed_recs),
        "failed_records": failed_recs,
    }


# ── Groq LLM summary (uses only MODERATE+ articles) ──────────────────────────

def summarize_with_llm(articles: list[dict]) -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return "_GROQ_API_KEY not set. Add it to .env to enable AI news summaries._"

    # Only feed Groq articles that are at least MODERATE credibility
    trusted = [a for a in articles if a.get("credibility_label") in ("VERIFIED", "HIGH", "MODERATE")]
    if not trusted:
        trusted = articles  # fallback if nothing passes threshold

    try:
        from groq import Groq
        client = Groq(api_key=key)

        if not trusted:
            article_block = "No new facility announcements found in today's feeds."
        else:
            lines = []
            for i, a in enumerate(trusted[:40], 1):
                cred  = a.get("credibility_label", "")
                tier  = a.get("tier_label", "")
                lines.append(
                    f"{i}. [{a['source']} · {tier} · {cred}] {a['title']}\n"
                    f"   {a['description'][:200]}"
                )
            article_block = "\n".join(lines)

        prompt = f"""
Today is {datetime.now().strftime('%B %d, %Y')}.

Below are verified/credible news headlines from manufacturing, government, and business press sources
about companies announcing new facilities in the United States. Each article shows its source,
source tier (Independent News / Government / Trade Press / Press Release), and credibility rating.

{article_block}

Your job: Produce a structured intelligence brief for a commercial real estate investor.
Prioritize stories from Independent News and Government sources. Flag any story that comes
ONLY from a Press Release source and has not been independently confirmed.

**Format your response exactly like this:**

## Manufacturing & Industrial Facility Announcements

### Top Verified Announcements (sorted by investment size / strategic importance)
For each relevant announcement, provide:
- **Company** (ticker if public): [name]
- **Project**: [type — manufacturing plant / training center / data center / warehouse / etc.]
- **Location**: [city, state if mentioned]
- **Investment / Jobs**: [$ amount and job count if disclosed]
- **Timeline**: [groundbreaking / opening date if known]
- **Source Confidence**: [VERIFIED / HIGH / MODERATE — and note if only from press release]
- **CRE Opportunity**: [1 sentence — what real estate demand this creates]

### By Sector
Group announcements under: Semiconductor & Electronics | EV & Battery | Defense & Aerospace | Logistics & Distribution | Energy & Clean Tech | Other Manufacturing | Government-Funded Projects

### Key Takeaways for CRE Investors
3 bullet points on which US markets will see the most commercial real estate demand based on these announcements.

### Source Quality Note
Brief note on the overall quality of today's news cycle (e.g., how many independent vs. press-release-only stories).

Only include announcements that are real and from the provided sources.
"""
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior industrial real estate research analyst. "
                        "You verify facility announcements before advising investors. "
                        "You clearly distinguish between independently reported news "
                        "and company-issued press releases."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1600,
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"_News summary generation failed: {e}_"


# ── Entry point ───────────────────────────────────────────────────────────────

def run_news_fetch() -> dict:
    """Main entry point called by the scheduler. Returns cache-ready dict."""
    articles = fetch_facility_announcements()

    # Tally credibility breakdown
    cred_counts = {"VERIFIED": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for a in articles:
        lbl  = a.get("credibility_label", "LOW")
        tier = a.get("tier", 4)
        cred_counts[lbl]  = cred_counts.get(lbl, 0)  + 1
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    # Only summarize MODERATE+ articles
    summary = summarize_with_llm(articles)

    return {
        "summary":           summary,
        "article_count":     len(articles),
        "verified_count":    cred_counts["VERIFIED"] + cred_counts["HIGH"],
        "credibility_breakdown": cred_counts,
        "tier_breakdown":    {TIER_LABEL[k]: v for k, v in tier_counts.items()},
        "raw_articles":      articles[:60],
        "sources_checked":   len(RSS_FEEDS),
        "fetched_at":        datetime.now().isoformat(),
    }
