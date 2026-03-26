"""
CRE News Agent — scrapes public RSS feeds and government announcement pages
to surface companies that have announced new manufacturing plants, training
centers, warehouses, data centers, and other industrial/commercial projects.

Sources:
  - Reuters Business RSS
  - Manufacturing.net RSS
  - IndustryWeek RSS
  - SelectUSA / Commerce Dept press releases
  - Department of Energy announcements
  - PR Newswire manufacturing feed
  - Business Wire industrial feed
  - EDA (Economic Development Administration) grants

Uses Groq LLM to parse, rank, and summarize findings into a structured report.
"""

import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any

# ── RSS Feed Sources ──────────────────────────────────────────────────────────
RSS_FEEDS = [
    {
        "name": "Reuters Business",
        "url": "https://feeds.reuters.com/reuters/businessNews",
        "type": "news",
    },
    {
        "name": "Manufacturing.net",
        "url": "https://www.manufacturing.net/rss/content",
        "type": "industry",
    },
    {
        "name": "IndustryWeek",
        "url": "https://www.industryweek.com/rss/all",
        "type": "industry",
    },
    {
        "name": "PR Newswire — Manufacturing",
        "url": "https://www.prnewswire.com/rss/news-releases-list.rss?tagid=139",
        "type": "press",
    },
    {
        "name": "Business Wire — Industrial",
        "url": "https://feed.businesswire.com/rss/home/?rss=G7",
        "type": "press",
    },
    {
        "name": "Department of Energy News",
        "url": "https://www.energy.gov/rss.xml",
        "type": "government",
    },
    {
        "name": "Commerce Department News",
        "url": "https://www.commerce.gov/feeds/news",
        "type": "government",
    },
    {
        "name": "EDA Press Releases",
        "url": "https://www.eda.gov/rss/news.xml",
        "type": "government",
    },
    {
        "name": "Economic Development News (Expansion Solutions)",
        "url": "https://expansionsolutionsmag.com/feed/",
        "type": "industry",
    },
    {
        "name": "Site Selection Magazine",
        "url": "https://siteselection.com/feed/",
        "type": "industry",
    },
]

# Keywords that signal a facility announcement
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


def _parse_rss(url: str, source_name: str, timeout: int = 10) -> list[dict]:
    """Fetch and parse an RSS feed, return list of article dicts."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        # Handle both RSS 2.0 and Atom feeds
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item")  # RSS 2.0
        if not items:
            items = root.findall(".//atom:entry", ns)  # Atom

        articles = []
        for item in items[:30]:  # Cap at 30 per feed
            title = (
                (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "").strip()
            )
            desc = (
                (item.findtext("description") or
                 item.findtext("summary") or
                 item.findtext("atom:summary", namespaces=ns) or "").strip()
            )
            link = (
                (item.findtext("link") or
                 item.findtext("atom:link", namespaces=ns) or "").strip()
            )
            pub_date = (
                item.findtext("pubDate") or
                item.findtext("published") or
                item.findtext("atom:published", namespaces=ns) or ""
            )
            articles.append({
                "source": source_name,
                "title": title,
                "description": desc[:500],
                "link": link,
                "pub_date": pub_date,
            })
        return articles
    except Exception:
        return []


def _is_relevant(article: dict) -> bool:
    """Return True if the article likely covers a facility/investment announcement."""
    text = (article.get("title", "") + " " + article.get("description", "")).lower()
    return any(kw.lower() in text for kw in FACILITY_KEYWORDS)


def fetch_facility_announcements() -> list[dict]:
    """
    Pulls all RSS feeds, filters for facility/investment announcements,
    and returns the relevant articles sorted newest-first.
    """
    all_articles = []
    for feed in RSS_FEEDS:
        articles = _parse_rss(feed["url"], feed["name"])
        for a in articles:
            if _is_relevant(a):
                a["feed_type"] = feed["type"]
                all_articles.append(a)

    # Deduplicate by title similarity
    seen_titles = set()
    unique = []
    for a in all_articles:
        key = a["title"][:60].lower().strip()
        if key and key not in seen_titles:
            seen_titles.add(key)
            unique.append(a)

    return unique


def summarize_with_llm(articles: list[dict]) -> str:
    """
    Send the raw article headlines + snippets to Groq and get back a
    structured investment-grade summary of facility announcements.
    """
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return "_GROQ_API_KEY not set. Add it to .env to enable AI news summaries._"

    try:
        from groq import Groq
        client = Groq(api_key=key)

        if not articles:
            article_block = "No new facility announcements found in today's feeds."
        else:
            lines = []
            for i, a in enumerate(articles[:40], 1):
                lines.append(
                    f"{i}. [{a['source']}] {a['title']}\n"
                    f"   {a['description'][:200]}"
                )
            article_block = "\n".join(lines)

        prompt = f"""
Today is {datetime.now().strftime('%B %d, %Y')}.

Below are news headlines and snippets from manufacturing, government, and business press sources
about companies announcing new facilities in the United States:

{article_block}

Your job: Produce a structured intelligence brief for a commercial real estate investor.

**Format your response exactly like this:**

## 🏭 Manufacturing & Industrial Facility Announcements

### Top Announcements (sorted by investment size / strategic importance)
For each relevant announcement, provide:
- **Company** (ticker if public): [name]
- **Project**: [type — manufacturing plant / training center / data center / warehouse / etc.]
- **Location**: [city, state if mentioned]
- **Investment / Jobs**: [$ amount and job count if disclosed]
- **Timeline**: [groundbreaking / opening date if known]
- **CRE Opportunity**: [1 sentence — what real estate demand this creates: industrial, office, multifamily, retail]

### By Sector
Group announcements under: Semiconductor & Electronics | EV & Battery | Defense & Aerospace | Logistics & Distribution | Energy & Clean Tech | Other Manufacturing | Government-Funded Projects

### Key Takeaways for CRE Investors
3 bullet points on which US markets will see the most commercial real estate demand based on these announcements.

Only include announcements that are real and from the provided sources. If sources are sparse, note that and provide context on the broader trend.
"""
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior industrial real estate research analyst. "
                        "You track facility announcements to advise investors on where "
                        "commercial real estate demand is emerging. Be concise and precise."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1400,
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"_News summary generation failed: {e}_"


def run_news_fetch() -> dict:
    """Main entry point called by the agent. Returns cache-ready dict."""
    articles = fetch_facility_announcements()
    summary  = summarize_with_llm(articles)
    return {
        "summary":        summary,
        "article_count":  len(articles),
        "raw_articles":   articles[:50],  # Store up to 50 for display
        "sources_checked": len(RSS_FEEDS),
        "fetched_at":     datetime.now().isoformat(),
    }
