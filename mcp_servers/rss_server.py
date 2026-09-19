"""FastMCP Server providing Alphabet/Google RSS news feeds and updates.

Follows Model Context Protocol (MCP) standards to decouple external RSS
ingestion and XML parsing from agent reasoning.
"""

from typing import Any
import feedparser
from bs4 import BeautifulSoup
from fastmcp import FastMCP

mcp = FastMCP("alphabet-rss-news")

GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q=Alphabet+Google+stock+GOOGL&hl=en-US&gl=US&ceid=US:en"
)
GOOGLE_BLOG_RSS_URL = "https://blog.google/rss/"


def _clean_html(raw_html: str) -> str:
    """Strips HTML markup and truncates excessive whitespace."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


@mcp.tool()
def get_alphabet_news_headlines(max_results: int = 5) -> dict[str, Any]:
    """Retrieves the latest verified public news headlines specifically covering Alphabet and Google stock.

    Use this tool whenever you need current market sentiment, company developments,
    regulatory updates, or recent press coverage concerning Alphabet Inc. (GOOG/GOOGL).

    Args:
        max_results: Maximum number of recent news stories to return (default 5, max 10).

    Returns:
        Structured dictionary containing status and list of news stories with
        headline title, published timestamp, cleaned summary, and source link.
    """
    bounded_max = max(1, min(max_results, 10))
    try:
        feed = feedparser.parse(GOOGLE_NEWS_RSS_URL)
        if not feed.entries:
            return {
                "status": "warning",
                "message": "No RSS entries found. Verify internet connectivity.",
                "stories": [],
            }

        stories = []
        for entry in feed.entries[:bounded_max]:
            summary = _clean_html(getattr(entry, "summary", ""))
            stories.append(
                {
                    "title": getattr(entry, "title", "Untitled"),
                    "published": getattr(entry, "published", "Unknown date"),
                    "summary": summary[:300] + ("..." if len(summary) > 300 else ""),
                    "link": getattr(entry, "link", ""),
                }
            )

        return {
            "status": "success",
            "count": len(stories),
            "stories": stories,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve RSS feed: {e!s}. Suggest retrying or using cached sentiment.",
            "stories": [],
        }


@mcp.tool()
def get_alphabet_ir_updates(max_results: int = 3) -> dict[str, Any]:
    """Retrieves official Alphabet corporate announcements and product updates from the Google Blog RSS.

    Use this tool when users ask about official company announcements, executive leadership
    statements, or corporate press releases rather than third-party financial news.

    Args:
        max_results: Maximum number of corporate updates to return (default 3, max 5).

    Returns:
        Structured dictionary containing status and list of official announcements.
    """
    bounded_max = max(1, min(max_results, 5))
    try:
        feed = feedparser.parse(GOOGLE_BLOG_RSS_URL)
        if not feed.entries:
            return {
                "status": "warning",
                "message": "No official blog entries found. Please verify network access.",
                "updates": [],
            }

        updates = []
        for entry in feed.entries[:bounded_max]:
            summary = _clean_html(getattr(entry, "summary", ""))
            updates.append(
                {
                    "title": getattr(entry, "title", "Untitled"),
                    "published": getattr(entry, "published", "Unknown date"),
                    "summary": summary[:300] + ("..." if len(summary) > 300 else ""),
                    "link": getattr(entry, "link", ""),
                }
            )

        return {
            "status": "success",
            "count": len(updates),
            "updates": updates,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to fetch corporate updates: {e!s}.",
            "updates": [],
        }


if __name__ == "__main__":
    mcp.run()
