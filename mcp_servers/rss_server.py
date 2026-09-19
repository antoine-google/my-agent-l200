# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastMCP Server providing Alphabet/Google RSS news feeds and updates.

Follows Model Context Protocol (MCP) standards and ADK best practices:
1. Explicit Pydantic response models providing strict JSON schemas.
2. Decouples external RSS ingestion and XML parsing from agent reasoning.
3. Subscriptable Pydantic models ensuring backwards compatibility.
"""

from typing import Any, Literal, Optional
from bs4 import BeautifulSoup
import feedparser
from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

mcp = FastMCP("alphabet-rss-news")

GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q=Alphabet+Google+stock+GOOGL&hl=en-US&gl=US&ceid=US:en"
)
GOOGLE_BLOG_RSS_URL = "https://blog.google/rss/"


# ---------------------------------------------------------------------------
# Structured Tool Return Schemas
# ---------------------------------------------------------------------------

class StructuredToolResult(BaseModel):
    """Base model providing strict JSON schemas while preserving dict-style subscripting."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item) and getattr(self, item) is not None


class NewsStory(StructuredToolResult):
    """Schema for an individual Alphabet news story headline and summary."""
    title: str = Field(description="Headline title of the news story.")
    published: str = Field(description="Publication timestamp or date string.")
    summary: str = Field(description="Sanitized and truncated story synopsis.")
    link: str = Field(description="Direct URL link to the original news article.")


class NewsHeadlinesResponse(StructuredToolResult):
    """Structured response schema for Alphabet/Google news headlines."""
    status: Literal["success", "warning", "error"] = Field(description="Retrieval status: 'success', 'warning', or 'error'.")
    count: int = Field(default=0, description="Total number of news stories returned.")
    stories: list[NewsStory] = Field(default_factory=list, description="List of parsed news stories.")
    message: Optional[str] = Field(default=None, description="Diagnostic message or connectivity warning if applicable.")


class IRItem(StructuredToolResult):
    """Schema for an individual corporate press release or official announcement."""
    title: str = Field(description="Official corporate headline title.")
    published: str = Field(description="Publication date of the announcement.")
    summary: str = Field(description="Cleaned excerpt of the corporate release.")
    link: str = Field(description="URL to the official blog or IR statement.")


class IRUpdatesResponse(StructuredToolResult):
    """Structured response schema for official Alphabet/Google corporate announcements."""
    status: Literal["success", "warning", "error"] = Field(description="Retrieval status: 'success', 'warning', or 'error'.")
    count: int = Field(default=0, description="Total count of corporate updates returned.")
    updates: list[IRItem] = Field(default_factory=list, description="List of official corporate updates.")
    message: Optional[str] = Field(default=None, description="Diagnostic message or warning if applicable.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_html(raw_html: str) -> str:
    """Strips HTML markup and truncates excessive whitespace."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_alphabet_news_headlines(max_results: int = 5) -> NewsHeadlinesResponse:
    """Retrieves the latest verified public news headlines specifically covering Alphabet and Google stock.

    Use this tool whenever you need current market sentiment, company developments,
    regulatory updates, or recent press coverage concerning Alphabet Inc. (GOOG/GOOGL).

    Args:
        max_results: Maximum number of recent news stories to return (default 5, max 10).

    Returns:
        Structured NewsHeadlinesResponse with explicit JSON schema containing status,
        story count, and list of news stories.
    """
    bounded_max = max(1, min(max_results, 10))
    try:
        feed = feedparser.parse(GOOGLE_NEWS_RSS_URL)
        if not feed.entries:
            return NewsHeadlinesResponse(
                status="warning",
                count=0,
                stories=[],
                message="No RSS entries found. Verify internet connectivity.",
            )

        stories = []
        for entry in feed.entries[:bounded_max]:
            summary = _clean_html(getattr(entry, "summary", ""))
            stories.append(
                NewsStory(
                    title=getattr(entry, "title", "Untitled"),
                    published=getattr(entry, "published", "Unknown date"),
                    summary=summary[:300] + ("..." if len(summary) > 300 else ""),
                    link=getattr(entry, "link", ""),
                )
            )

        return NewsHeadlinesResponse(
            status="success",
            count=len(stories),
            stories=stories,
        )
    except Exception as e:
        return NewsHeadlinesResponse(
            status="error",
            count=0,
            stories=[],
            message=f"Failed to retrieve RSS feed: {e!s}. Suggest retrying or using cached sentiment.",
        )


@mcp.tool()
def get_alphabet_ir_updates(max_results: int = 3) -> IRUpdatesResponse:
    """Retrieves official Alphabet corporate announcements and product updates from the Google Blog RSS.

    Use this tool when users ask about official company announcements, executive leadership
    statements, or corporate press releases rather than third-party financial news.

    Args:
        max_results: Maximum number of corporate updates to return (default 3, max 5).

    Returns:
        Structured IRUpdatesResponse with explicit JSON schema containing status,
        count, and list of corporate announcements.
    """
    bounded_max = max(1, min(max_results, 5))
    try:
        feed = feedparser.parse(GOOGLE_BLOG_RSS_URL)
        if not feed.entries:
            return IRUpdatesResponse(
                status="warning",
                count=0,
                updates=[],
                message="No official blog entries found. Please verify network access.",
            )

        updates = []
        for entry in feed.entries[:bounded_max]:
            summary = _clean_html(getattr(entry, "summary", ""))
            updates.append(
                IRItem(
                    title=getattr(entry, "title", "Untitled"),
                    published=getattr(entry, "published", "Unknown date"),
                    summary=summary[:300] + ("..." if len(summary) > 300 else ""),
                    link=getattr(entry, "link", ""),
                )
            )

        return IRUpdatesResponse(
            status="success",
            count=len(updates),
            updates=updates,
        )
    except Exception as e:
        return IRUpdatesResponse(
            status="error",
            count=0,
            updates=[],
            message=f"Failed to fetch corporate updates: {e!s}.",
        )


if __name__ == "__main__":
    mcp.run()
