"""Unit tests for my-agent-l200 agent definition."""

from app.agent import (
    DISCLAIMER_TEXT,
    market_metrics_agent,
    news_sentiment_agent,
    root_agent,
)


def test_root_agent_configuration():
    """Verify root coordinator agent configuration and sub-agents."""
    assert root_agent.name == "my_agent_l200"
    sub_agent_names = [s.name for s in root_agent.sub_agents]
    assert "market_metrics_agent" in sub_agent_names
    assert "news_sentiment_agent" in sub_agent_names


def test_sub_agent_descriptions():
    """Sub-agents must have clear descriptions for delegation."""
    assert market_metrics_agent.description is not None
    assert len(market_metrics_agent.description) > 20
    assert news_sentiment_agent.description is not None
    assert len(news_sentiment_agent.description) > 20


def test_disclaimer_presence():
    """Mandatory financial disclaimer must be defined and included in coordinator instruction."""
    assert "Disclaimer" in DISCLAIMER_TEXT
    assert "not constitute financial" in DISCLAIMER_TEXT
    assert "Disclaimer" in root_agent.instruction
