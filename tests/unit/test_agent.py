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


def test_strategic_model_routing():
    """Verify each agent uses a specialized model according to strategic routing."""
    from app.agent import COORDINATOR_MODEL, METRICS_MODEL, NEWS_MODEL, SUMMARIZER_MODEL, app

    # Coordinator uses high-reasoning model
    assert root_agent.model.model == COORDINATOR_MODEL
    assert "pro" in COORDINATOR_MODEL

    # News specialist uses fast multimodal model
    assert news_sentiment_agent.model.model == NEWS_MODEL
    assert "flash" in NEWS_MODEL

    # Market metrics uses ultra-low latency model
    assert market_metrics_agent.model.model == METRICS_MODEL
    assert "lite" in METRICS_MODEL or "flash" in METRICS_MODEL

    # Compaction summarizer uses cost-effective model
    assert app.events_compaction_config.summarizer._llm.model == SUMMARIZER_MODEL


def test_hitl_mock_trade_tool_attached():
    """Market metrics agent must have the HITL mock trade tool with confirmation enabled."""
    tool_names = [getattr(t, "name", getattr(t, "__name__", None)) for t in market_metrics_agent.tools]
    assert "execute_mock_trade_order" in tool_names
    assert "create_price_alert" in tool_names

    # Verify mock trade tool requires confirmation (HITL gate)
    trade_tool = next(
        t for t in market_metrics_agent.tools
        if getattr(t, "name", getattr(t, "__name__", None)) == "execute_mock_trade_order"
    )
    assert hasattr(trade_tool, "_require_confirmation")
    assert trade_tool._require_confirmation is not False


def test_app_resumability_and_observability():
    """App must have resumability enabled for HITL and ObservabilityPlugin configured."""
    from app.agent import app

    assert app.resumability_config is not None
    assert app.resumability_config.is_resumable is True

    plugin_names = [p.name for p in app.plugins]
    assert "observability_plugin" in plugin_names

