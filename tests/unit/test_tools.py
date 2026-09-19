"""Unit tests for my-agent-l200 tools."""

from unittest.mock import MagicMock, patch
import pandas as pd

from app.tools.market_metrics import (
    ALLOWED_TICKERS,
    _validate_symbol,
    get_stock_quote,
    get_technical_indicators,
)
from mcp_servers.rss_server import (
    _clean_html,
    get_alphabet_ir_updates,
    get_alphabet_news_headlines,
)


def test_symbol_validation():
    """Verify that only GOOG and GOOGL are allowed."""
    assert _validate_symbol("GOOGL") is None
    assert _validate_symbol("goog") is None
    assert _validate_symbol("AAPL") is not None
    assert "not supported" in _validate_symbol("MSFT")


def test_stock_quote_invalid_ticker():
    """Invalid tickers should return instructive error message."""
    res = get_stock_quote("AMZN")
    assert res["status"] == "error"
    assert "strictly specialized for Alphabet Inc" in res["message"]


def test_technical_indicators_invalid_ticker():
    """Invalid tickers should return instructive error message."""
    res = get_technical_indicators("TSLA")
    assert res["status"] == "error"
    assert "strictly specialized for Alphabet Inc" in res["message"]


@patch("yfinance.Ticker")
def test_stock_quote_success(mock_ticker_cls):
    """Verify get_stock_quote parses yfinance info correctly."""
    mock_instance = MagicMock()
    mock_instance.fast_info.last_price = 185.50
    mock_instance.fast_info.previous_close = 183.00
    mock_instance.fast_info.day_high = 187.00
    mock_instance.fast_info.day_low = 184.20
    mock_instance.fast_info.year_high = 191.75
    mock_instance.fast_info.year_low = 129.50
    mock_instance.fast_info.last_volume = 25000000
    mock_instance.fast_info.currency = "USD"
    mock_ticker_cls.return_value = mock_instance

    res = get_stock_quote("GOOGL")
    assert res["status"] == "success"
    assert res["symbol"] == "GOOGL"
    assert res["current_price"] == 185.50
    assert res["change"] == 2.50
    assert res["change_percent"] == round(2.50 / 183.00 * 100, 2)


@patch("yfinance.Ticker")
def test_technical_indicators_calculation(mock_ticker_cls):
    """Verify SMA 50 and RSI 14 calculation logic."""
    mock_instance = MagicMock()
    # Create 220 days of simulated prices
    prices = [100.0 + i * 0.5 for i in range(220)]
    df = pd.DataFrame({"Close": prices})
    mock_instance.history.return_value = df
    mock_ticker_cls.return_value = mock_instance

    res = get_technical_indicators("GOOGL")
    assert res["status"] == "success"
    assert "sma_50" in res
    assert "sma_200" in res
    assert "rsi_14" in res
    assert "trend_signal" in res
    assert res["sma_50"] > res["sma_200"]  # Since prices were monotonically increasing


def test_clean_html():
    """Verify HTML cleaning utility."""
    raw = "<p>Google reports strong <b>Q4</b> results <a href='...'>link</a></p>"
    cleaned = _clean_html(raw)
    assert "<b>" not in cleaned
    assert "Google reports strong Q4 results link" == cleaned


@patch("feedparser.parse")
def test_news_headlines_parsing(mock_parse):
    """Verify RSS headlines parsing and bounding."""
    mock_feed = MagicMock()
    mock_entry = MagicMock()
    mock_entry.title = "Google launches new AI feature"
    mock_entry.published = "Sat, 19 Sep 2026 12:00:00 GMT"
    mock_entry.summary = "Summary of AI announcement."
    mock_entry.link = "https://example.com/news/1"
    mock_feed.entries = [mock_entry] * 15
    mock_parse.return_value = mock_feed

    res = get_alphabet_news_headlines(max_results=3)
    assert res["status"] == "success"
    assert res["count"] == 3
    assert len(res["stories"]) == 3
    assert res["stories"][0]["title"] == "Google launches new AI feature"
