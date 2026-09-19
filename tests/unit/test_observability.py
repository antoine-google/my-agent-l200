"""Unit tests for observability, structured JSON logging, and PII redaction."""

import json
import logging
from unittest.mock import MagicMock
import pytest

from app.app_utils.observability import (
    IntentClassifier,
    JsonLogFormatter,
    PIIRedactor,
    get_structured_logger,
)
from app.tools.market_metrics import (
    create_price_alert,
    execute_mock_trade_order,
    needs_trade_confirmation,
)


def test_pii_email_and_phone_redaction():
    """Emails and phone numbers must be detected and redacted."""
    raw = "Reach me at analyst.investor@gmail.com or call +1 (555) 234-5678 today."
    sanitized, detected = PIIRedactor.redact_text(raw)
    assert "[REDACTED_EMAIL]" in sanitized
    assert "analyst.investor@gmail.com" not in sanitized
    assert "[REDACTED_PHONE]" in sanitized
    assert "555" not in sanitized
    assert "EMAIL" in detected
    assert "PHONE" in detected


def test_pii_ssn_and_credit_card_redaction():
    """SSN and credit card numbers must be sanitized."""
    raw = "SSN is 987-65-4321 and card is 4111-2222-3333-4444."
    sanitized, detected = PIIRedactor.redact_text(raw)
    assert "[REDACTED_SSN]" in sanitized
    assert "987-65-4321" not in sanitized
    assert "[REDACTED_CREDIT_CARD]" in sanitized
    assert "4111" not in sanitized
    assert "SSN" in detected
    assert "CREDIT_CARD" in detected


def test_pii_secret_redaction():
    """API keys and bearer tokens must be stripped."""
    raw = "Key AIzaSyD1234567890abcdef1234567890abcdef and Bearer eyJhbGciOi"
    sanitized, detected = PIIRedactor.redact_text(raw)
    assert "[REDACTED_API_KEY_OR_SECRET]" in sanitized
    assert "AIzaSyD" not in sanitized


def test_pii_recursive_data_structure():
    """Nested dict and list structures must be recursively sanitized."""
    data = {
        "user": {
            "email": "john.doe@google.com",
            "phones": ["+1-800-555-0199", "123-456-7890"],
        },
        "query": "What is the price of GOOGL?",
    }
    redacted = PIIRedactor.redact_data(data)
    assert redacted["user"]["email"] == "[REDACTED_EMAIL]"
    assert redacted["user"]["phones"][0] == "[REDACTED_PHONE]"
    assert redacted["query"] == "What is the price of GOOGL?"


def test_intent_classification_market_quote():
    """Quote inquiries must be categorized as MARKET_QUOTE."""
    res = IntentClassifier.classify_intent("What is the current stock price and volume for GOOGL?")
    assert res["primary_intent"] == "MARKET_QUOTE"
    assert "GOOGL" in res["detected_tickers"]


def test_intent_classification_technical_analysis():
    """Moving averages and RSI queries must be categorized as TECHNICAL_ANALYSIS."""
    res = IntentClassifier.classify_intent("Calculate the 50-day SMA, 200-day SMA and RSI momentum for GOOG")
    assert res["primary_intent"] == "TECHNICAL_ANALYSIS"
    assert "GOOG" in res["detected_tickers"]


def test_intent_classification_mock_trade():
    """Buy/sell requests must be categorized as MOCK_TRADE."""
    res = IntentClassifier.classify_intent("Please execute a buy order for 10 shares of GOOGL")
    assert res["primary_intent"] == "MOCK_TRADE"
    assert "GOOGL" in res["detected_tickers"]


def test_intent_classification_price_alert():
    """Price threshold alert requests must be categorized as PRICE_ALERT."""
    res = IntentClassifier.classify_intent("Set an alert if Alphabet crosses above 200")
    assert res["primary_intent"] == "PRICE_ALERT"
    assert "GOOGL" in res["detected_tickers"]


def test_intent_classification_news_sentiment():
    """Headlines and PR queries must be categorized as NEWS_SENTIMENT."""
    res = IntentClassifier.classify_intent("What are the latest news headlines and regulatory updates on Google?")
    assert res["primary_intent"] == "NEWS_SENTIMENT"


def test_json_log_formatter():
    """JsonLogFormatter must emit valid single-line JSON with standard fields."""
    formatter = JsonLogFormatter(service_name="test-agent")
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Sample test log message",
        args=(),
        exc_info=None,
    )
    record.payload = {"intent": "MARKET_QUOTE", "latency_ms": 120.5}

    formatted = formatter.format(record)
    assert "\n" not in formatted  # Single line
    parsed = json.loads(formatted)
    assert parsed["severity"] == "INFO"
    assert parsed["service"] == "test-agent"
    assert parsed["message"] == "Sample test log message"
    assert parsed["intent"] == "MARKET_QUOTE"
    assert parsed["latency_ms"] == 120.5


def test_hitl_trade_confirmation_gate():
    """Verify that trade confirmation gate always returns True."""
    assert needs_trade_confirmation("GOOGL", "BUY", 10) is True
    assert needs_trade_confirmation("GOOG", "SELL", 1) is True


def test_execute_mock_trade_validation():
    """Verify mock trade parameter validation."""
    # Invalid ticker
    inv_tick = execute_mock_trade_order(symbol="AAPL", action="BUY", shares=1)
    assert inv_tick["status"] == "error"
    assert "strictly specialized for Alphabet Inc" in inv_tick["message"]

    # Invalid action
    inv_act = execute_mock_trade_order(symbol="GOOGL", action="SHORT", shares=1)
    assert inv_act["status"] == "error"
    assert "BUY' or 'SELL" in inv_act["message"]

    # Invalid shares
    inv_shares = execute_mock_trade_order(symbol="GOOGL", action="BUY", shares=0)
    assert inv_shares["status"] == "error"
    assert "greater than 0" in inv_shares["message"]


def test_create_price_alert_validation():
    """Verify price alert parameter validation."""
    # Invalid ticker
    inv_tick = create_price_alert(symbol="MSFT", target_price=400.0)
    assert inv_tick["status"] == "error"

    # Invalid condition
    inv_cond = create_price_alert(symbol="GOOGL", target_price=200.0, condition="BETWEEN")
    assert inv_cond["status"] == "error"
    assert "ABOVE' or 'BELOW" in inv_cond["message"]

    # Invalid target price
    inv_p = create_price_alert(symbol="GOOGL", target_price=-10.0)
    assert inv_p["status"] == "error"
    assert "greater than 0" in inv_p["message"]
