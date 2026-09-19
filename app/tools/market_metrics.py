"""Granular, single-purpose financial tools for Alphabet Inc. market metrics.

Follows ADK best practices:
1. Task-oriented docstrings describing business capabilities and invocation timing.
2. Granular, single-purposed tool boundaries.
3. Instructive error handling for LLM self-correction.
4. Input validation gates restricting scope to Alphabet Inc. (GOOG / GOOGL).
"""

from typing import Any
import pandas as pd
import yfinance as yf

ALLOWED_TICKERS = {"GOOG", "GOOGL"}


def _validate_symbol(symbol: str) -> str | None:
    """Validates ticker symbol against the Alphabet focus policy."""
    clean_sym = symbol.strip().upper()
    if clean_sym not in ALLOWED_TICKERS:
        return (
            f"Error: Symbol '{symbol}' is not supported. This agent is strictly specialized "
            f"for Alphabet Inc. Please query either 'GOOGL' (Class A) or 'GOOG' (Class C)."
        )
    return None


def get_stock_quote(symbol: str = "GOOGL") -> dict[str, Any]:
    """Retrieves real-time/latest price quotes and daily market statistics for Alphabet Inc.

    Use this tool whenever the user asks for the current price, today's high/low, trading
    volume, or 52-week trading boundaries for Google or Alphabet.

    Args:
        symbol: The stock ticker symbol. Must be either 'GOOGL' or 'GOOG'.

    Returns:
        Structured dictionary containing price, currency, daily changes, day range,
        volume, and 52-week trading boundaries.
    """
    error_msg = _validate_symbol(symbol)
    if error_msg:
        return {"status": "error", "message": error_msg}

    clean_symbol = symbol.strip().upper()
    try:
        ticker = yf.Ticker(clean_symbol)
        info = ticker.fast_info

        current_price = getattr(info, "last_price", None)
        prev_close = getattr(info, "previous_close", None)
        day_high = getattr(info, "day_high", None)
        day_low = getattr(info, "day_low", None)
        year_high = getattr(info, "year_high", None)
        year_low = getattr(info, "year_low", None)
        volume = getattr(info, "last_volume", None)
        currency = getattr(info, "currency", "USD")

        if current_price is None:
            # Fallback to history lookup
            hist = ticker.history(period="5d")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
                day_high = float(hist["High"].iloc[-1])
                day_low = float(hist["Low"].iloc[-1])
                volume = int(hist["Volume"].iloc[-1])

        if current_price is None:
            return {
                "status": "error",
                "message": (
                    f"Market data for {clean_symbol} is temporarily unavailable from the public feed. "
                    "Suggest retrying in a moment or checking historical indicators instead."
                ),
            }

        change = current_price - prev_close if prev_close else 0.0
        change_pct = (change / prev_close * 100.0) if prev_close else 0.0

        return {
            "status": "success",
            "symbol": clean_symbol,
            "currency": currency,
            "current_price": round(current_price, 2),
            "previous_close": round(prev_close, 2) if prev_close else None,
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "day_high": round(day_high, 2) if day_high else None,
            "day_low": round(day_low, 2) if day_low else None,
            "fifty_two_week_high": round(year_high, 2) if year_high else None,
            "fifty_two_week_low": round(year_low, 2) if year_low else None,
            "volume": volume,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve market quote for {clean_symbol}: {e!s}. Suggest retrying.",
        }


def get_technical_indicators(symbol: str = "GOOGL") -> dict[str, Any]:
    """Computes technical trend indicators including 50-day SMA, 200-day SMA, and 14-day RSI for Alphabet Inc.

    Use this tool whenever the user requests moving averages, trend direction, momentum,
    Relative Strength Index (RSI), golden cross / death cross signals, or technical outlooks.

    Args:
        symbol: The stock ticker symbol. Must be either 'GOOGL' or 'GOOG'.

    Returns:
        Structured dictionary containing current price, SMA 50, SMA 200, MA trend signal,
        14-day RSI, and overbought/oversold momentum classification.
    """
    error_msg = _validate_symbol(symbol)
    if error_msg:
        return {"status": "error", "message": error_msg}

    clean_symbol = symbol.strip().upper()
    try:
        ticker = yf.Ticker(clean_symbol)
        # Fetch 1 year of daily history to accurately calculate 200-day SMA and RSI
        hist = ticker.history(period="1y")

        if hist.empty or len(hist) < 50:
            return {
                "status": "error",
                "message": (
                    f"Insufficient historical price data for {clean_symbol} (found {len(hist)} bars). "
                    "Requires at least 50 days of trading data."
                ),
            }

        close_series = hist["Close"]
        current_price = float(close_series.iloc[-1])

        # Moving Averages
        sma_50 = float(close_series.tail(50).mean())
        sma_200 = float(close_series.tail(200).mean()) if len(close_series) >= 200 else None

        # MA Signal classification
        if sma_200:
            if sma_50 > sma_200 and current_price > sma_50:
                trend_signal = "Strong Bullish (Price > SMA50 > SMA200, Golden Cross posture)"
            elif sma_50 > sma_200:
                trend_signal = "Bullish posture (SMA50 > SMA200)"
            elif sma_50 < sma_200 and current_price < sma_50:
                trend_signal = "Strong Bearish (Price < SMA50 < SMA200, Death Cross posture)"
            else:
                trend_signal = "Neutral / Consolidating"
        else:
            trend_signal = "Bullish" if current_price > sma_50 else "Bearish"

        # RSI (14-day Wilder smoothing)
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi_14 = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

        if rsi_14 >= 70:
            momentum_state = "Overbought (RSI >= 70) - potential short-term pullback risk"
        elif rsi_14 <= 30:
            momentum_state = "Oversold (RSI <= 30) - potential mean-reversion buying opportunity"
        else:
            momentum_state = "Neutral momentum (30 < RSI < 70)"

        return {
            "status": "success",
            "symbol": clean_symbol,
            "current_price": round(current_price, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2) if sma_200 else None,
            "price_vs_sma50_pct": round((current_price - sma_50) / sma_50 * 100.0, 2),
            "trend_signal": trend_signal,
            "rsi_14": round(rsi_14, 2),
            "momentum_state": momentum_state,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to compute technical indicators for {clean_symbol}: {e!s}.",
        }


def needs_trade_confirmation(
    symbol: str, action: str, shares: int, **kwargs: Any
) -> bool:
    """Human-in-the-Loop approval gate: gates trade execution behind human confirmation.

    All mock trade orders involve simulated capital allocation and require explicit user consent.
    """
    return True


def execute_mock_trade_order(
    symbol: str = "GOOGL",
    action: str = "BUY",
    shares: int = 1,
    order_type: str = "MARKET",
) -> dict[str, Any]:
    """Executes a simulated mock stock trade (BUY or SELL) for Alphabet Inc.

    CRITICAL: This is a high-impact, state-changing transactional tool that requires
    mandatory Human-In-The-Loop (HITL) approval before execution. Do not execute
    without explicit user consent.

    Args:
        symbol: Ticker symbol to trade. Must be strictly 'GOOGL' or 'GOOG'.
        action: Trade direction. Must be either 'BUY' or 'SELL'.
        shares: Number of whole shares to trade. Must be an integer greater than 0.
        order_type: Order type, default is 'MARKET' (or 'LIMIT').

    Returns:
        Structured execution receipt with simulated order ID, execution price,
        total capital allocated, timestamp, and simulated fulfillment status.
    """
    error_msg = _validate_symbol(symbol)
    if error_msg:
        return {"status": "error", "message": error_msg}

    clean_symbol = symbol.strip().upper()
    clean_action = action.strip().upper()

    if clean_action not in {"BUY", "SELL"}:
        return {
            "status": "error",
            "message": f"Invalid trade action '{action}'. Action must be either 'BUY' or 'SELL'.",
        }

    try:
        shares_int = int(shares)
        if shares_int <= 0:
            return {
                "status": "error",
                "message": f"Shares must be a positive whole integer greater than 0, got {shares}.",
            }
    except (ValueError, TypeError):
        return {
            "status": "error",
            "message": f"Invalid shares count '{shares}'. Must be an integer greater than 0.",
        }

    quote = get_stock_quote(clean_symbol)
    if quote.get("status") != "success":
        return {
            "status": "error",
            "message": f"Unable to fetch current market price for trade execution: {quote.get('message')}",
        }

    exec_price = quote["current_price"]
    total_value = round(exec_price * shares_int, 2)
    import time
    order_id = f"MOCK-ORD-{clean_symbol}-{int(time.time())}"

    return {
        "status": "FILLED_SIMULATED",
        "order_id": order_id,
        "symbol": clean_symbol,
        "action": clean_action,
        "shares": shares_int,
        "order_type": order_type.upper(),
        "execution_price": exec_price,
        "total_estimated_usd": total_value,
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "hitl_approval": "CONFIRMED_BY_USER",
        "notice": (
            "This order was executed in simulated paper-trading mode. "
            "No real financial assets or capital were transferred."
        ),
    }


def create_price_alert(
    symbol: str = "GOOGL",
    target_price: float = 200.0,
    condition: str = "ABOVE",
) -> dict[str, Any]:
    """Registers a persistent threshold price alert for Alphabet Inc. (GOOG/GOOGL).

    Use this tool when the user asks to be notified or alerted when Alphabet's stock
    crosses a specific price boundary (above or below).

    Args:
        symbol: Ticker symbol. Must be either 'GOOGL' or 'GOOG'.
        target_price: The target price threshold in USD.
        condition: Crossing condition. Must be 'ABOVE' or 'BELOW'.

    Returns:
        Confirmation of registered price alert with alert ID and threshold details.
    """
    error_msg = _validate_symbol(symbol)
    if error_msg:
        return {"status": "error", "message": error_msg}

    clean_symbol = symbol.strip().upper()
    clean_condition = condition.strip().upper()
    if clean_condition not in {"ABOVE", "BELOW"}:
        return {
            "status": "error",
            "message": f"Invalid alert condition '{condition}'. Must be 'ABOVE' or 'BELOW'.",
        }

    try:
        price_float = float(target_price)
        if price_float <= 0:
            return {
                "status": "error",
                "message": f"Target price must be greater than 0, got {target_price}.",
            }
    except (ValueError, TypeError):
        return {
            "status": "error",
            "message": f"Invalid target price '{target_price}'. Must be a positive numeric value.",
        }

    quote = get_stock_quote(clean_symbol)
    current_price = quote.get("current_price")

    import time
    alert_id = f"ALERT-{clean_symbol}-{int(time.time())}"

    return {
        "status": "ACTIVE",
        "alert_id": alert_id,
        "symbol": clean_symbol,
        "target_price": round(price_float, 2),
        "condition": clean_condition,
        "current_price": current_price,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "message": (
            f"Alert {alert_id} successfully created. Will trigger when {clean_symbol} "
            f"crosses {clean_condition} ${round(price_float, 2)} USD (current: ${current_price})."
        ),
    }

