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

"""Granular, single-purpose financial tools for Alphabet Inc. market metrics.

Follows ADK best practices:
1. Explicit Pydantic response models providing strict JSON schemas for tool declarations.
2. Task-oriented docstrings describing business capabilities and invocation timing.
3. Granular, single-purposed tool boundaries.
4. Instructive error handling for LLM self-correction.
5. Input validation gates restricting scope to Alphabet Inc. (GOOG / GOOGL).
"""

import time
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
import yfinance as yf

ALLOWED_TICKERS = {"GOOG", "GOOGL"}


# ---------------------------------------------------------------------------
# Base & Concrete Structured Tool Return Schemas
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


class StockQuoteResponse(StructuredToolResult):
    """Structured response schema for Alphabet stock quotes and daily market statistics."""
    status: Literal["success", "error"] = Field(description="Execution outcome status: 'success' or 'error'.")
    symbol: str = Field(description="Stock ticker symbol (GOOG or GOOGL).")
    currency: str = Field(default="USD", description="Currency unit of price figures.")
    current_price: Optional[float] = Field(default=None, description="Latest market price in USD.")
    previous_close: Optional[float] = Field(default=None, description="Previous session closing price in USD.")
    change: Optional[float] = Field(default=None, description="Absolute daily change in USD.")
    change_percent: Optional[float] = Field(default=None, description="Percentage change since previous close.")
    day_high: Optional[float] = Field(default=None, description="Intraday highest price.")
    day_low: Optional[float] = Field(default=None, description="Intraday lowest price.")
    fifty_two_week_high: Optional[float] = Field(default=None, description="Highest price over trailing 52 weeks.")
    fifty_two_week_low: Optional[float] = Field(default=None, description="Lowest price over trailing 52 weeks.")
    volume: Optional[int] = Field(default=None, description="Current trading volume in shares.")
    message: Optional[str] = Field(default=None, description="Instructive error or diagnostic message if status is 'error'.")


class TechnicalIndicatorsResponse(StructuredToolResult):
    """Structured response schema for Alphabet technical trend and momentum indicators."""
    status: Literal["success", "error"] = Field(description="Execution outcome status: 'success' or 'error'.")
    symbol: str = Field(description="Stock ticker symbol (GOOG or GOOGL).")
    current_price: Optional[float] = Field(default=None, description="Current price used for indicator calculations.")
    sma_50: Optional[float] = Field(default=None, description="50-day Simple Moving Average in USD.")
    sma_200: Optional[float] = Field(default=None, description="200-day Simple Moving Average in USD.")
    price_vs_sma50_pct: Optional[float] = Field(default=None, description="Percentage divergence of price relative to 50-day SMA.")
    trend_signal: Optional[str] = Field(default=None, description="Moving average trend regime (e.g. Golden Cross, Bullish).")
    rsi_14: Optional[float] = Field(default=None, description="14-day Relative Strength Index (0 to 100).")
    momentum_state: Optional[str] = Field(default=None, description="Momentum state: Overbought (>=70), Oversold (<=30), or Neutral.")
    message: Optional[str] = Field(default=None, description="Instructive error or diagnostic message if status is 'error'.")


class MockTradeOrderResponse(StructuredToolResult):
    """Structured response schema for simulated paper-trading trade execution."""
    status: Literal["FILLED_SIMULATED", "error"] = Field(description="Order status: 'FILLED_SIMULATED' or 'error'.")
    order_id: Optional[str] = Field(default=None, description="Simulated order tracking identifier.")
    symbol: Optional[str] = Field(default=None, description="Stock ticker symbol (GOOG or GOOGL).")
    action: Optional[Literal["BUY", "SELL"]] = Field(default=None, description="Executed trade direction.")
    shares: Optional[int] = Field(default=None, description="Number of whole shares executed.")
    order_type: Optional[str] = Field(default="MARKET", description="Order execution type (e.g. MARKET, LIMIT).")
    execution_price: Optional[float] = Field(default=None, description="Simulated execution price per share in USD.")
    total_estimated_usd: Optional[float] = Field(default=None, description="Total simulated capital value in USD.")
    execution_timestamp: Optional[str] = Field(default=None, description="UTC timestamp of order execution.")
    hitl_approval: Optional[str] = Field(default=None, description="Confirmation state indicating explicit human sign-off.")
    notice: Optional[str] = Field(default=None, description="Simulated paper-trading regulatory notice.")
    message: Optional[str] = Field(default=None, description="Error explanation if status is 'error'.")


class PriceAlertResponse(StructuredToolResult):
    """Structured response schema for registered price threshold alerts."""
    status: Literal["ACTIVE", "error"] = Field(description="Alert status: 'ACTIVE' or 'error'.")
    alert_id: Optional[str] = Field(default=None, description="Unique tracking identifier for the price alert.")
    symbol: Optional[str] = Field(default=None, description="Stock ticker symbol (GOOG or GOOGL).")
    target_price: Optional[float] = Field(default=None, description="Price threshold boundary in USD.")
    condition: Optional[Literal["ABOVE", "BELOW"]] = Field(default=None, description="Trigger boundary condition.")
    current_price: Optional[float] = Field(default=None, description="Current market price when alert was registered.")
    created_at: Optional[str] = Field(default=None, description="UTC creation timestamp.")
    message: Optional[str] = Field(default=None, description="Confirmation or error message.")


# ---------------------------------------------------------------------------
# Helper Validators
# ---------------------------------------------------------------------------

def _validate_symbol(symbol: str) -> str | None:
    """Validates ticker symbol against the Alphabet focus policy."""
    clean_sym = symbol.strip().upper()
    if clean_sym not in ALLOWED_TICKERS:
        return (
            f"Error: Symbol '{symbol}' is not supported. This agent is strictly specialized "
            f"for Alphabet Inc. Please query either 'GOOGL' (Class A) or 'GOOG' (Class C)."
        )
    return None


# ---------------------------------------------------------------------------
# Quantitative Tools
# ---------------------------------------------------------------------------

def get_stock_quote(symbol: str = "GOOGL") -> StockQuoteResponse:
    """Retrieves real-time/latest price quotes and daily market statistics for Alphabet Inc.

    Use this tool whenever the user asks for the current price, today's high/low, trading
    volume, or 52-week trading boundaries for Google or Alphabet.

    Args:
        symbol: The stock ticker symbol. Must be either 'GOOGL' or 'GOOG'.

    Returns:
        Structured StockQuoteResponse with explicit JSON schema for price, currency,
        daily changes, day range, volume, and 52-week trading boundaries.
    """
    error_msg = _validate_symbol(symbol)
    if error_msg:
        return StockQuoteResponse(
            status="error",
            symbol=symbol.strip().upper(),
            message=error_msg,
        )

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
            return StockQuoteResponse(
                status="error",
                symbol=clean_symbol,
                message=(
                    f"Market data for {clean_symbol} is temporarily unavailable from the public feed. "
                    "Suggest retrying in a moment or checking historical indicators instead."
                ),
            )

        change = current_price - prev_close if prev_close else 0.0
        change_pct = (change / prev_close * 100.0) if prev_close else 0.0

        return StockQuoteResponse(
            status="success",
            symbol=clean_symbol,
            currency=currency,
            current_price=round(current_price, 2),
            previous_close=round(prev_close, 2) if prev_close else None,
            change=round(change, 2),
            change_percent=round(change_pct, 2),
            day_high=round(day_high, 2) if day_high else None,
            day_low=round(day_low, 2) if day_low else None,
            fifty_two_week_high=round(year_high, 2) if year_high else None,
            fifty_two_week_low=round(year_low, 2) if year_low else None,
            volume=volume,
        )
    except Exception as e:
        return StockQuoteResponse(
            status="error",
            symbol=clean_symbol,
            message=f"Failed to retrieve market quote for {clean_symbol}: {e!s}. Suggest retrying.",
        )


def get_technical_indicators(symbol: str = "GOOGL") -> TechnicalIndicatorsResponse:
    """Computes technical trend indicators including 50-day SMA, 200-day SMA, and 14-day RSI for Alphabet Inc.

    Use this tool whenever the user requests moving averages, trend direction, momentum,
    Relative Strength Index (RSI), golden cross / death cross signals, or technical outlooks.

    Args:
        symbol: The stock ticker symbol. Must be either 'GOOGL' or 'GOOG'.

    Returns:
        Structured TechnicalIndicatorsResponse with explicit JSON schema containing current price,
        SMA 50, SMA 200, MA trend signal, 14-day RSI, and overbought/oversold momentum classification.
    """
    error_msg = _validate_symbol(symbol)
    if error_msg:
        return TechnicalIndicatorsResponse(
            status="error",
            symbol=symbol.strip().upper(),
            message=error_msg,
        )

    clean_symbol = symbol.strip().upper()
    try:
        ticker = yf.Ticker(clean_symbol)
        hist = ticker.history(period="1y")

        if hist.empty or len(hist) < 50:
            return TechnicalIndicatorsResponse(
                status="error",
                symbol=clean_symbol,
                message=(
                    f"Insufficient historical price data for {clean_symbol} (found {len(hist)} bars). "
                    "Requires at least 50 days of trading data."
                ),
            )

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

        return TechnicalIndicatorsResponse(
            status="success",
            symbol=clean_symbol,
            current_price=round(current_price, 2),
            sma_50=round(sma_50, 2),
            sma_200=round(sma_200, 2) if sma_200 else None,
            price_vs_sma50_pct=round((current_price - sma_50) / sma_50 * 100.0, 2),
            trend_signal=trend_signal,
            rsi_14=round(rsi_14, 2),
            momentum_state=momentum_state,
        )
    except Exception as e:
        return TechnicalIndicatorsResponse(
            status="error",
            symbol=clean_symbol,
            message=f"Failed to compute technical indicators for {clean_symbol}: {e!s}.",
        )


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
) -> MockTradeOrderResponse:
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
        Structured MockTradeOrderResponse with simulated order ID, execution price,
        total capital allocated, timestamp, and simulated fulfillment status.
    """
    error_msg = _validate_symbol(symbol)
    if error_msg:
        return MockTradeOrderResponse(
            status="error",
            symbol=symbol.strip().upper(),
            message=error_msg,
        )

    clean_symbol = symbol.strip().upper()
    clean_action = action.strip().upper()

    if clean_action not in {"BUY", "SELL"}:
        return MockTradeOrderResponse(
            status="error",
            symbol=clean_symbol,
            message=f"Invalid trade action '{action}'. Action must be either 'BUY' or 'SELL'.",
        )

    try:
        shares_int = int(shares)
        if shares_int <= 0:
            return MockTradeOrderResponse(
                status="error",
                symbol=clean_symbol,
                message=f"Shares must be a positive whole integer greater than 0, got {shares}.",
            )
    except (ValueError, TypeError):
        return MockTradeOrderResponse(
            status="error",
            symbol=clean_symbol,
            message=f"Invalid shares count '{shares}'. Must be an integer greater than 0.",
        )

    quote = get_stock_quote(clean_symbol)
    if quote.status != "success" or quote.current_price is None:
        return MockTradeOrderResponse(
            status="error",
            symbol=clean_symbol,
            message=f"Unable to fetch current market price for trade execution: {quote.message}",
        )

    exec_price = quote.current_price
    total_value = round(exec_price * shares_int, 2)
    order_id = f"MOCK-ORD-{clean_symbol}-{int(time.time())}"

    return MockTradeOrderResponse(
        status="FILLED_SIMULATED",
        order_id=order_id,
        symbol=clean_symbol,
        action=clean_action,  # type: ignore[arg-type]
        shares=shares_int,
        order_type=order_type.upper(),
        execution_price=exec_price,
        total_estimated_usd=total_value,
        execution_timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        hitl_approval="CONFIRMED_BY_USER",
        notice=(
            "This order was executed in simulated paper-trading mode. "
            "No real financial assets or capital were transferred."
        ),
    )


def create_price_alert(
    symbol: str = "GOOGL",
    target_price: float = 200.0,
    condition: str = "ABOVE",
) -> PriceAlertResponse:
    """Registers a persistent threshold price alert for Alphabet Inc. (GOOG/GOOGL).

    Use this tool when the user asks to be notified or alerted when Alphabet's stock
    crosses a specific price boundary (above or below).

    Args:
        symbol: Ticker symbol. Must be either 'GOOGL' or 'GOOG'.
        target_price: The target price threshold in USD.
        condition: Crossing condition. Must be 'ABOVE' or 'BELOW'.

    Returns:
        Structured PriceAlertResponse confirming registered price alert with alert ID and threshold details.
    """
    error_msg = _validate_symbol(symbol)
    if error_msg:
        return PriceAlertResponse(
            status="error",
            symbol=symbol.strip().upper(),
            message=error_msg,
        )

    clean_symbol = symbol.strip().upper()
    clean_condition = condition.strip().upper()
    if clean_condition not in {"ABOVE", "BELOW"}:
        return PriceAlertResponse(
            status="error",
            symbol=clean_symbol,
            message=f"Invalid alert condition '{condition}'. Must be 'ABOVE' or 'BELOW'.",
        )

    try:
        price_float = float(target_price)
        if price_float <= 0:
            return PriceAlertResponse(
                status="error",
                symbol=clean_symbol,
                message=f"Target price must be greater than 0, got {target_price}.",
            )
    except (ValueError, TypeError):
        return PriceAlertResponse(
            status="error",
            symbol=clean_symbol,
            message=f"Invalid target price '{target_price}'. Must be a positive numeric value.",
        )

    quote = get_stock_quote(clean_symbol)
    current_price = quote.current_price

    alert_id = f"ALERT-{clean_symbol}-{int(time.time())}"

    return PriceAlertResponse(
        status="ACTIVE",
        alert_id=alert_id,
        symbol=clean_symbol,
        target_price=round(price_float, 2),
        condition=clean_condition,  # type: ignore[arg-type]
        current_price=current_price,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        message=(
            f"Alert {alert_id} successfully created. Will trigger when {clean_symbol} "
            f"crosses {clean_condition} ${round(price_float, 2)} USD (current: ${current_price})."
        ),
    )
