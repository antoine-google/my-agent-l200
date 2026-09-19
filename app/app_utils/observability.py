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

"""Enterprise Observability, Structured JSON Logging, Intent-vs-Outcome Tracking,
and PII Redaction for Alphabet Financial Agent.
"""

from datetime import datetime, timezone
import json
import logging
import re
import time
from typing import Any, Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

# ---------------------------------------------------------------------------
# 1. PII Redaction Engine
# ---------------------------------------------------------------------------

class PIIRedactor:
    """Detects and redacts Personally Identifiable Information (PII) and credentials.

    Protects customer privacy by replacing sensitive patterns (emails, phone numbers,
    SSNs, credit cards, banking IBANs, and API credentials) before data reaches
    model reasoning, persistent memory, or log sinks.
    """

    PATTERNS: dict[str, re.Pattern[str]] = {
        "EMAIL": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        "PHONE": re.compile(
            r"(?:\+\d{1,3}[-.\s]*)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "SSN": re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b"
        ),
        "CREDIT_CARD": re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        ),
        "IBAN": re.compile(
            r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"
        ),
        "API_KEY_OR_SECRET": re.compile(
            r"\b(?:AIza[0-9A-Za-z-_]{35}|sk-[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9_\-\.]+)\b"
        ),
    }

    @classmethod
    def redact_text(cls, text: str) -> tuple[str, list[str]]:
        """Redacts all known PII patterns from the text string.

        Args:
            text: Raw input string.

        Returns:
            Tuple of (sanitized_text, list_of_detected_pii_types).
        """
        if not text or not isinstance(text, str):
            return text, []

        sanitized = text
        detected: list[str] = []

        for pii_type, pattern in cls.PATTERNS.items():
            matches = pattern.findall(sanitized)
            if matches:
                detected.append(pii_type)
                sanitized = pattern.sub(f"[REDACTED_{pii_type}]", sanitized)

        return sanitized, detected

    @classmethod
    def redact_data(cls, data: Any) -> Any:
        """Recursively sanitizes dictionary, list, and string data structures."""
        if isinstance(data, str):
            sanitized, _ = cls.redact_text(data)
            return sanitized
        elif isinstance(data, dict):
            return {k: cls.redact_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.redact_data(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(cls.redact_data(item) for item in data)
        return data


# ---------------------------------------------------------------------------
# 2. Structured JSON Log Formatter
# ---------------------------------------------------------------------------

class JsonLogFormatter(logging.Formatter):
    """Formats log records as structured, single-line JSON entries.

    Compatible with Google Cloud Logging jsonPayload and modern log collectors.
    """

    def __init__(self, service_name: str = "alphabet-agent-l200"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }

        # Include custom structured payload if present on record
        if hasattr(record, "payload") and isinstance(record.payload, dict):
            log_entry.update(record.payload)

        # Include exception trace if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def get_structured_logger(name: str = "alphabet_agent") -> logging.Logger:
    """Configures and returns a logger with Structured JSON formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


structured_logger = get_structured_logger()


# ---------------------------------------------------------------------------
# 3. Intent Classification Engine
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Analyzes prompt text to classify user intent and target entities."""

    INTENT_KEYWORDS = {
        "MOCK_TRADE": ["buy", "sell", "trade", "shares", "order", "purchase"],
        "PRICE_ALERT": ["alert", "notify", "threshold", "price cross", "trigger"],
        "TECHNICAL_ANALYSIS": [
            "technical", "sma", "moving average", "rsi", "momentum",
            "golden cross", "death cross", "overbought", "oversold", "trend"
        ],
        "MARKET_QUOTE": [
            "quote", "price", "volume", "high", "low", "market cap",
            "how much is", "52-week", "trading at"
        ],
        "NEWS_SENTIMENT": [
            "news", "headline", "sentiment", "press release", "ir update",
            "blog", "announcement", "earnings", "antitrust", "regulatory"
        ],
        "COMPREHENSIVE_BRIEFING": [
            "briefing", "outlook", "synthesis", "comprehensive", "full analysis",
            "summary of alphabet", "bull vs bear", "predict", "forecast"
        ],
    }

    @classmethod
    def classify_intent(cls, prompt_text: str) -> dict[str, Any]:
        """Classifies the user intent category and extracts Alphabet tickers."""
        lower_prompt = prompt_text.lower() if prompt_text else ""

        # Detect tickers
        tickers = []
        if "googl" in lower_prompt:
            tickers.append("GOOGL")
        if "goog" in lower_prompt and "googl" not in lower_prompt:
            tickers.append("GOOG")
        if not tickers and ("google" in lower_prompt or "alphabet" in lower_prompt):
            tickers.append("GOOGL")

        # Classify primary intent by matching keywords
        scores: dict[str, int] = {}
        for category, keywords in cls.INTENT_KEYWORDS.items():
            matched = sum(1 for kw in keywords if kw in lower_prompt)
            if matched > 0:
                scores[category] = matched

        if not scores:
            primary_intent = "GENERAL_INQUIRY"
        else:
            primary_intent = max(scores.items(), key=lambda item: item[1])[0]

        return {
            "primary_intent": primary_intent,
            "detected_tickers": tickers or ["GOOGL"],
            "matched_signals": scores,
        }


# ---------------------------------------------------------------------------
# 4. ADK Observability & Intent-vs-Outcome Plugin
# ---------------------------------------------------------------------------

class ObservabilityPlugin(BasePlugin):
    """ADK Plugin providing PII sanitization, tool tracing, and intent-vs-outcome logging.

    Lifecycle hooks:
    1. on_user_message_callback: Redacts PII in user messages before LLM execution.
    2. before_run_callback: Captures invocation metadata, timestamp, and intent.
    3. before_tool_callback / after_tool_callback: Traces tool executions and arguments.
    4. after_run_callback: Synthesizes intent-vs-outcome record with latency, tools executed,
       status, and financial compliance disclaimer verification.
    """

    def __init__(self, service_name: str = "alphabet-agent-l200"):
        super().__init__(name="observability_plugin")
        self.service_name = service_name
        self._run_contexts: dict[str, dict[str, Any]] = {}

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        """Sanitizes user input for PII before passing it to LLM or state storage."""
        if not user_message or not user_message.parts:
            return None

        pii_found_any = False
        new_parts = []

        for part in user_message.parts:
            if part.text:
                sanitized_text, pii_detected = PIIRedactor.redact_text(part.text)
                if pii_detected:
                    pii_found_any = True
                    structured_logger.warning(
                        "PII detected and redacted from user input",
                        extra={
                            "payload": {
                                "event_type": "pii_redaction_applied",
                                "pii_types": pii_detected,
                                "session_id": getattr(invocation_context.session, "id", "unknown"),
                            }
                        },
                    )
                new_parts.append(types.Part(text=sanitized_text))
            else:
                new_parts.append(part)

        if pii_found_any:
            return types.Content(role=user_message.role, parts=new_parts)

        return None

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[types.Content]:
        """Initializes intent tracking and latency counter for the invocation run."""
        inv_id = str(invocation_context.invocation_id)
        user_text = ""
        if invocation_context.user_content and invocation_context.user_content.parts:
            for part in invocation_context.user_content.parts:
                if part.text:
                    user_text += part.text + " "

        sanitized_query, _ = PIIRedactor.redact_text(user_text.strip())
        intent_data = IntentClassifier.classify_intent(sanitized_query)

        self._run_contexts[inv_id] = {
            "start_time": time.time(),
            "query": sanitized_query,
            "intent": intent_data,
            "tools_executed": [],
            "session_id": getattr(invocation_context.session, "id", "unknown"),
            "user_id": getattr(invocation_context.session, "user_id", "anonymous"),
        }
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict[str, Any]]:
        """Tracks tool execution initiation with redacted arguments."""
        inv_id = str(tool_context.invocation_id)
        sanitized_args = PIIRedactor.redact_data(tool_args)

        tool_record = {
            "tool_name": tool.name,
            "arguments": sanitized_args,
            "agent": tool_context.agent_name,
            "start_time": time.time(),
            "status": "RUNNING",
        }

        if inv_id in self._run_contexts:
            self._run_contexts[inv_id]["tools_executed"].append(tool_record)

        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Records tool completion status and latency."""
        inv_id = str(tool_context.invocation_id)
        if inv_id in self._run_contexts:
            for record in reversed(self._run_contexts[inv_id]["tools_executed"]):
                if record["tool_name"] == tool.name and record["status"] == "RUNNING":
                    record["status"] = (
                        "ERROR" if isinstance(result, dict) and result.get("status") == "error"
                        else "SUCCESS"
                    )
                    record["duration_ms"] = round((time.time() - record["start_time"]) * 1000, 2)
                    break
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """Synthesizes and logs the complete Intent-vs-Outcome structured record."""
        inv_id = str(invocation_context.invocation_id)
        run_data = self._run_contexts.pop(inv_id, None)

        if not run_data:
            return None

        total_latency_ms = round((time.time() - run_data["start_time"]) * 1000, 2)
        tools_summary = [
            {
                "tool": t["tool_name"],
                "status": t["status"],
                "duration_ms": t.get("duration_ms", 0),
            }
            for t in run_data["tools_executed"]
        ]

        # Scan recent session events to verify financial disclaimer compliance
        disclaimer_present = False
        if invocation_context.session and invocation_context.session.events:
            for ev in reversed(invocation_context.session.events[-3:]):
                if ev.content and ev.content.parts:
                    for part in ev.content.parts:
                        if part.text and ("Disclaimer:" in part.text or "not constitute financial" in part.text):
                            disclaimer_present = True
                            break

        intent_category = run_data["intent"]["primary_intent"]
        requires_disclaimer = intent_category in {
            "TECHNICAL_ANALYSIS", "COMPREHENSIVE_BRIEFING", "MOCK_TRADE"
        }

        # Intent vs Outcome record
        record: dict[str, Any] = {
            "event_type": "intent_vs_outcome",
            "invocation_id": inv_id,
            "session_id": run_data["session_id"],
            "user_id": run_data["user_id"],
            "intent": {
                "category": intent_category,
                "sanitized_query": run_data["query"][:250],
                "detected_tickers": run_data["intent"]["detected_tickers"],
            },
            "outcome": {
                "status": "SUCCESS",
                "latency_ms": total_latency_ms,
                "tool_count": len(run_data["tools_executed"]),
                "tools_executed": tools_summary,
                "compliance": {
                    "requires_disclaimer": requires_disclaimer,
                    "disclaimer_verified": disclaimer_present,
                    "compliant": (not requires_disclaimer) or disclaimer_present,
                },
            },
        }

        structured_logger.info(
            f"Intent vs Outcome: {intent_category} -> {len(run_data['tools_executed'])} tools in {total_latency_ms}ms",
            extra={"payload": record},
        )
        return None
