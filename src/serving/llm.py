"""
LLM-powered listing analysis for the Mercari Price Prediction Engine.

Given a product listing and the ML model's price prediction, asks an LLM for:
- an independent (zero-shot) price estimate with reasoning
- a quality score and concrete critique of the listing itself

The two estimates are compared so users can see when the trained model and
the LLM agree — and when they don't.

Two interchangeable providers share the same structured-output schema:
- GeminiAnalyzer    — activates when GEMINI_API_KEY is set (free tier available)
- AnthropicAnalyzer — activates when ANTHROPIC_API_KEY is set

`create_analyzer()` picks whichever is configured; the endpoint degrades
gracefully (503) when neither key is present.
"""

import logging
import os
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a pricing analyst for a US peer-to-peer marketplace \
(similar to Mercari or Poshmark), where most items are second-hand and sell for \
$3-$200. You will receive a product listing and a price estimate produced by a \
machine-learning model trained on 1.48M historical listings.

Your job:
1. Give your own independent price estimate in USD for what this item would \
realistically sell for on such a marketplace. Do not anchor on the ML model's \
number — estimate first from the item itself, then comment on any disagreement \
in your reasoning.
2. Score the listing quality from 1 (bare, will underperform) to 10 (excellent, \
maximizes sale price) based on how well the title, description, brand, and \
category information would attract buyers and justify the price.
3. List the listing's concrete strengths and, more importantly, specific \
improvements the seller should make (e.g. missing brand, vague condition, no \
size, description too short to justify price).
4. If the title could be improved, suggest a better one; otherwise omit it.

Be concrete and grounded in what actually drives resale prices: brand \
recognition, condition, completeness of information, and category norms."""

CONDITION_LABELS = {1: "New with tags", 2: "New without tags",
                    3: "Good", 4: "Fair", 5: "Poor"}


class ListingAnalysis(BaseModel):
    """Structured output schema for the LLM's listing analysis."""

    llm_estimated_price: float = Field(
        description="Your independent price estimate in USD"
    )
    price_reasoning: str = Field(
        description="2-4 sentences explaining your estimate, including whether "
                    "you agree with the ML model's number and why"
    )
    listing_score: int = Field(
        ge=1, le=10, description="Listing quality score from 1 to 10"
    )
    strengths: List[str] = Field(
        description="What the listing already does well (may be empty)"
    )
    improvements: List[str] = Field(
        description="Specific, actionable changes to improve the listing"
    )
    suggested_title: Optional[str] = Field(
        default=None,
        description="A better listing title, only if the current one is weak"
    )


def build_comparison(model_price: float, llm_price: float) -> dict:
    """Compare the ML model's estimate against the LLM's estimate."""
    delta = llm_price - model_price
    delta_pct = (delta / model_price * 100) if model_price > 0 else 0.0
    abs_pct = abs(delta_pct)
    if abs_pct < 15:
        agreement = "close"
    elif abs_pct < 40:
        agreement = "moderate"
    else:
        agreement = "divergent"
    return {
        "model_price": round(model_price, 2),
        "llm_price": round(llm_price, 2),
        "delta": round(delta, 2),
        "delta_pct": round(delta_pct, 1),
        "agreement": agreement,
    }


def build_user_message(listing, model_prediction) -> str:
    """Render the listing and the ML prediction into the analysis prompt."""
    return f"""Listing:
- Title: {listing.name}
- Description: {listing.item_description or "(none)"}
- Category: {listing.category_name or "(none)"}
- Brand: {listing.brand_name or "(none)"}
- Condition: {CONDITION_LABELS.get(listing.item_condition_id, "Unknown")}
- Shipping: {"seller pays" if listing.shipping else "buyer pays"}

ML model estimate: ${model_prediction.predicted_price:.2f} \
(range ${model_prediction.confidence_range["low"]:.2f}-\
${model_prediction.confidence_range["high"]:.2f})

Analyze this listing."""


class GeminiAnalyzer:
    """Listing analysis via the Google Gemini API (free tier available)."""

    provider = "gemini"

    def __init__(self, config: Optional[dict] = None):
        llm_cfg = (config or {}).get("llm", {})
        self.model = os.environ.get(
            "GEMINI_MODEL", llm_cfg.get("gemini_model", DEFAULT_GEMINI_MODEL)
        )
        self.max_tokens = llm_cfg.get("max_tokens", DEFAULT_MAX_TOKENS)
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client()
        return self._client

    async def analyze(self, listing, model_prediction) -> ListingAnalysis:
        from google.genai import types

        client = self._get_client()
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=build_user_message(listing, model_prediction),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=ListingAnalysis,
                max_output_tokens=self.max_tokens,
                # Disable thinking so the token budget goes to the JSON answer
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        if response.parsed is not None:
            return response.parsed
        # SDK returns parsed=None if schema coercion failed — validate manually
        return ListingAnalysis.model_validate_json(response.text)


class AnthropicAnalyzer:
    """Listing analysis via the Anthropic API with structured outputs."""

    provider = "anthropic"

    def __init__(self, config: Optional[dict] = None):
        llm_cfg = (config or {}).get("llm", {})
        self.model = os.environ.get(
            "ANTHROPIC_MODEL", llm_cfg.get("anthropic_model", DEFAULT_ANTHROPIC_MODEL)
        )
        self.max_tokens = llm_cfg.get("max_tokens", DEFAULT_MAX_TOKENS)
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic()
        return self._client

    async def analyze(self, listing, model_prediction) -> ListingAnalysis:
        client = self._get_client()
        response = await client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": build_user_message(listing, model_prediction),
            }],
            output_format=ListingAnalysis,
        )
        return response.parsed_output


def create_analyzer(config: Optional[dict] = None):
    """
    Pick an analyzer based on which API key is configured.

    Gemini is checked first (free tier), then Anthropic. When neither key is
    set, returns a disabled GeminiAnalyzer so callers can still read
    `.enabled` and `.model`.
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return GeminiAnalyzer(config)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicAnalyzer(config)
    return GeminiAnalyzer(config)


# Backwards-compatible alias (original single-provider implementation)
ListingAnalyzer = AnthropicAnalyzer
