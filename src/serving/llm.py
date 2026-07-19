"""
Claude-powered listing analysis for the Mercari Price Prediction Engine.

Given a product listing and the ML model's price prediction, asks Claude for:
- an independent (zero-shot) price estimate with reasoning
- a quality score and concrete critique of the listing itself

The two estimates are compared so users can see when the trained model and
the LLM agree — and when they don't.

Requires the ANTHROPIC_API_KEY environment variable; the endpoint degrades
gracefully (503) when it is not configured.
"""

import logging
import os
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-8"
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


class ListingAnalysis(BaseModel):
    """Structured output schema for Claude's listing analysis."""

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


class ListingAnalyzer:
    """Wraps the Anthropic API for listing analysis with structured outputs."""

    def __init__(self, config: Optional[dict] = None):
        llm_cfg = (config or {}).get("llm", {})
        self.model = os.environ.get("ANTHROPIC_MODEL", llm_cfg.get("model", DEFAULT_MODEL))
        self.max_tokens = llm_cfg.get("max_tokens", DEFAULT_MAX_TOKENS)
        self._client = None

    @property
    def enabled(self) -> bool:
        """Analysis is available only when an API key is configured."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic()
        return self._client

    async def analyze(self, listing, model_prediction) -> ListingAnalysis:
        """
        Ask Claude to analyze a listing.

        Args:
            listing: PredictionRequest with the raw listing fields
            model_prediction: PredictionResponse from the trained model

        Returns:
            Validated ListingAnalysis instance.
        """
        condition_labels = {1: "New with tags", 2: "New without tags",
                            3: "Good", 4: "Fair", 5: "Poor"}
        user_message = f"""Listing:
- Title: {listing.name}
- Description: {listing.item_description or "(none)"}
- Category: {listing.category_name or "(none)"}
- Brand: {listing.brand_name or "(none)"}
- Condition: {condition_labels.get(listing.item_condition_id, "Unknown")}
- Shipping: {"seller pays" if listing.shipping else "buyer pays"}

ML model estimate: ${model_prediction.predicted_price:.2f} \
(range ${model_prediction.confidence_range["low"]:.2f}-\
${model_prediction.confidence_range["high"]:.2f})

Analyze this listing."""

        client = self._get_client()
        response = await client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            output_format=ListingAnalysis,
        )
        return response.parsed_output
