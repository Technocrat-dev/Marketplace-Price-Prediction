"""
Pydantic schemas for the Mercari Price Prediction API.

Defines request/response models for:
- Single product prediction
- Batch prediction
- Health check
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# =========================================================================
# Request Schemas
# =========================================================================

class PredictionRequest(BaseModel):
    """Request schema for a single price prediction."""
    
    name: str = Field(
        ..., 
        description="Product name",
        examples=["Nike Air Max 90"],
    )
    item_description: str = Field(
        default="",
        description="Product description",
        examples=["Classic sneakers, barely worn, great condition"],
    )
    category_name: str = Field(
        default="",
        description="Full category path (e.g. 'Women/Tops & Blouses/Blouse')",
        examples=["Men/Shoes/Athletic"],
    )
    brand_name: str = Field(
        default="unknown",
        description="Brand name",
        examples=["Nike"],
    )
    item_condition_id: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Item condition (1=New, 5=Poor)",
    )
    shipping: int = Field(
        default=0,
        ge=0,
        le=1,
        description="0=buyer pays shipping, 1=seller pays",
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    
    items: List[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of products to predict prices for (max 100)",
    )


# =========================================================================
# Response Schemas
# =========================================================================

class PredictionResponse(BaseModel):
    """Response schema for a single price prediction."""
    
    predicted_price: float = Field(
        description="Predicted price in USD",
    )
    predicted_log_price: float = Field(
        description="Raw model output (log1p scale)",
    )
    confidence_range: Dict[str, float] = Field(
        description="Approximate price range (low/high)",
    )
    input_summary: Dict[str, str] = Field(
        description="Summary of processed input features",
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse]
    count: int
    model_version: str


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str
    model_version: str
    model_loaded: bool
    mongodb_status: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str
    detail: Optional[str] = None
