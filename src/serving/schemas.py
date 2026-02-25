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


# =========================================================================
# Model Info
# =========================================================================

class LossCurvePoint(BaseModel):
    """A single epoch's loss values."""
    epoch: int
    train_loss: float
    val_loss: float


class ModelInfoResponse(BaseModel):
    """Response schema for /model/info endpoint."""
    
    model_version: str
    model_parameters: int
    architecture: str = "BiLSTM + Embeddings + Fusion MLP"
    
    # Training metrics
    test_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    loss_curve: List[LossCurvePoint]
    best_epoch: int
    total_epochs: int
    
    # Data stats
    train_size: int
    val_size: int
    test_size: int
    
    # Config summary
    config_summary: Dict[str, str]


# =========================================================================
# Product Endpoints
# =========================================================================

class ProductResponse(BaseModel):
    """A single product from the catalog."""
    
    product_id: str
    name: str
    brand_name: str = "unknown"
    category_name: str = ""
    main_category: str = ""
    item_condition_id: int = 3
    shipping: int = 0
    price: float = 0.0


class ProductSearchResponse(BaseModel):
    """Response for product search."""
    
    products: List[ProductResponse]
    total: int
    query: str


class CategoryStat(BaseModel):
    """Product count per category."""
    category: str
    count: int


class BrandStat(BaseModel):
    """Product count per brand."""
    brand: str
    count: int


class ProductStatsResponse(BaseModel):
    """Aggregate stats about the product catalog."""
    
    total_products: int
    total_brands: int
    total_categories: int
    avg_price: float
    category_distribution: List[CategoryStat]
    top_brands: List[BrandStat]


# =========================================================================
# Prediction History
# =========================================================================

class RecentPredictionItem(BaseModel):
    """A single prediction from history."""
    
    product_name: str
    brand: str
    predicted_price: float
    confidence_low: float
    confidence_high: float
    predicted_at: str


class RecentPredictionsResponse(BaseModel):
    """Response for recent predictions."""
    
    predictions: List[RecentPredictionItem]
    total: int
