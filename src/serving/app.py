"""
FastAPI application for the Mercari Price Prediction Engine.

Endpoints:
    POST /predict       — Predict price for a single product
    POST /predict/batch — Predict prices for multiple products
    GET  /health        — Health check (model + database status)

Usage:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.data.preprocess import (
    Vocabulary,
    CategoricalEncoder,
    clean_text,
    parse_category,
)
from src.models.multimodal import MercariPricePredictor
from src.serving.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    LossCurvePoint,
    ProductResponse,
    ProductSearchResponse,
    ProductStatsResponse,
    CategoryStat,
    BrandStat,
    RecentPredictionItem,
    RecentPredictionsResponse,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Global State — loaded at startup
# =========================================================================

class ModelServer:
    """Holds the loaded model and preprocessing artifacts."""
    
    def __init__(self):
        self.model: Optional[MercariPricePredictor] = None
        self.name_vocab: Optional[Vocabulary] = None
        self.desc_vocab: Optional[Vocabulary] = None
        self.cat_encoder: Optional[CategoricalEncoder] = None
        self.metadata: Optional[Dict] = None
        self.config: Optional[Dict] = None
        self.device = torch.device("cpu")  # Serve on CPU for consistency
        self.is_loaded = False
        self.training_results: Optional[Dict] = None
        self.db_client = None
        self.product_repo = None
        self.prediction_repo = None
    
    def load(self, config_path: str = "config/config.yaml") -> None:
        """Load model, vocabularies, and encoders from disk."""
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        cfg = self.config
        data_dir = Path(cfg["paths"]["processed_data"])
        checkpoint_dir = Path(cfg["paths"]["checkpoints"])
        
        # Load metadata
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        # Load vocabularies
        self.name_vocab = Vocabulary.load(str(data_dir / "name_vocab.json"))
        self.desc_vocab = Vocabulary.load(str(data_dir / "desc_vocab.json"))
        
        # Load categorical encoder
        self.cat_encoder = CategoricalEncoder.load(str(data_dir / "cat_encoder.json"))
        
        # Build and load model
        self.model = MercariPricePredictor(
            name_vocab_size=self.metadata["name_vocab_size"],
            desc_vocab_size=self.metadata["desc_vocab_size"],
            cat_dims=self.metadata["cat_sizes"],
        )
        
        checkpoint_path = checkpoint_dir / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from epoch {checkpoint['epoch']}, "
                        f"val_loss={checkpoint['best_val_loss']:.4f}")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, using random weights")
        
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        
        # Load training results if available
        results_path = Path(cfg["paths"]["outputs"]) / "training_results.json"
        if results_path.exists():
            with open(results_path) as f:
                self.training_results = json.load(f)
            logger.info("Loaded training results")
        
        # Connect to MongoDB (non-fatal if unavailable)
        try:
            from src.db.mongo import MongoDBClient, ProductRepository, PredictionRepository
            self.db_client = MongoDBClient(
                uri=cfg["database"]["uri"],
                db_name=cfg["database"]["name"],
            )
            self.db_client.connect()
            self.product_repo = ProductRepository(self.db_client.db)
            self.prediction_repo = PredictionRepository(self.db_client.db)
            logger.info("MongoDB connected for product/prediction queries")
        except Exception as e:
            logger.warning(f"MongoDB not available for queries: {e}")
        
        logger.info(f"Model server ready — {self.model.count_parameters():,} parameters")
    
    def preprocess(self, request: PredictionRequest) -> Dict[str, torch.Tensor]:
        """Convert a prediction request into model-ready tensors."""
        cfg = self.config
        max_name_len = cfg["data"]["max_name_len"]
        max_desc_len = cfg["data"]["max_desc_len"]
        
        # Clean text
        name_clean = clean_text(request.name)
        desc_clean = clean_text(request.item_description)
        
        # Tokenize
        name_seq = self.name_vocab.encode(name_clean, max_name_len)
        desc_seq = self.desc_vocab.encode(desc_clean, max_desc_len)
        
        # Parse and encode category
        main_cat, sub_cat1, sub_cat2 = parse_category(request.category_name)
        brand = request.brand_name.lower().strip() if request.brand_name else "unknown"
        condition = max(1, min(5, request.item_condition_id))
        
        cat_columns = ["main_cat", "sub_cat1", "sub_cat2", "brand_name", "item_condition_id"]
        cat_values = [main_cat, sub_cat1, sub_cat2, brand, str(condition)]
        
        categoricals = []
        for col, val in zip(cat_columns, cat_values):
            mapping = self.cat_encoder.encoders.get(col, {})
            categoricals.append(mapping.get(val, 0))  # 0 = unknown
        
        # Build tensors (batch_size=1)
        return {
            "name_seq": torch.tensor([name_seq], dtype=torch.long),
            "desc_seq": torch.tensor([desc_seq], dtype=torch.long),
            "categoricals": torch.tensor([categoricals], dtype=torch.long),
            "shipping": torch.tensor([float(request.shipping)], dtype=torch.float32),
        }
    
    @torch.no_grad()
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Run inference for a single product."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Preprocess
        tensors = self.preprocess(request)
        
        # Move to device
        tensors = {k: v.to(self.device) for k, v in tensors.items()}
        
        # Inference
        log_price = self.model(tensors).item()
        predicted_price = float(np.expm1(log_price))
        predicted_price = max(0, predicted_price)  # Clamp to non-negative
        
        # Confidence range (approximate ±1 std in log space)
        price_low = max(0, float(np.expm1(log_price - 0.3)))
        price_high = float(np.expm1(log_price + 0.3))
        
        # Parse category for summary
        main_cat, sub_cat1, sub_cat2 = parse_category(request.category_name)
        
        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            predicted_log_price=round(log_price, 4),
            confidence_range={
                "low": round(price_low, 2),
                "high": round(price_high, 2),
            },
            input_summary={
                "name": clean_text(request.name),
                "brand": (request.brand_name or "unknown").lower(),
                "category": f"{main_cat}/{sub_cat1}/{sub_cat2}",
                "condition": str(request.item_condition_id),
                "shipping": "seller pays" if request.shipping else "buyer pays",
            },
        )
    
    @torch.no_grad()
    def predict_batch(self, requests: list[PredictionRequest]) -> list[PredictionResponse]:
        """Run inference for multiple products efficiently."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Preprocess all items
        all_tensors = [self.preprocess(req) for req in requests]
        
        # Stack into batches
        batch = {
            key: torch.cat([t[key] for t in all_tensors], dim=0).to(self.device)
            for key in all_tensors[0].keys()
        }
        
        # Batch inference
        log_prices = self.model(batch).cpu().numpy()
        
        # Build responses
        responses = []
        for i, req in enumerate(requests):
            log_price = float(log_prices[i])
            predicted_price = max(0, float(np.expm1(log_price)))
            price_low = max(0, float(np.expm1(log_price - 0.3)))
            price_high = float(np.expm1(log_price + 0.3))
            
            main_cat, sub_cat1, sub_cat2 = parse_category(req.category_name)
            
            responses.append(PredictionResponse(
                predicted_price=round(predicted_price, 2),
                predicted_log_price=round(log_price, 4),
                confidence_range={
                    "low": round(price_low, 2),
                    "high": round(price_high, 2),
                },
                input_summary={
                    "name": clean_text(req.name),
                    "brand": (req.brand_name or "unknown").lower(),
                    "category": f"{main_cat}/{sub_cat1}/{sub_cat2}",
                    "condition": str(req.item_condition_id),
                    "shipping": "seller pays" if req.shipping else "buyer pays",
                },
            ))
        
        return responses


# =========================================================================
# FastAPI Application
# =========================================================================

# Global model server instance
server = ModelServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Loading model server...")
    try:
        server.load()
        logger.info("Model server ready!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    logger.info("Shutting down model server")


app = FastAPI(
    title="Mercari Price Prediction API",
    description="Predict product prices using a multimodal deep learning model "
                "trained on the Mercari marketplace dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, status, and latency."""
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"latency={latency_ms:.1f}ms "
        f"client={request.client.host if request.client else 'unknown'}"
    )
    
    response.headers["X-Response-Time"] = f"{latency_ms:.1f}ms"
    return response


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("60/minute")
async def predict(request: Request, prediction: PredictionRequest):
    """Predict the price of a single product."""
    if not server.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = server.predict(prediction)
        
        # Persist prediction to MongoDB (non-blocking, non-fatal)
        if server.prediction_repo:
            try:
                server.prediction_repo.insert_one({
                    "product_name": prediction.name,
                    "brand": prediction.brand_name or "unknown",
                    "category": prediction.category_name,
                    "condition": prediction.item_condition_id,
                    "shipping": prediction.shipping,
                    "predicted_price": result.predicted_price,
                    "predicted_log_price": result.predicted_log_price,
                    "confidence_low": result.confidence_range["low"],
                    "confidence_high": result.confidence_range["high"],
                    "model_version": server.config["serving"]["model_version"],
                })
            except Exception as db_err:
                logger.warning(f"Failed to persist prediction: {db_err}")
        
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
@limiter.limit("10/minute")
async def predict_batch(request: Request, batch: BatchPredictionRequest):
    """Predict prices for multiple products (max 100)."""
    if not server.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = server.predict_batch(batch.items)
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            model_version=server.config["serving"]["model_version"],
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health."""
    mongo_status = "not_configured"
    if server.db_client:
        try:
            health = server.db_client.health_check()
            mongo_status = health["status"]
        except Exception:
            mongo_status = "unavailable"
    
    return HealthResponse(
        status="healthy" if server.is_loaded else "degraded",
        model_version=server.config["serving"]["model_version"] if server.config else "unknown",
        model_loaded=server.is_loaded,
        mongodb_status=mongo_status,
        timestamp=datetime.now(timezone.utc),
    )


# =========================================================================
# Model Info Endpoint
# =========================================================================

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model architecture, training metrics, and config."""
    if not server.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = server.training_results
    if not results:
        raise HTTPException(status_code=404, detail="Training results not found")
    
    cfg = server.config
    
    # Build loss curve from training results
    train_losses = results.get("train_losses", [])
    val_losses = results.get("val_losses", [])
    loss_curve = [
        LossCurvePoint(epoch=i + 1, train_loss=tl, val_loss=vl)
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses))
    ]
    
    # Build config summary
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    config_summary = {
        "Architecture": "BiLSTM + Embeddings + Fusion MLP",
        "Text Encoder": f"{model_cfg.get('text_embed_dim', '?')}d embeddings → {model_cfg.get('text_hidden_dim', '?')}d BiLSTM → {model_cfg.get('text_hidden_dim', 0) * 2}d output",
        "Tabular Encoder": f"{model_cfg.get('cat_embed_dim', '?')}d embeddings → {model_cfg.get('tabular_hidden_dim', '?')}d FC",
        "Fusion Head": " → ".join(str(d) for d in model_cfg.get('fusion_hidden_dims', [])) + " → 1",
        "Optimizer": f"Adam (lr={training_cfg.get('learning_rate', '?')}, weight_decay={training_cfg.get('weight_decay', '?')})",
        "Batch Size": str(training_cfg.get('batch_size', '?')),
        "Epochs": str(results.get('total_epochs', '?')),
        "Early Stopping": f"Patience {training_cfg.get('patience', '?')}",
        "Gradient Clipping": str(training_cfg.get('grad_clip', '?')),
        "LR Scheduler": f"ReduceLROnPlateau (factor={training_cfg.get('scheduler_factor', '?')}, patience={training_cfg.get('scheduler_patience', '?')})",
    }
    
    metadata = server.metadata or {}
    
    return ModelInfoResponse(
        model_version=cfg["serving"]["model_version"],
        model_parameters=results.get("model_parameters", 0),
        test_metrics=results.get("test_metrics", {}),
        val_metrics=results.get("val_metrics", {}),
        loss_curve=loss_curve,
        best_epoch=results.get("best_epoch", 0),
        total_epochs=results.get("total_epochs", 0),
        train_size=metadata.get("train_size", 0),
        val_size=metadata.get("val_size", 0),
        test_size=metadata.get("test_size", 0),
        config_summary=config_summary,
    )


# =========================================================================
# Product Endpoints (Explore page)
# =========================================================================

@app.get("/products/search", response_model=ProductSearchResponse)
async def search_products(q: str = "", limit: int = 20, offset: int = 0):
    """Search products in the MongoDB catalog."""
    if not server.product_repo:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    
    try:
        if q.strip():
            results = server.product_repo.search(q, limit=limit + offset)
            results = results[offset:offset + limit]
        else:
            # No query — return recent products
            cursor = server.product_repo.collection.find().limit(limit).skip(offset)
            results = list(cursor)
        
        products = []
        for doc in results:
            products.append(ProductResponse(
                product_id=str(doc.get("product_id", doc.get("_id", ""))),
                name=doc.get("name", ""),
                brand_name=doc.get("brand_name", "unknown"),
                category_name=doc.get("category_name", ""),
                main_category=doc.get("main_category", ""),
                item_condition_id=doc.get("item_condition_id", 3),
                shipping=doc.get("shipping", 0),
                price=doc.get("price", 0.0),
            ))
        
        total = server.product_repo.count()
        return ProductSearchResponse(products=products, total=total, query=q)
    except Exception as e:
        logger.error(f"Product search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/stats", response_model=ProductStatsResponse)
async def product_stats():
    """Get aggregate product catalog statistics."""
    if not server.product_repo:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    
    try:
        total = server.product_repo.count()
        cat_stats = server.product_repo.get_category_stats()
        brand_stats = server.product_repo.get_brand_stats()
        
        # Compute average price
        pipeline = [{"$group": {"_id": None, "avg": {"$avg": "$price"}}}]
        avg_result = list(server.product_repo.collection.aggregate(pipeline))
        avg_price = avg_result[0]["avg"] if avg_result else 0.0
        
        return ProductStatsResponse(
            total_products=total,
            total_brands=len(brand_stats),
            total_categories=len(cat_stats),
            avg_price=round(avg_price, 2),
            category_distribution=[
                CategoryStat(category=s["_id"] or "unknown", count=s["count"])
                for s in cat_stats
            ],
            top_brands=[
                BrandStat(brand=s["_id"] or "unknown", count=s["count"])
                for s in brand_stats
            ],
        )
    except Exception as e:
        logger.error(f"Product stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Prediction History Endpoint
# =========================================================================

@app.get("/predictions/recent", response_model=RecentPredictionsResponse)
async def recent_predictions(limit: int = 20):
    """Get the most recent predictions."""
    if not server.prediction_repo:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    
    try:
        results = server.prediction_repo.find_latest(limit=limit)
        predictions = []
        for doc in results:
            predictions.append(RecentPredictionItem(
                product_name=doc.get("product_name", doc.get("product_id", "Unknown")),
                brand=doc.get("brand", "unknown"),
                predicted_price=doc.get("predicted_price", 0.0),
                confidence_low=doc.get("confidence_low", 0.0),
                confidence_high=doc.get("confidence_high", 0.0),
                predicted_at=str(doc.get("predicted_at", "")),
            ))
        
        total = server.prediction_repo.count()
        return RecentPredictionsResponse(predictions=predictions, total=total)
    except Exception as e:
        logger.error(f"Recent predictions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
