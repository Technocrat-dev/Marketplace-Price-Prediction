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
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict the price of a single product."""
    if not server.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        return server.predict(request)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict prices for multiple products (max 100)."""
    if not server.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = server.predict_batch(request.items)
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
    # Check MongoDB
    mongo_status = "not_configured"
    try:
        from src.db.mongo import MongoDBClient
        client = MongoDBClient(
            uri=server.config["database"]["uri"],
            db_name=server.config["database"]["name"],
        )
        client.connect()
        health = client.health_check()
        mongo_status = health["status"]
        client.close()
    except Exception:
        mongo_status = "unavailable"
    
    return HealthResponse(
        status="healthy" if server.is_loaded else "degraded",
        model_version=server.config["serving"]["model_version"] if server.config else "unknown",
        model_loaded=server.is_loaded,
        mongodb_status=mongo_status,
        timestamp=datetime.now(timezone.utc),
    )
