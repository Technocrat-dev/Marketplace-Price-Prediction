"""
Unit tests for the FastAPI serving layer.

Tests:
- Single prediction endpoint
- Batch prediction endpoint
- Health check endpoint
- Error handling (model not loaded, invalid input)
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.serving.app import app, server
from src.serving.schemas import PredictionRequest


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_loaded_server():
    """Mock the model server as loaded with a fake model."""
    original_loaded = server.is_loaded
    original_config = server.config
    
    server.is_loaded = True
    server.config = {
        "serving": {"model_version": "1.0.0"},
        "data": {"max_name_len": 10, "max_desc_len": 75},
        "database": {"uri": "mongodb://localhost:27017", "name": "test_db"},
    }
    
    yield server
    
    server.is_loaded = original_loaded
    server.config = original_config


# =========================================================================
# Health Check Tests
# =========================================================================

class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "mongodb_status" in data
        assert "timestamp" in data
    
    def test_health_degraded_when_model_not_loaded(self, client):
        """When model isn't loaded, status should be degraded."""
        original = server.is_loaded
        server.is_loaded = False
        server.config = server.config or {
            "serving": {"model_version": "unknown"},
            "database": {"uri": "mongodb://localhost:27017", "name": "test_db"},
        }
        response = client.get("/health")
        data = response.json()
        server.is_loaded = original
        # Status depends on model load state
        assert data["model_loaded"] == False


# =========================================================================
# Predict Endpoint Tests
# =========================================================================

class TestPredictEndpoint:
    """Tests for the POST /predict endpoint."""
    
    def test_predict_returns_503_when_model_not_loaded(self, client):
        """Should return 503 when model is not loaded."""
        original = server.is_loaded
        server.is_loaded = False
        response = client.post("/predict", json={
            "name": "Test Product",
            "item_description": "A test item",
            "category_name": "Electronics/Phones/Cases",
            "brand_name": "TestBrand",
            "item_condition_id": 3,
            "shipping": 0,
        })
        server.is_loaded = original
        assert response.status_code == 503
    
    def test_predict_requires_name(self, client):
        """Name is a required field."""
        response = client.post("/predict", json={
            "item_description": "test",
        })
        assert response.status_code == 422  # Validation error
    
    def test_predict_validates_condition_range(self, client):
        """item_condition_id must be 1-5."""
        response = client.post("/predict", json={
            "name": "Test",
            "item_condition_id": 10,
        })
        assert response.status_code == 422
    
    def test_predict_validates_shipping_range(self, client):
        """shipping must be 0 or 1."""
        response = client.post("/predict", json={
            "name": "Test",
            "shipping": 5,
        })
        assert response.status_code == 422
    
    def test_predict_with_defaults(self, client):
        """Should accept request with only required fields."""
        response = client.post("/predict", json={
            "name": "Simple Product",
        })
        # 503 because model isn't loaded, but request was valid
        assert response.status_code in [200, 503]


# =========================================================================
# Batch Predict Endpoint Tests
# =========================================================================

class TestBatchPredictEndpoint:
    """Tests for the POST /predict/batch endpoint."""
    
    def test_batch_returns_503_when_model_not_loaded(self, client):
        """Should return 503 when model is not loaded."""
        original = server.is_loaded
        server.is_loaded = False
        response = client.post("/predict/batch", json={
            "items": [{"name": "Test Product"}],
        })
        server.is_loaded = original
        assert response.status_code == 503
    
    def test_batch_requires_items(self, client):
        """items field is required."""
        response = client.post("/predict/batch", json={})
        assert response.status_code == 422
    
    def test_batch_rejects_empty_items(self, client):
        """items list cannot be empty."""
        response = client.post("/predict/batch", json={
            "items": [],
        })
        assert response.status_code == 422
    
    def test_batch_rejects_over_100_items(self, client):
        """items list cannot exceed 100."""
        items = [{"name": f"Product {i}"} for i in range(101)]
        response = client.post("/predict/batch", json={
            "items": items,
        })
        assert response.status_code == 422


# =========================================================================
# Schema Validation Tests
# =========================================================================

class TestSchemaValidation:
    """Tests for Pydantic request/response schema validation."""
    
    def test_prediction_request_defaults(self):
        """Default values should be set correctly."""
        req = PredictionRequest(name="Test")
        assert req.item_description == ""
        assert req.category_name == ""
        assert req.brand_name == "unknown"
        assert req.item_condition_id == 3
        assert req.shipping == 0
    
    def test_prediction_request_all_fields(self):
        """All fields should be accepted."""
        req = PredictionRequest(
            name="Nike Air Max",
            item_description="Great shoes",
            category_name="Men/Shoes/Athletic",
            brand_name="Nike",
            item_condition_id=1,
            shipping=1,
        )
        assert req.name == "Nike Air Max"
        assert req.shipping == 1
    
    def test_prediction_request_condition_validation(self):
        """Condition must be 1-5."""
        with pytest.raises(Exception):
            PredictionRequest(name="Test", item_condition_id=0)
        with pytest.raises(Exception):
            PredictionRequest(name="Test", item_condition_id=6)
    
    def test_prediction_request_shipping_validation(self):
        """Shipping must be 0 or 1."""
        with pytest.raises(Exception):
            PredictionRequest(name="Test", shipping=2)
