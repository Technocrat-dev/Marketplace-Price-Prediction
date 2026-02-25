"""
Unit tests for the FastAPI serving layer.

Tests:
- Single prediction endpoint
- Batch prediction endpoint
- Health check endpoint
- Error handling (model not loaded, invalid input)
"""

import pytest
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
        assert data["model_loaded"] is False


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


# =========================================================================
# Model Info Endpoint Tests
# =========================================================================

class TestModelInfoEndpoint:
    """Tests for the GET /model/info endpoint."""
    
    def test_model_info_returns_503_when_not_loaded(self, client):
        """Should return 503 when model is not loaded."""
        original = server.is_loaded
        server.is_loaded = False
        response = client.get("/model/info")
        server.is_loaded = original
        assert response.status_code == 503
    
    def test_model_info_returns_404_without_training_results(self, client, mock_loaded_server):
        """Should return 404 when training results are missing."""
        original = server.training_results
        server.training_results = None
        response = client.get("/model/info")
        server.training_results = original
        assert response.status_code == 404


# =========================================================================
# CSV Predict Endpoint Tests
# =========================================================================

class TestCSVPredictEndpoint:
    """Tests for the POST /predict/csv endpoint."""
    
    def test_csv_returns_503_when_not_loaded(self, client):
        """Should return 503 when model is not loaded."""
        original = server.is_loaded
        server.is_loaded = False
        import io
        csv_content = b"name\nTest Product"
        response = client.post(
            "/predict/csv",
            files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
        )
        server.is_loaded = original
        assert response.status_code == 503
    
    def test_csv_rejects_non_csv(self, client, mock_loaded_server):
        """Should reject non-CSV files."""
        import io
        response = client.post(
            "/predict/csv",
            files={"file": ("test.txt", io.BytesIO(b"data"), "text/plain")},
        )
        assert response.status_code == 400


# =========================================================================
# Explain Endpoint Tests
# =========================================================================

class TestExplainEndpoint:
    """Tests for the POST /predict/explain endpoint."""
    
    def test_explain_returns_503_when_not_loaded(self, client):
        """Should return 503 when model is not loaded."""
        original = server.is_loaded
        server.is_loaded = False
        response = client.post("/predict/explain", json={
            "name": "Test Product",
        })
        server.is_loaded = original
        assert response.status_code == 503
    
    def test_explain_requires_name(self, client):
        """Name is a required field."""
        response = client.post("/predict/explain", json={
            "item_description": "test",
        })
        assert response.status_code == 422


# =========================================================================
# Product Search Endpoint Tests
# =========================================================================

class TestProductSearchEndpoint:
    """Tests for the GET /products/search endpoint."""
    
    def test_search_returns_503_without_mongo(self, client):
        """Should return 503 when MongoDB is not available."""
        original = server.product_repo
        server.product_repo = None
        response = client.get("/products/search")
        server.product_repo = original
        assert response.status_code == 503
    
    def test_search_rejects_limit_over_100(self, client):
        """limit must be <= 100."""
        response = client.get("/products/search?limit=200")
        assert response.status_code == 422
    
    def test_search_rejects_negative_offset(self, client):
        """offset must be >= 0."""
        response = client.get("/products/search?offset=-1")
        assert response.status_code == 422


# =========================================================================
# Product Stats Endpoint Tests
# =========================================================================

class TestProductStatsEndpoint:
    """Tests for the GET /products/stats endpoint."""
    
    def test_stats_returns_503_without_mongo(self, client):
        """Should return 503 when MongoDB is not available."""
        original = server.product_repo
        server.product_repo = None
        response = client.get("/products/stats")
        server.product_repo = original
        assert response.status_code == 503


# =========================================================================
# Recent Predictions Endpoint Tests
# =========================================================================

class TestRecentPredictionsEndpoint:
    """Tests for the GET /predictions/recent endpoint."""
    
    def test_recent_returns_503_without_mongo(self, client):
        """Should return 503 when MongoDB is not available."""
        original = server.prediction_repo
        server.prediction_repo = None
        response = client.get("/predictions/recent")
        server.prediction_repo = original
        assert response.status_code == 503
    
    def test_recent_rejects_limit_over_100(self, client):
        """limit must be <= 100."""
        response = client.get("/predictions/recent?limit=200")
        assert response.status_code == 422
