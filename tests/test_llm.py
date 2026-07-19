"""
Unit tests for the Claude-powered listing analysis.

Tests:
- Model/LLM price comparison logic
- /predict/analyze endpoint (disabled, success, upstream failure)

The Anthropic API is never called — the analyzer is mocked.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

import src.serving.app as app_module
from src.serving.app import app, server
from src.serving.llm import ListingAnalysis, ListingAnalyzer, build_comparison
from src.serving.schemas import PredictionResponse


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def client():
    return TestClient(app)


SAMPLE_ANALYSIS = ListingAnalysis(
    llm_estimated_price=25.0,
    price_reasoning="Nike sneakers in good condition typically resell around $25.",
    listing_score=6,
    strengths=["Brand is specified"],
    improvements=["Add size information", "Describe the condition in detail"],
    suggested_title="Nike Air Max 90 - Men's Size 10, Good Condition",
)

SAMPLE_PREDICTION = PredictionResponse(
    predicted_price=22.5,
    predicted_log_price=3.16,
    confidence_range={"low": 16.7, "high": 30.4},
    input_summary={},
)


@pytest.fixture
def mock_loaded_server(monkeypatch):
    """Mark the server loaded and stub out model inference."""
    monkeypatch.setattr(server, "is_loaded", True)
    monkeypatch.setattr(server, "config", {
        "serving": {"model_version": "1.0.0"},
        "data": {"max_name_len": 10, "max_desc_len": 75},
        "database": {"uri": "mongodb://localhost:27017", "name": "test_db"},
    })
    monkeypatch.setattr(server, "predict", lambda req: SAMPLE_PREDICTION)
    yield server


@pytest.fixture
def mock_analyzer(monkeypatch):
    """Replace the module-level analyzer with an enabled mock."""
    analyzer = ListingAnalyzer()
    monkeypatch.setattr(type(analyzer), "enabled", property(lambda self: True))
    analyzer.analyze = AsyncMock(return_value=SAMPLE_ANALYSIS)
    monkeypatch.setattr(app_module, "analyzer", analyzer)
    return analyzer


LISTING = {
    "name": "Nike Air Max 90",
    "item_description": "Barely worn",
    "brand_name": "Nike",
    "item_condition_id": 3,
    "shipping": 0,
}


# =========================================================================
# Comparison Logic
# =========================================================================

class TestBuildComparison:

    def test_close_agreement(self):
        result = build_comparison(20.0, 21.0)
        assert result["agreement"] == "close"
        assert result["delta"] == 1.0
        assert result["delta_pct"] == 5.0

    def test_moderate_agreement(self):
        result = build_comparison(20.0, 26.0)
        assert result["agreement"] == "moderate"

    def test_divergent(self):
        result = build_comparison(20.0, 50.0)
        assert result["agreement"] == "divergent"
        assert result["delta_pct"] == 150.0

    def test_llm_below_model(self):
        result = build_comparison(50.0, 20.0)
        assert result["delta"] == -30.0
        assert result["agreement"] == "divergent"

    def test_zero_model_price_does_not_crash(self):
        result = build_comparison(0.0, 10.0)
        assert result["delta_pct"] == 0.0


# =========================================================================
# /predict/analyze Endpoint
# =========================================================================

class TestAnalyzeEndpoint:

    def test_503_when_model_not_loaded(self, client, monkeypatch):
        monkeypatch.setattr(server, "is_loaded", False)
        response = client.post("/predict/analyze", json=LISTING)
        assert response.status_code == 503

    def test_503_when_llm_disabled(self, client, mock_loaded_server, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(app_module, "analyzer", ListingAnalyzer())
        response = client.post("/predict/analyze", json=LISTING)
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]

    def test_successful_analysis(self, client, mock_loaded_server, mock_analyzer):
        response = client.post("/predict/analyze", json=LISTING)
        assert response.status_code == 200
        data = response.json()

        assert data["prediction"]["predicted_price"] == 22.5
        assert data["ai_analysis"]["llm_estimated_price"] == 25.0
        assert data["ai_analysis"]["listing_score"] == 6
        assert len(data["ai_analysis"]["improvements"]) == 2
        assert data["comparison"]["agreement"] == "close"
        assert data["comparison"]["model_price"] == 22.5
        assert data["comparison"]["llm_price"] == 25.0

    def test_analyzer_receives_listing_and_prediction(
        self, client, mock_loaded_server, mock_analyzer
    ):
        client.post("/predict/analyze", json=LISTING)
        mock_analyzer.analyze.assert_awaited_once()
        listing_arg, prediction_arg = mock_analyzer.analyze.await_args.args
        assert listing_arg.name == "Nike Air Max 90"
        assert prediction_arg.predicted_price == 22.5

    def test_502_when_llm_call_fails(self, client, mock_loaded_server, mock_analyzer):
        mock_analyzer.analyze = AsyncMock(side_effect=RuntimeError("api down"))
        response = client.post("/predict/analyze", json=LISTING)
        assert response.status_code == 502

    def test_health_reports_llm_flag(self, client, mock_loaded_server, mock_analyzer):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["llm_enabled"] is True
