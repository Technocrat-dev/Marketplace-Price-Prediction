"""
Unit tests for the MongoDB integration layer.

Tests run against a real local MongoDB instance using a temporary test database
that is dropped after each test session.

Skip condition: If MongoDB is not running, all tests are skipped gracefully.
"""

import pytest

# Try to import and connect — skip all tests if MongoDB unavailable
try:
    from pymongo import MongoClient
    _client = MongoClient("localhost", 27017, serverSelectionTimeoutMS=2000)
    _client.admin.command("ping")
    _client.close()
    MONGO_AVAILABLE = True
except Exception:
    MONGO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MONGO_AVAILABLE, reason="MongoDB not running on localhost:27017"
)

from src.db.mongo import MongoDBClient, ProductRepository, PredictionRepository


# =========================================================================
# Fixtures
# =========================================================================

TEST_DB_NAME = "mercari_test_db"


@pytest.fixture(scope="module")
def db_client():
    """Create a test database connection, drop it when done."""
    client = MongoDBClient(db_name=TEST_DB_NAME)
    client.connect()
    yield client
    client.drop_database()
    client.close()


@pytest.fixture(autouse=True)
def clean_collections(db_client):
    """Clean all collections before each test for isolation."""
    db_client.db["products"].delete_many({})
    db_client.db["predictions"].delete_many({})


@pytest.fixture
def product_repo(db_client):
    return ProductRepository(db_client.db)


@pytest.fixture
def prediction_repo(db_client):
    return PredictionRepository(db_client.db)


@pytest.fixture
def sample_product():
    return {
        "product_id": "test_001",
        "name": "Nike Air Max 90",
        "item_description": "Classic sneakers, barely worn",
        "brand_name": "nike",
        "category_name": "Men/Shoes/Athletic",
        "main_category": "men",
        "sub_category_1": "shoes",
        "sub_category_2": "athletic",
        "item_condition_id": 2,
        "shipping": 1,
        "price": 75.0,
    }


@pytest.fixture
def sample_prediction():
    return {
        "product_id": "test_001",
        "predicted_price": 68.50,
        "predicted_log_price": 4.24,
        "actual_price": 75.0,
        "model_version": "1.0.0",
        "features_used": {
            "name": "nike air max 90",
            "brand": "nike",
            "category": "men/shoes/athletic",
        },
    }


# =========================================================================
# MongoDBClient Tests
# =========================================================================

class TestMongoDBClient:
    
    def test_connection(self, db_client):
        assert db_client.db is not None
    
    def test_health_check(self, db_client):
        health = db_client.health_check()
        assert health["status"] == "healthy"
        assert health["database"] == TEST_DB_NAME
    
    def test_health_check_shows_collections(self, db_client):
        health = db_client.health_check()
        assert "documents" in health


# =========================================================================
# ProductRepository Tests
# =========================================================================

class TestProductRepository:
    
    def test_insert_and_find(self, product_repo, sample_product):
        product_repo.insert_one(sample_product)
        
        found = product_repo.find_by_id("test_001")
        assert found is not None
        assert found["name"] == "Nike Air Max 90"
        assert found["price"] == 75.0
    
    def test_insert_adds_timestamps(self, product_repo, sample_product):
        product_repo.insert_one(sample_product)
        found = product_repo.find_by_id("test_001")
        assert "created_at" in found
        assert "updated_at" in found
    
    def test_insert_batch(self, product_repo):
        products = [
            {"product_id": f"batch_{i}", "name": f"Product {i}", "price": 10.0 + i,
             "brand_name": "testbrand", "main_category": "electronics"}
            for i in range(5)
        ]
        count = product_repo.insert_batch(products)
        assert count == 5
        assert product_repo.count() == 5
    
    def test_find_by_brand(self, product_repo, sample_product):
        product_repo.insert_one(sample_product)
        results = product_repo.find_by_brand("nike")
        assert len(results) == 1
        assert results[0]["brand_name"] == "nike"
    
    def test_find_by_brand_case_insensitive(self, product_repo, sample_product):
        product_repo.insert_one(sample_product)
        results = product_repo.find_by_brand("NIKE")
        assert len(results) == 1
    
    def test_find_by_category(self, product_repo, sample_product):
        product_repo.insert_one(sample_product)
        results = product_repo.find_by_category("men")
        assert len(results) == 1
    
    def test_find_by_price_range(self, product_repo):
        for i in range(5):
            product_repo.insert_one({
                "product_id": f"price_{i}",
                "price": 10.0 * (i + 1),  # 10, 20, 30, 40, 50
            })
        
        results = product_repo.find_by_price_range(15, 35)
        prices = [r["price"] for r in results]
        assert all(15 <= p <= 35 for p in prices)
        assert len(results) == 2  # 20 and 30
    
    def test_search(self, product_repo, sample_product):
        product_repo.insert_one(sample_product)
        results = product_repo.search("Air Max")
        assert len(results) == 1
    
    def test_update_price(self, product_repo, sample_product):
        product_repo.insert_one(sample_product)
        success = product_repo.update_price("test_001", 80.0)
        assert success
        
        found = product_repo.find_by_id("test_001")
        assert found["price"] == 80.0
    
    def test_delete(self, product_repo, sample_product):
        product_repo.insert_one(sample_product)
        assert product_repo.count() == 1
        
        success = product_repo.delete_by_id("test_001")
        assert success
        assert product_repo.count() == 0
    
    def test_brand_stats(self, product_repo):
        for i in range(3):
            product_repo.insert_one({
                "product_id": f"nike_{i}", "brand_name": "nike"
            })
        product_repo.insert_one({"product_id": "adidas_0", "brand_name": "adidas"})
        
        stats = product_repo.get_brand_stats()
        assert stats[0]["_id"] == "nike"
        assert stats[0]["count"] == 3
    
    def test_category_stats(self, product_repo, sample_product):
        product_repo.insert_one(sample_product)
        stats = product_repo.get_category_stats()
        assert len(stats) > 0


# =========================================================================
# PredictionRepository Tests
# =========================================================================

class TestPredictionRepository:
    
    def test_insert_and_find(self, prediction_repo, sample_prediction):
        prediction_repo.insert_one(sample_prediction)
        
        results = prediction_repo.find_by_product("test_001")
        assert len(results) == 1
        assert results[0]["predicted_price"] == 68.50
    
    def test_auto_computes_error(self, prediction_repo, sample_prediction):
        prediction_repo.insert_one(sample_prediction)
        
        result = prediction_repo.find_by_product("test_001")[0]
        assert result["error"] == pytest.approx(6.50)  # |68.50 - 75.0|
    
    def test_insert_adds_timestamp(self, prediction_repo, sample_prediction):
        prediction_repo.insert_one(sample_prediction)
        result = prediction_repo.find_by_product("test_001")[0]
        assert "predicted_at" in result
    
    def test_insert_batch(self, prediction_repo):
        predictions = [
            {
                "product_id": f"prod_{i}",
                "predicted_price": 20.0 + i,
                "actual_price": 25.0,
                "model_version": "1.0.0",
            }
            for i in range(10)
        ]
        count = prediction_repo.insert_batch(predictions)
        assert count == 10
    
    def test_find_latest(self, prediction_repo):
        for i in range(5):
            prediction_repo.insert_one({
                "product_id": f"prod_{i}",
                "predicted_price": float(i * 10),
                "model_version": "1.0.0",
            })
        
        latest = prediction_repo.find_latest(limit=3)
        assert len(latest) == 3
    
    def test_accuracy_stats(self, prediction_repo):
        predictions = [
            {"product_id": f"p_{i}", "predicted_price": 20.0 + i,
             "actual_price": 25.0, "model_version": "1.0.0"}
            for i in range(5)
        ]
        prediction_repo.insert_batch(predictions)
        
        stats = prediction_repo.get_accuracy_stats(model_version="1.0.0")
        assert stats["count"] == 5
        assert stats["mae"] is not None
    
    def test_delete_by_product(self, prediction_repo):
        for i in range(3):
            prediction_repo.insert_one({
                "product_id": "same_product",
                "predicted_price": float(i * 10),
            })
        
        deleted = prediction_repo.delete_by_product("same_product")
        assert deleted == 3
        assert prediction_repo.count() == 0
