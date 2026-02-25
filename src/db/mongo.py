"""
MongoDB integration for the Mercari Price Prediction Engine.

Collections:
  - products: Product catalog with features and metadata
  - predictions: Model predictions with timestamps and metrics

Provides:
  - MongoDBClient: Connection manager with health checks
  - ProductRepository: CRUD for product documents
  - PredictionRepository: CRUD for prediction documents
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)


# =========================================================================
# Connection Manager
# =========================================================================

class MongoDBClient:
    """
    MongoDB connection manager.
    
    Handles connection lifecycle, health checks, and index creation.
    
    Usage:
        client = MongoDBClient("mongodb://localhost:27017", "mercari_predictions")
        client.connect()
        # ... use client.db ...
        client.close()
    """
    
    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "mercari_predictions"):
        self.uri = uri
        self.db_name = db_name
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
    
    def connect(self) -> "MongoDBClient":
        """Establish connection and create indexes."""
        try:
            self._client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Verify connection
            self._client.admin.command("ping")
            self._db = self._client[self.db_name]
            logger.info(f"Connected to MongoDB at {self.uri}, database: {self.db_name}")
            
            # Create indexes
            self._create_indexes()
            return self
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self) -> None:
        """Create indexes for efficient queries."""
        # Products collection indexes
        products = self._db["products"]
        products.create_index("product_id", unique=True)
        products.create_index("brand_name")
        products.create_index("main_category")
        products.create_index("price")
        products.create_index([("created_at", DESCENDING)])
        
        # Predictions collection indexes
        predictions = self._db["predictions"]
        predictions.create_index("product_id")
        predictions.create_index([("predicted_at", DESCENDING)])
        predictions.create_index("model_version")
        
        logger.info("Database indexes created/verified")
    
    @property
    def db(self) -> Database:
        """Get the database instance."""
        if self._db is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._db
    
    def health_check(self) -> Dict:
        """Check connection health and return stats."""
        try:
            self._client.admin.command("ping")
            stats = self._db.command("dbStats")
            return {
                "status": "healthy",
                "database": self.db_name,
                "collections": self._db.list_collection_names(),
                "storage_mb": round(stats.get("storageSize", 0) / 1024 / 1024, 2),
                "documents": {
                    "products": self._db["products"].count_documents({}),
                    "predictions": self._db["predictions"].count_documents({}),
                },
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def close(self) -> None:
        """Close the connection."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")
    
    def __enter__(self) -> "MongoDBClient":
        """Context manager entry — connect and return self."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit — close connection."""
        self.close()
    
    def drop_database(self) -> None:
        """Drop the entire database. USE WITH CAUTION."""
        if self._client:
            self._client.drop_database(self.db_name)
            logger.warning(f"Dropped database: {self.db_name}")


# =========================================================================
# Product Repository
# =========================================================================

class ProductRepository:
    """
    CRUD operations for product documents.
    
    Product schema:
    {
        "product_id": str,           # Unique identifier
        "name": str,                 # Product name
        "item_description": str,     # Product description
        "brand_name": str,           # Brand
        "category_name": str,        # Full category path (e.g. "Women/Tops/Blouse")
        "main_category": str,        # Parsed main category
        "sub_category_1": str,       # Parsed sub-category 1
        "sub_category_2": str,       # Parsed sub-category 2
        "item_condition_id": int,    # Condition (1-5)
        "shipping": int,             # 0=buyer pays, 1=seller pays
        "price": float,              # Actual price (if known)
        "created_at": datetime,      # When the product was added
        "updated_at": datetime,      # Last modification time
    }
    """
    
    def __init__(self, db: Database):
        self.collection: Collection = db["products"]
    
    def insert_one(self, product: Dict) -> str:
        """Insert a single product. Returns the product_id."""
        now = datetime.now(timezone.utc)
        product.setdefault("created_at", now)
        product.setdefault("updated_at", now)
        
        result = self.collection.insert_one(product)
        logger.debug(f"Inserted product: {product.get('product_id', result.inserted_id)}")
        return str(result.inserted_id)
    
    def insert_batch(self, products: List[Dict]) -> int:
        """Insert multiple products. Returns count inserted."""
        now = datetime.now(timezone.utc)
        for p in products:
            p.setdefault("created_at", now)
            p.setdefault("updated_at", now)
        
        result = self.collection.insert_many(products, ordered=False)
        count = len(result.inserted_ids)
        logger.info(f"Inserted {count} products")
        return count
    
    def find_by_id(self, product_id: str) -> Optional[Dict]:
        """Find a product by its product_id."""
        return self.collection.find_one({"product_id": product_id})
    
    def find_by_brand(self, brand: str, limit: int = 50) -> List[Dict]:
        """Find products by brand name."""
        safe_brand = re.escape(brand)
        cursor = self.collection.find(
            {"brand_name": {"$regex": safe_brand, "$options": "i"}}
        ).limit(limit)
        return list(cursor)
    
    def find_by_category(self, category: str, limit: int = 50) -> List[Dict]:
        """Find products by main category."""
        safe_category = re.escape(category)
        cursor = self.collection.find(
            {"main_category": {"$regex": safe_category, "$options": "i"}}
        ).limit(limit)
        return list(cursor)
    
    def find_by_price_range(
        self, min_price: float, max_price: float, limit: int = 50
    ) -> List[Dict]:
        """Find products within a price range."""
        cursor = self.collection.find(
            {"price": {"$gte": min_price, "$lte": max_price}}
        ).sort("price", ASCENDING).limit(limit)
        return list(cursor)
    
    def search(self, query: str, limit: int = 20, offset: int = 0) -> List[Dict]:
        """Text search across name and description."""
        # Query should already be escaped by the caller for safety
        regex_filter = {
            "$or": [
                {"name": {"$regex": query, "$options": "i"}},
                {"item_description": {"$regex": query, "$options": "i"}},
            ]
        }
        cursor = self.collection.find(regex_filter).skip(offset).limit(limit)
        return list(cursor)
    
    def update_price(self, product_id: str, new_price: float) -> bool:
        """Update the price of a product."""
        result = self.collection.update_one(
            {"product_id": product_id},
            {"$set": {"price": new_price, "updated_at": datetime.now(timezone.utc)}},
        )
        return result.modified_count > 0
    
    def delete_by_id(self, product_id: str) -> bool:
        """Delete a product by its product_id."""
        result = self.collection.delete_one({"product_id": product_id})
        return result.deleted_count > 0
    
    def count(self) -> int:
        """Return total number of products."""
        return self.collection.count_documents({})
    
    def get_brand_stats(self) -> List[Dict]:
        """Get product count per brand (top 20)."""
        pipeline = [
            {"$group": {"_id": "$brand_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 20},
        ]
        return list(self.collection.aggregate(pipeline))
    
    def get_category_stats(self) -> List[Dict]:
        """Get product count per main category."""
        pipeline = [
            {"$group": {"_id": "$main_category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        return list(self.collection.aggregate(pipeline))


# =========================================================================
# Prediction Repository
# =========================================================================

class PredictionRepository:
    """
    CRUD operations for prediction documents.
    
    Prediction schema:
    {
        "product_id": str,           # References the product
        "predicted_price": float,    # Model's predicted price (original scale)
        "predicted_log_price": float,# Model's raw output (log1p scale)
        "actual_price": float,       # Actual price (if known)
        "error": float,              # |predicted - actual| (if actual known)
        "model_version": str,        # Which model version made this prediction
        "predicted_at": datetime,    # When the prediction was made
        "features_used": dict,       # Input features snapshot
    }
    """
    
    def __init__(self, db: Database):
        self.collection: Collection = db["predictions"]
    
    def insert_one(self, prediction: Dict) -> str:
        """Insert a single prediction. Returns the inserted ID."""
        prediction.setdefault("predicted_at", datetime.now(timezone.utc))
        
        # Compute error if actual price is known
        if "actual_price" in prediction and "predicted_price" in prediction:
            prediction["error"] = abs(
                prediction["predicted_price"] - prediction["actual_price"]
            )
        
        result = self.collection.insert_one(prediction)
        return str(result.inserted_id)
    
    def insert_batch(self, predictions: List[Dict]) -> int:
        """Insert multiple predictions. Returns count inserted."""
        now = datetime.now(timezone.utc)
        for p in predictions:
            p.setdefault("predicted_at", now)
            if "actual_price" in p and "predicted_price" in p:
                p["error"] = abs(p["predicted_price"] - p["actual_price"])
        
        result = self.collection.insert_many(predictions, ordered=False)
        count = len(result.inserted_ids)
        logger.info(f"Inserted {count} predictions")
        return count
    
    def find_by_product(self, product_id: str) -> List[Dict]:
        """Get all predictions for a product (newest first)."""
        cursor = self.collection.find(
            {"product_id": product_id}
        ).sort("predicted_at", DESCENDING)
        return list(cursor)
    
    def find_latest(self, limit: int = 20) -> List[Dict]:
        """Get the most recent predictions."""
        cursor = self.collection.find().sort(
            "predicted_at", DESCENDING
        ).limit(limit)
        return list(cursor)
    
    def get_accuracy_stats(self, model_version: Optional[str] = None) -> Dict:
        """
        Compute prediction accuracy statistics.
        
        Returns MAE, median error, and count for predictions that have
        both predicted and actual prices.
        """
        match_filter = {"error": {"$exists": True}}
        if model_version:
            match_filter["model_version"] = model_version
        
        pipeline = [
            {"$match": match_filter},
            {"$group": {
                "_id": "$model_version",
                "count": {"$sum": 1},
                "mae": {"$avg": "$error"},
                "max_error": {"$max": "$error"},
                "min_error": {"$min": "$error"},
                "avg_predicted": {"$avg": "$predicted_price"},
                "avg_actual": {"$avg": "$actual_price"},
            }},
            {"$sort": {"count": -1}},
        ]
        
        results = list(self.collection.aggregate(pipeline))
        if not results:
            return {"count": 0, "mae": None}
        
        return results[0] if model_version else results
    
    def delete_by_product(self, product_id: str) -> int:
        """Delete all predictions for a product."""
        result = self.collection.delete_many({"product_id": product_id})
        return result.deleted_count
    
    def count(self) -> int:
        """Return total number of predictions."""
        return self.collection.count_documents({})


# =========================================================================
# Convenience: Ingest products from training data
# =========================================================================

def ingest_training_data(
    db: Database,
    tsv_path: str,
    max_rows: Optional[int] = None,
) -> int:
    """
    Load raw training data into MongoDB products collection.
    
    Args:
        db: MongoDB database instance
        tsv_path: Path to train.tsv
        max_rows: Optional limit on rows to ingest
    
    Returns:
        Number of products inserted
    """
    import pandas as pd
    from src.data.preprocess import parse_category
    
    print(f"[INFO] Loading data from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep="\t", nrows=max_rows)
    
    # Filter valid prices
    df = df[df["price"] > 0].copy()
    
    # Parse categories
    cats = df["category_name"].apply(
        lambda x: parse_category(x) if isinstance(x, str) else ("missing", "missing", "missing")
    )
    df["main_category"] = cats.apply(lambda x: x[0])
    df["sub_category_1"] = cats.apply(lambda x: x[1])
    df["sub_category_2"] = cats.apply(lambda x: x[2])
    
    # Fill missing brands
    df["brand_name"] = df["brand_name"].fillna("unknown")
    
    # Build product documents
    products = []
    for _, row in df.iterrows():
        products.append({
            "product_id": str(row["train_id"]),
            "name": str(row.get("name", "")),
            "item_description": str(row.get("item_description", "")),
            "brand_name": str(row["brand_name"]).lower(),
            "category_name": str(row.get("category_name", "")),
            "main_category": row["main_category"],
            "sub_category_1": row["sub_category_1"],
            "sub_category_2": row["sub_category_2"],
            "item_condition_id": int(row["item_condition_id"]),
            "shipping": int(row["shipping"]),
            "price": float(row["price"]),
        })
    
    # Batch insert
    repo = ProductRepository(db)
    
    # Insert in chunks of 5000 for memory efficiency
    total = 0
    chunk_size = 5000
    for i in range(0, len(products), chunk_size):
        chunk = products[i:i + chunk_size]
        count = repo.insert_batch(chunk)
        total += count
        print(f"  Inserted {total:,} / {len(products):,} products...")
    
    print(f"[SUCCESS] Ingested {total:,} products into MongoDB")
    return total
