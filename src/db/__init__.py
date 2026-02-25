"""MongoDB integration layer."""

from src.db.mongo import MongoDBClient, ProductRepository, PredictionRepository

__all__ = ["MongoDBClient", "ProductRepository", "PredictionRepository"]
