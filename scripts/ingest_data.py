"""
Script to ingest Mercari training data into MongoDB.

Usage:
    # Ingest all data
    python scripts/ingest_data.py

    # Ingest a subset (for testing)
    python scripts/ingest_data.py --limit 1000
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.mongo import MongoDBClient, ingest_training_data


def main():
    parser = argparse.ArgumentParser(description="Ingest training data into MongoDB")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max rows to ingest (default: all)")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config file")
    parser.add_argument("--drop", action="store_true",
                        help="Drop existing database before ingesting")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Connect to MongoDB
    client = MongoDBClient(
        uri=cfg["database"]["uri"],
        db_name=cfg["database"]["name"],
    )
    client.connect()
    
    # Drop if requested
    if args.drop:
        print("[WARNING] Dropping existing database...")
        client.drop_database()
        client.connect()  # Reconnect to recreate indexes
    
    # Ingest
    tsv_path = str(Path(cfg["paths"]["raw_data"]) / cfg["data"]["train_file"])
    ingest_training_data(
        db=client.db,
        tsv_path=tsv_path,
        max_rows=args.limit,
    )
    
    # Show health
    print("\n" + "=" * 50)
    health = client.health_check()
    print(f"  Database: {health['database']}")
    print(f"  Products: {health['documents']['products']:,}")
    print(f"  Predictions: {health['documents']['predictions']:,}")
    print(f"  Storage: {health['storage_mb']} MB")
    print("=" * 50)
    
    client.close()


if __name__ == "__main__":
    main()
