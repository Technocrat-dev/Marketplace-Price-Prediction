# PriceScope — Marketplace Price Prediction Engine

[![CI](https://github.com/Technocrat-dev/Marketplace-Price-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Technocrat-dev/Marketplace-Price-Prediction/actions)

A **multimodal deep learning system** that predicts marketplace product prices by analyzing text descriptions, brand reputation, category context, and item condition. Trained on **1.48 million** real Mercari listings.

## Architecture

```
┌─────────────┐   ┌─────────────┐   ┌─────────────────┐
│ Product Name │   │ Description │   │ Categoricals    │
│   (text)     │   │   (text)    │   │ brand+cat+cond  │
└──────┬───────┘   └──────┬──────┘   └────────┬────────┘
       │                  │                    │
  ┌────▼────┐       ┌─────▼─────┐       ┌─────▼──────┐
  │ BiLSTM  │       │  BiLSTM   │       │ Embeddings │
  │ Encoder │       │  Encoder  │       │  + FC      │
  │ (256d)  │       │  (256d)   │       │  (64d)     │
  └────┬────┘       └─────┬─────┘       └─────┬──────┘
       │                  │                    │
       └──────────────────┼────────────────────┘
                          │
                   ┌──────▼──────┐
                   │ Fusion MLP  │
                   │ 576→256→128 │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │ Predicted   │
                   │ log(price)  │
                   └─────────────┘
```

**Stack:** PyTorch · FastAPI · MongoDB · ONNX Runtime · Next.js · Recharts · XGBoost · Optuna · SHAP

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB (optional, for data persistence)
- Kaggle API credentials (for dataset download)

### 1. Clone & Setup Python Environment

```bash
git clone https://github.com/Technocrat-dev/Marketplace-Price-Prediction.git
cd Marketplace-Price-Prediction

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download & Preprocess Data

```bash
# Set up Kaggle credentials (~/.kaggle/kaggle.json)
python data/download.py

# Preprocess raw TSV → numpy arrays
python -c "from src.data.preprocess import run_preprocessing_pipeline; run_preprocessing_pipeline('config/config.yaml')"
```

### 3. Train the Model

```bash
python scripts/train.py --config config/config.yaml --epochs 10
```

### 4. Start the API Server

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

## API Documentation

### `POST /predict` — Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Nike Air Max 90",
    "item_description": "Classic sneakers, barely worn, size 10",
    "category_name": "Men/Shoes/Athletic",
    "brand_name": "Nike",
    "item_condition_id": 2,
    "shipping": 1
  }'
```

**Response:**
```json
{
  "predicted_price": 68.42,
  "predicted_log_price": 4.23,
  "confidence_range": { "low": 54.74, "high": 85.53 },
  "input_summary": {
    "name": "nike air max 90",
    "brand": "nike",
    "category": "men/shoes/athletic",
    "condition": "2",
    "shipping": "seller pays"
  }
}
```

### `POST /predict/batch` — Batch Prediction (up to 100 items)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"name": "iPhone 12 Case"}, {"name": "Vintage Dress"}]}'
```

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

### `GET /model/info` — Model Architecture & Metrics

Returns training metrics, loss curves, architecture summary, and config.

```bash
curl http://localhost:8000/model/info
```

### `GET /products/search` — Search Product Catalog

```bash
curl "http://localhost:8000/products/search?q=nike+shoes&limit=10"
```

### `GET /products/stats` — Catalog Statistics

```bash
curl http://localhost:8000/products/stats
```

### `GET /predictions/recent` — Prediction History

```bash
curl "http://localhost:8000/predictions/recent?limit=20"
```

### `POST /predict/csv` — CSV Batch Upload (up to 500 rows)

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@products.csv"
```

### `POST /predict/explain` — Prediction with Feature Analysis

Returns the prediction plus a per-feature contribution breakdown.

```bash
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"name": "Nike Air Max 90", "brand_name": "Nike"}'
```

## Docker Deployment

```bash
docker-compose up --build
```

This starts three services:
| Service    | Port  | Description            |
|------------|-------|------------------------|
| `api`      | 8000  | FastAPI prediction API  |
| `frontend` | 3000  | Next.js web interface   |
| `mongodb`  | 27017 | MongoDB data store      |

## Project Structure

```
├── config/
│   └── config.yaml          # All hyperparameters and paths
├── data/
│   ├── download.py          # Kaggle dataset downloader
│   ├── raw/                 # Raw TSV files (gitignored)
│   └── processed/           # Numpy arrays (gitignored)
├── frontend/                # Next.js frontend
│   ├── src/app/             # Pages: predict, dashboard, explore, about
│   ├── src/components/      # Navbar with responsive design
│   └── src/lib/api.ts       # API client
├── scripts/
│   ├── train.py             # Training entry point
│   ├── train_baselines.py   # XGBoost / LightGBM / Ridge baselines
│   ├── tune.py              # Optuna hyperparameter search
│   ├── explain.py           # SHAP feature explanations
│   ├── ingest_data.py       # MongoDB data ingestion
│   └── export_onnx.py       # ONNX model export
├── src/
│   ├── data/                # Dataset, preprocessing, feature engineering
│   ├── db/                  # MongoDB repositories
│   ├── models/              # BiLSTM + Attention + TabularEncoder + Fusion MLP
│   ├── serving/             # FastAPI app + schemas + rate limiting
│   └── training/            # Trainer + evaluation metrics
├── tests/                   # 88+ unit tests (pytest)
├── .github/workflows/       # CI: test + lint + Docker build
├── Dockerfile               # Multi-stage API container
├── docker-compose.yml       # Full stack orchestration
├── requirements.txt         # Python dependencies
└── README.md
```

## Model Performance

| Metric      | Value  |
|-------------|--------|
| RMSLE       | 0.430  |
| MAE         | $8.42  |
| R² Score    | 0.482  |
| Parameters  | 15.2M  |
| Train Data  | 1.18M  |
| Val Data    | 148K   |
| Test Data   | 148K   |

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_serving.py -v
```

## ML Engineering Scripts

```bash
# Train baseline models (XGBoost, LightGBM, Ridge)
python scripts/train_baselines.py

# Optuna hyperparameter search
python scripts/tune.py --n-trials 30

# Generate SHAP feature explanations
python scripts/explain.py --sample 100
```

## Configuration

All hyperparameters are in `config/config.yaml`:

| Section    | Key Parameters                                    |
|------------|---------------------------------------------------|
| `paths`    | Data directories, output paths                    |
| `data`     | Max sequence lengths, vocab min frequency          |
| `model`    | Embedding dims, hidden dims, dropout, attention    |
| `training` | Batch size, learning rate, epochs, scheduler       |
| `serving`  | Host, port, model version, API key, cache settings |
| `database` | MongoDB URI, database name                         |

## API Features

| Feature             | Description                                    |
|---------------------|------------------------------------------------|
| Rate limiting       | 60/min on predict, 10/min on batch             |
| Request logging     | Method, path, latency, client IP per request   |
| LRU response cache  | Deduplicates identical predictions via MD5     |
| API key auth        | Optional `X-API-Key` header (configurable)     |
| CSV batch upload    | Upload CSV file for batch predictions          |
| Feature explanation | Per-feature contribution breakdown             |

## License

MIT
