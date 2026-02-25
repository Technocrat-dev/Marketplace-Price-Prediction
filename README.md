# PriceScope

[![CI](https://github.com/Technocrat-dev/Marketplace-Price-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Technocrat-dev/Marketplace-Price-Prediction/actions)

A multimodal deep learning system for marketplace price prediction. Combines bidirectional LSTMs for text encoding with categorical embeddings and a fusion MLP to predict product prices from names, descriptions, brands, categories, and condition ratings. Trained on 1.48 million Mercari product listings.

## Architecture

```
[Product Name]     [Description]      [Categoricals]
    (text)             (text)         brand+cat+cond
       |                  |                  |
  +----v----+       +-----v-----+      +-----v------+
  | BiLSTM  |       |  BiLSTM   |      | Embeddings |
  | Encoder |       |  Encoder  |      |   + FC     |
  | (256d)  |       |  (256d)   |      |   (64d)    |
  +----+----+       +-----+-----+      +-----+------+
       |                  |                  |
       +------------------+------------------+
                          |
                   +------v------+
                   | Fusion MLP  |
                   | 576>256>128 |
                   +------+------+
                          |
                   +------v------+
                   |  Predicted  |
                   |  log(price) |
                   +-------------+
```

**Key capabilities:** self-attention mechanism, Optuna hyperparameter tuning, XGBoost/LightGBM/Ridge baselines, SHAP explainability, rate-limited API with caching and optional auth, CSV batch upload, CI/CD pipeline.

**Stack:** PyTorch, FastAPI, MongoDB, ONNX Runtime, Next.js, Recharts, XGBoost, Optuna, SHAP

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB (optional, for data persistence)
- Kaggle API credentials (for dataset download)

### 1. Clone and Setup Python Environment

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

### 2. Download and Preprocess Data

```bash
# Set up Kaggle credentials (~/.kaggle/kaggle.json)
python data/download.py

# Preprocess raw TSV into numpy arrays
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

Interactive docs available at `http://localhost:8000/docs` when the server is running.

### `POST /predict` -- Single Prediction

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

Response:

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

### `POST /predict/batch` -- Batch Prediction (up to 100 items)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"name": "iPhone 12 Case"}, {"name": "Vintage Dress"}]}'
```

### `POST /predict/csv` -- CSV Batch Upload (up to 500 rows)

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@products.csv"
```

### `POST /predict/explain` -- Prediction with Feature Analysis

Returns the prediction along with a per-feature contribution breakdown.

```bash
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"name": "Nike Air Max 90", "brand_name": "Nike"}'
```

### `GET /model/info` -- Model Architecture and Metrics

Returns training metrics, loss curves, architecture details, and configuration.

```bash
curl http://localhost:8000/model/info
```

### `GET /products/search` -- Search Product Catalog

```bash
curl "http://localhost:8000/products/search?q=nike+shoes&limit=10"
```

### `GET /products/stats` -- Catalog Statistics

```bash
curl http://localhost:8000/products/stats
```

### `GET /predictions/recent` -- Prediction History

```bash
curl "http://localhost:8000/predictions/recent?limit=20"
```

### `GET /health` -- Health Check

```bash
curl http://localhost:8000/health
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
config/
  config.yaml              All hyperparameters and paths
data/
  download.py              Kaggle dataset downloader
  raw/                     Raw TSV files (gitignored)
  processed/               Numpy arrays (gitignored)
frontend/                  Next.js frontend
  src/app/                 Pages: predict, dashboard, explore, about
  src/components/          Navbar with responsive design
  src/lib/api.ts           API client
scripts/
  train.py                 Training entry point
  train_baselines.py       XGBoost, LightGBM, Ridge baselines
  tune.py                  Optuna hyperparameter search
  explain.py               SHAP feature explanations
  ingest_data.py           MongoDB data ingestion
  export_onnx.py           ONNX model export
src/
  data/                    Dataset, preprocessing, feature engineering
  db/                      MongoDB repositories
  models/                  BiLSTM + Attention + TabularEncoder + Fusion MLP
  serving/                 FastAPI app, schemas, rate limiting, caching
  training/                Trainer, evaluation metrics
tests/                     88+ unit tests (pytest)
.github/workflows/         CI: test, lint, Docker build
Dockerfile                 Multi-stage API container
docker-compose.yml         Full stack orchestration
requirements.txt           Python dependencies
```

## Model Performance

| Metric     | Value |
|------------|-------|
| RMSLE      | 0.430 |
| MAE        | $8.42 |
| R2 Score   | 0.482 |
| Parameters | 15.2M |
| Train Data | 1.18M |
| Val Data   | 148K  |
| Test Data  | 148K  |

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

All hyperparameters are centralized in `config/config.yaml`:

| Section    | Key Parameters                                    |
|------------|---------------------------------------------------|
| `paths`    | Data directories, output paths                    |
| `data`     | Max sequence lengths, vocab min frequency          |
| `model`    | Embedding dims, hidden dims, dropout, attention    |
| `training` | Batch size, learning rate, epochs, scheduler       |
| `serving`  | Host, port, model version, API key, cache settings |
| `database` | MongoDB URI, database name                         |

## API Features

| Feature             | Description                                  |
|---------------------|----------------------------------------------|
| Rate limiting       | 60/min on predict, 10/min on batch           |
| Request logging     | Method, path, latency, client IP per request |
| LRU response cache  | Deduplicates identical predictions via MD5   |
| API key auth        | Optional X-API-Key header (configurable)     |
| CSV batch upload    | Upload CSV file for batch predictions        |
| Feature explanation | Per-feature contribution breakdown           |

## License

MIT
