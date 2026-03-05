#!/bin/bash
# ============================================================
# Marketplace Price Prediction — GCP Cloud Run Deployment
# Free-tier optimized deployment script
# ============================================================
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated
#   2. Docker installed
#   3. MongoDB Atlas free cluster created
#   4. Model artifacts exist locally:
#      - outputs/checkpoints/best_model.pt
#      - outputs/training_results.json
#      - data/processed/metadata.json
#      - data/processed/name_vocab.json
#      - data/processed/desc_vocab.json
#      - data/processed/cat_encoder.json
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# ============================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-pricescope}"
REGION="us-central1"
REPO_NAME="pricescope-repo"
API_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/api"
FRONTEND_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/frontend"

# MongoDB Atlas connection (set these before running)
MONGODB_URI="${MONGODB_URI:-}"
MONGODB_DB="${MONGODB_DB:-mercari_predictions}"

echo "═══════════════════════════════════════════════════════"
echo "  PriceScope — GCP Cloud Run Deployment"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "═══════════════════════════════════════════════════════"

# ── Step 1: Set project ────────────────────────────────────
echo ""
echo "▸ Setting GCP project..."
gcloud config set project "${PROJECT_ID}"

# ── Step 2: Enable required APIs ───────────────────────────
echo "▸ Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com

# ── Step 3: Create Artifact Registry repository ────────────
echo "▸ Creating Artifact Registry repository..."
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --quiet 2>/dev/null || echo "  (repository already exists)"

# ── Step 4: Configure Docker for Artifact Registry ─────────
echo "▸ Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── Step 5: Build and push API image ──────────────────────
echo ""
echo "▸ Building API image (this may take a few minutes)..."
docker build -t "${API_IMAGE}:latest" .
echo "▸ Pushing API image..."
docker push "${API_IMAGE}:latest"

# ── Step 6: Build and push Frontend image ─────────────────
echo ""
echo "▸ Building Frontend image..."
docker build -t "${FRONTEND_IMAGE}:latest" ./frontend
echo "▸ Pushing Frontend image..."
docker push "${FRONTEND_IMAGE}:latest"

# ── Step 7: Deploy API to Cloud Run ───────────────────────
echo ""
echo "▸ Deploying API to Cloud Run..."

DEPLOY_ENV="MONGODB_DB=${MONGODB_DB}"
if [ -n "${MONGODB_URI}" ]; then
  DEPLOY_ENV="${DEPLOY_ENV},MONGODB_URI=${MONGODB_URI}"
fi

gcloud run deploy pricescope-api \
  --image "${API_IMAGE}:latest" \
  --region "${REGION}" \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 2 \
  --cpu-boost \
  --set-env-vars "${DEPLOY_ENV}" \
  --allow-unauthenticated \
  --quiet

# Get the API URL
API_URL=$(gcloud run services describe pricescope-api \
  --region "${REGION}" \
  --format 'value(status.url)')
echo "  ✓ API deployed at: ${API_URL}"

# ── Step 8: Deploy Frontend to Cloud Run ──────────────────
echo ""
echo "▸ Deploying Frontend to Cloud Run..."
gcloud run deploy pricescope-frontend \
  --image "${FRONTEND_IMAGE}:latest" \
  --region "${REGION}" \
  --port 3000 \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "NEXT_PUBLIC_API_URL=${API_URL}" \
  --allow-unauthenticated \
  --quiet

FRONTEND_URL=$(gcloud run services describe pricescope-frontend \
  --region "${REGION}" \
  --format 'value(status.url)')
echo "  ✓ Frontend deployed at: ${FRONTEND_URL}"

# ── Step 9: Set up warmup scheduler (free tier) ───────────
echo ""
echo "▸ Setting up Cloud Scheduler warmup ping..."
gcloud scheduler jobs create http pricescope-warmup \
  --schedule="*/10 * * * *" \
  --uri="${API_URL}/health" \
  --http-method=GET \
  --location="${REGION}" \
  --quiet 2>/dev/null || echo "  (scheduler job already exists)"

# ── Done ──────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ Deployment Complete!"
echo ""
echo "  🌐 Frontend: ${FRONTEND_URL}"
echo "  🔌 API:      ${API_URL}"
echo "  📊 Health:   ${API_URL}/health"
echo "  📖 Docs:     ${API_URL}/docs"
echo ""
echo "  Note: First request may take 10-30s (cold start)."
echo "  Cloud Scheduler will keep the API warm every 10 min."
echo "═══════════════════════════════════════════════════════"
