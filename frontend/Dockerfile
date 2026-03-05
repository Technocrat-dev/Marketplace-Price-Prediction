# ============================================================
# Marketplace Price Prediction — Frontend Dockerfile
# Next.js standalone production build (optimized for Cloud Run)
# ============================================================

FROM node:20-alpine AS builder

WORKDIR /app

# Accept API URL as build argument (required for NEXT_PUBLIC_ env vars)
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Build (standalone mode — produces self-contained output)
COPY . .
RUN npm run build

# Stage 2: Minimal runtime
FROM node:20-alpine

WORKDIR /app

# Copy standalone server + static assets
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
