const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface PredictionRequest {
    name: string;
    item_description: string;
    category_name: string;
    brand_name: string;
    item_condition_id: number;
    shipping: number;
}

export interface PredictionResponse {
    predicted_price: number;
    predicted_log_price: number;
    confidence_range: { low: number; high: number };
    input_summary: Record<string, string>;
}

export interface HealthResponse {
    status: string;
    model_version: string;
    model_loaded: boolean;
    mongodb_status: string;
    timestamp: string;
}

// --- Model Info ---
export interface LossCurvePoint {
    epoch: number;
    train_loss: number;
    val_loss: number;
}

export interface ModelInfoResponse {
    model_version: string;
    model_parameters: number;
    architecture: string;
    test_metrics: Record<string, number>;
    val_metrics: Record<string, number>;
    loss_curve: LossCurvePoint[];
    best_epoch: number;
    total_epochs: number;
    train_size: number;
    val_size: number;
    test_size: number;
    config_summary: Record<string, string>;
}

// --- Products ---
export interface ProductItem {
    product_id: string;
    name: string;
    brand_name: string;
    category_name: string;
    main_category: string;
    item_condition_id: number;
    shipping: number;
    price: number;
}

export interface ProductSearchResponse {
    products: ProductItem[];
    total: number;
    query: string;
}

export interface CategoryStat {
    category: string;
    count: number;
}

export interface BrandStat {
    brand: string;
    count: number;
}

export interface ProductStatsResponse {
    total_products: number;
    total_brands: number;
    total_categories: number;
    avg_price: number;
    category_distribution: CategoryStat[];
    top_brands: BrandStat[];
}

// --- Prediction History ---
export interface RecentPredictionItem {
    product_name: string;
    brand: string;
    predicted_price: number;
    confidence_low: number;
    confidence_high: number;
    predicted_at: string;
}

export interface RecentPredictionsResponse {
    predictions: RecentPredictionItem[];
    total: number;
}


// =========================================================================
// API Functions
// =========================================================================

export async function predictPrice(data: PredictionRequest): Promise<PredictionResponse> {
    const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}

export async function checkHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
    const res = await fetch(`${API_BASE}/model/info`);
    if (!res.ok) throw new Error(`Model info failed: ${res.status}`);
    return res.json();
}

export async function searchProducts(query: string = '', limit: number = 20, offset: number = 0): Promise<ProductSearchResponse> {
    const params = new URLSearchParams({ q: query, limit: limit.toString(), offset: offset.toString() });
    const res = await fetch(`${API_BASE}/products/search?${params}`);
    if (!res.ok) throw new Error(`Product search failed: ${res.status}`);
    return res.json();
}

export async function fetchProductStats(): Promise<ProductStatsResponse> {
    const res = await fetch(`${API_BASE}/products/stats`);
    if (!res.ok) throw new Error(`Product stats failed: ${res.status}`);
    return res.json();
}

export async function fetchRecentPredictions(limit: number = 20): Promise<RecentPredictionsResponse> {
    const res = await fetch(`${API_BASE}/predictions/recent?limit=${limit}`);
    if (!res.ok) throw new Error(`Recent predictions failed: ${res.status}`);
    return res.json();
}

export function formatPrice(price: number): string {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
    }).format(price);
}

export const CATEGORIES = [
    'Men', 'Women', 'Beauty', 'Kids', 'Electronics',
    'Home', 'Vintage & Collectibles', 'Other', 'Handmade',
    'Sports & Outdoors',
];

export const CONDITIONS = [
    { value: 1, label: 'New with tags' },
    { value: 2, label: 'New without tags' },
    { value: 3, label: 'Good' },
    { value: 4, label: 'Fair' },
    { value: 5, label: 'Poor' },
];

