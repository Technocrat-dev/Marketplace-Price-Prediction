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
