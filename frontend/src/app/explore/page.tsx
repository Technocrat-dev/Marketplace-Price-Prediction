'use client';

import { useState, useEffect, useCallback } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, PieChart, Pie, Legend,
} from 'recharts';
import styles from './explore.module.css';
import {
    formatPrice, searchProducts, fetchProductStats,
    type ProductItem, type ProductStatsResponse,
} from '@/lib/api';

const CONDITIONS_MAP: Record<number, string> = {
    1: 'New with tags',
    2: 'New no tags',
    3: 'Good',
    4: 'Fair',
    5: 'Poor',
};

const CHART_COLORS = [
    '#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#818cf8',
    '#7c3aed', '#5b21b6', '#4f46e5', '#4338ca', '#3730a3',
    '#6d28d9', '#9333ea', '#a855f7', '#d946ef', '#ec4899',
];

const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    return (
        <div className={styles.chartTooltip}>
            <p className={styles.chartTooltipLabel}>{label}</p>
            <p className={styles.chartTooltipValue}>
                {payload[0].value.toLocaleString()} products
            </p>
        </div>
    );
};

export default function ExplorePage() {
    const [query, setQuery] = useState('');
    const [products, setProducts] = useState<ProductItem[]>([]);
    const [stats, setStats] = useState<ProductStatsResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [searching, setSearching] = useState(false);
    const [error, setError] = useState('');
    const [total, setTotal] = useState(0);

    // Load initial data
    useEffect(() => {
        async function loadInitial() {
            try {
                const [productRes, statsRes] = await Promise.allSettled([
                    searchProducts('', 20, 0),
                    fetchProductStats(),
                ]);

                if (productRes.status === 'fulfilled') {
                    setProducts(productRes.value.products);
                    setTotal(productRes.value.total);
                }
                if (statsRes.status === 'fulfilled') {
                    setStats(statsRes.value);
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load data');
            } finally {
                setLoading(false);
            }
        }

        loadInitial();
    }, []);

    const handleSearch = useCallback(async () => {
        setSearching(true);
        setError('');
        try {
            const res = await searchProducts(query, 20, 0);
            setProducts(res.products);
            setTotal(res.total);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Search failed');
        } finally {
            setSearching(false);
        }
    }, [query]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') handleSearch();
    };

    // Prepare chart data
    const categoryData = stats?.category_distribution
        ?.slice(0, 10)
        .map(c => ({ name: c.category, count: c.count })) || [];

    const brandData = stats?.top_brands
        ?.slice(0, 8)
        .map(b => ({ name: b.brand, count: b.count })) || [];

    return (
        <div className={styles.page}>
            <div className={styles.container}>
                <div className={styles.header}>
                    <h1>Explore Products</h1>
                    <p>Browse the marketplace catalog and see pricing patterns</p>
                </div>

                {/* Stats */}
                <div className={styles.statsGrid}>
                    {loading ? (
                        <>
                            <div className={styles.statCard}><div className={styles.statSkeleton} /></div>
                            <div className={styles.statCard}><div className={styles.statSkeleton} /></div>
                            <div className={styles.statCard}><div className={styles.statSkeleton} /></div>
                            <div className={styles.statCard}><div className={styles.statSkeleton} /></div>
                        </>
                    ) : (
                        <>
                            <div className={styles.statCard}>
                                <div className={styles.statValue}>
                                    {stats ? stats.total_products.toLocaleString() : total.toLocaleString()}
                                </div>
                                <div className={styles.statLabel}>Total Products</div>
                            </div>
                            <div className={styles.statCard}>
                                <div className={styles.statValue}>
                                    {stats ? stats.total_brands.toLocaleString() : '—'}
                                </div>
                                <div className={styles.statLabel}>Unique Brands</div>
                            </div>
                            <div className={styles.statCard}>
                                <div className={styles.statValue}>
                                    {stats ? stats.total_categories.toLocaleString() : '—'}
                                </div>
                                <div className={styles.statLabel}>Categories</div>
                            </div>
                            <div className={styles.statCard}>
                                <div className={styles.statValue}>
                                    {stats ? formatPrice(stats.avg_price) : '—'}
                                </div>
                                <div className={styles.statLabel}>Avg Price</div>
                            </div>
                        </>
                    )}
                </div>

                {/* Charts */}
                {!loading && stats && (categoryData.length > 0 || brandData.length > 0) && (
                    <div className={styles.chartsSection}>
                        {categoryData.length > 0 && (
                            <div className={styles.chartCard}>
                                <h3 className={styles.chartTitle}>Top Categories</h3>
                                <div className={styles.chartContainer}>
                                    <ResponsiveContainer width="100%" height={320}>
                                        <BarChart
                                            data={categoryData}
                                            layout="vertical"
                                            margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                                            <XAxis
                                                type="number"
                                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                                tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                                            />
                                            <YAxis
                                                type="category"
                                                dataKey="name"
                                                tick={{ fill: '#cbd5e1', fontSize: 12 }}
                                                width={75}
                                            />
                                            <Tooltip content={<CustomTooltip />} />
                                            <Bar dataKey="count" radius={[0, 6, 6, 0]} barSize={22}>
                                                {categoryData.map((_, i) => (
                                                    <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        {brandData.length > 0 && (
                            <div className={styles.chartCard}>
                                <h3 className={styles.chartTitle}>Top Brands</h3>
                                <div className={styles.chartContainer}>
                                    <ResponsiveContainer width="100%" height={320}>
                                        <BarChart
                                            data={brandData}
                                            margin={{ top: 5, right: 30, left: 10, bottom: 50 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                                            <XAxis
                                                dataKey="name"
                                                tick={{ fill: '#cbd5e1', fontSize: 11 }}
                                                angle={-30}
                                                textAnchor="end"
                                                height={60}
                                            />
                                            <YAxis
                                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                                tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                                            />
                                            <Tooltip content={<CustomTooltip />} />
                                            <Bar dataKey="count" radius={[6, 6, 0, 0]} barSize={36}>
                                                {brandData.map((_, i) => (
                                                    <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Search */}
                <div className={styles.searchBox}>
                    <input
                        id="product-search"
                        className={styles.searchInput}
                        placeholder="Search by product name, brand, or category..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        aria-label="Search products"
                    />
                    <button
                        className={styles.searchBtn}
                        onClick={handleSearch}
                        disabled={searching}
                        aria-label="Search"
                    >
                        {searching ? 'Searching...' : 'Search'}
                    </button>
                </div>

                {error && (
                    <div className={styles.errorBanner}>{error}</div>
                )}

                {/* Products */}
                <div className={styles.productsGrid}>
                    {loading && (
                        Array.from({ length: 6 }).map((_, i) => (
                            <div key={i} className={`${styles.productCard} ${styles.productSkeleton}`}>
                                <div className={styles.skeletonLine} style={{ width: '80%', height: 18 }} />
                                <div className={styles.skeletonLine} style={{ width: '60%', height: 14 }} />
                                <div className={styles.skeletonLine} style={{ width: '40%', height: 24 }} />
                            </div>
                        ))
                    )}

                    {!loading && products.length === 0 && (
                        <div className={styles.emptyState}>
                            {query
                                ? 'No products match your search. Try a different query.'
                                : 'No products in the catalog. Run the data ingestion script first.'}
                        </div>
                    )}

                    {!loading && products.map((product) => (
                        <div key={product.product_id} className={styles.productCard}>
                            <div className={styles.productName}>{product.name}</div>
                            <div className={styles.productMeta}>
                                <span className={`${styles.badge} ${styles.brandBadge}`}>
                                    {product.brand_name}
                                </span>
                                <span className={styles.badge}>
                                    {product.main_category || product.category_name?.split('/')[0] || 'Other'}
                                </span>
                                <span className={styles.badge}>
                                    {CONDITIONS_MAP[product.item_condition_id] || 'Unknown'}
                                </span>
                            </div>
                            <div className={styles.productPrice}>
                                {formatPrice(product.price)}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
