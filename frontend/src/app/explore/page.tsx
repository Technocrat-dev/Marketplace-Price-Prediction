'use client';

import { useState, useEffect, useCallback } from 'react';
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
                        </>
                    )}
                </div>

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
