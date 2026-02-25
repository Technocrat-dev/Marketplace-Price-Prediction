'use client';

import { useState } from 'react';
import styles from './explore.module.css';
import { formatPrice } from '@/lib/api';

// Sample products to showcase — replace with API data when backend is ready
const SAMPLE_PRODUCTS = [
    { id: '1', name: 'Lululemon Align Pant 25"', brand: 'Lululemon', category: 'Women/Athletic Apparel', price: 68, condition: 2 },
    { id: '2', name: 'Nintendo Switch Console Bundle', brand: 'Nintendo', category: 'Electronics/Games', price: 250, condition: 3 },
    { id: '3', name: 'Coach Swagger Shoulder Bag', brand: 'Coach', category: 'Women/Bags/Shoulder Bags', price: 95, condition: 1 },
    { id: '4', name: 'Vintage Levi 501 Jeans', brand: "Levi's", category: 'Men/Jeans', price: 45, condition: 3 },
    { id: '5', name: 'Apple AirPods Pro 2nd Gen', brand: 'Apple', category: 'Electronics/Audio', price: 180, condition: 1 },
    { id: '6', name: 'Patagonia Better Sweater Fleece', brand: 'Patagonia', category: 'Men/Sweaters', price: 55, condition: 2 },
    { id: '7', name: 'KitchenAid Stand Mixer 5qt', brand: 'KitchenAid', category: 'Home/Kitchen', price: 189, condition: 3 },
    { id: '8', name: 'Funko Pop Collection (12 pcs)', brand: 'Funko', category: 'Vintage & Collectibles', price: 75, condition: 1 },
    { id: '9', name: 'The North Face Nuptse Jacket', brand: 'The North Face', category: 'Men/Outerwear', price: 150, condition: 2 },
];

const CONDITIONS_MAP: Record<number, string> = {
    1: 'New with tags',
    2: 'New no tags',
    3: 'Good',
    4: 'Fair',
    5: 'Poor',
};

export default function ExplorePage() {
    const [query, setQuery] = useState('');

    const filtered = SAMPLE_PRODUCTS.filter(
        (p) =>
            p.name.toLowerCase().includes(query.toLowerCase()) ||
            p.brand.toLowerCase().includes(query.toLowerCase()) ||
            p.category.toLowerCase().includes(query.toLowerCase())
    );

    return (
        <div className={styles.page}>
            <div className={styles.container}>
                <div className={styles.header}>
                    <h1>Explore Products</h1>
                    <p>Browse the marketplace catalog and see pricing patterns</p>
                </div>

                {/* Stats */}
                <div className={styles.statsGrid}>
                    <div className={styles.statCard}>
                        <div className={styles.statValue}>1.48M</div>
                        <div className={styles.statLabel}>Total Products</div>
                    </div>
                    <div className={styles.statCard}>
                        <div className={styles.statValue}>4,809</div>
                        <div className={styles.statLabel}>Unique Brands</div>
                    </div>
                    <div className={styles.statCard}>
                        <div className={styles.statValue}>871</div>
                        <div className={styles.statLabel}>Categories</div>
                    </div>
                </div>

                {/* Search */}
                <div className={styles.searchBox}>
                    <input
                        className={styles.searchInput}
                        placeholder="Search by product name, brand, or category..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                    />
                    <button className={styles.searchBtn}>Search</button>
                </div>

                {/* Products */}
                <div className={styles.productsGrid}>
                    {filtered.length === 0 && (
                        <div className={styles.emptyState}>
                            No products match your search. Try a different query.
                        </div>
                    )}
                    {filtered.map((product) => (
                        <div key={product.id} className={styles.productCard}>
                            <div className={styles.productName}>{product.name}</div>
                            <div className={styles.productMeta}>
                                <span className={`${styles.badge} ${styles.brandBadge}`}>
                                    {product.brand}
                                </span>
                                <span className={styles.badge}>
                                    {product.category.split('/')[0]}
                                </span>
                                <span className={styles.badge}>
                                    {CONDITIONS_MAP[product.condition]}
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
