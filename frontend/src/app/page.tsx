'use client';

import { useState } from 'react';
import styles from './page.module.css';
import { predictPrice, formatPrice, CONDITIONS, type PredictionResponse } from '@/lib/api';

export default function PredictPage() {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('');
  const [brand, setBrand] = useState('');
  const [condition, setCondition] = useState(3);
  const [shipping, setShipping] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    setLoading(true);
    setError('');

    try {
      const response = await predictPrice({
        name,
        item_description: description,
        category_name: category,
        brand_name: brand || 'unknown',
        item_condition_id: condition,
        shipping,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.hero}>
        <div className={styles.heroHeader}>
          <h1>Predict a fair price</h1>
          <p>Enter product details and our model will estimate the marketplace value.</p>
        </div>

        <div className={styles.grid}>
          {/* Left: Form */}
          <form className={styles.formPanel} onSubmit={handleSubmit}>
            <div className={styles.formSection}>
              <div className={styles.sectionLabel}>Product Info</div>
              <div className={styles.fieldGroup}>
                <div className={styles.field}>
                  <label className={styles.label}>What are you selling?</label>
                  <input
                    className={styles.input}
                    type="text"
                    placeholder="e.g. Nike Air Max 90 Classic"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>

                <div className={styles.field}>
                  <label className={styles.label}>Describe it</label>
                  <textarea
                    className={styles.textarea}
                    placeholder="Condition details, size, color, any flaws..."
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                  />
                </div>

                <div className={styles.fieldRow}>
                  <div className={styles.field}>
                    <label className={styles.label}>Category</label>
                    <input
                      className={styles.input}
                      type="text"
                      placeholder="e.g. Men/Shoes/Athletic"
                      value={category}
                      onChange={(e) => setCategory(e.target.value)}
                    />
                  </div>
                  <div className={styles.field}>
                    <label className={styles.label}>Brand</label>
                    <input
                      className={styles.input}
                      type="text"
                      placeholder="e.g. Nike"
                      value={brand}
                      onChange={(e) => setBrand(e.target.value)}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className={styles.formSection}>
              <div className={styles.sectionLabel}>Condition</div>
              <div className={styles.conditionRow}>
                {CONDITIONS.map((c) => (
                  <button
                    type="button"
                    key={c.value}
                    className={`${styles.conditionBtn} ${condition === c.value ? styles.conditionBtnActive : ''
                      }`}
                    onClick={() => setCondition(c.value)}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.formSection}>
              <div className={styles.sectionLabel}>Shipping</div>
              <div
                className={styles.toggle}
                onClick={() => setShipping(shipping === 1 ? 0 : 1)}
              >
                <div className={`${styles.toggleTrack} ${shipping === 1 ? styles.toggleTrackActive : ''}`}>
                  <div className={`${styles.toggleThumb} ${shipping === 1 ? styles.toggleThumbActive : ''}`} />
                </div>
                <span className={styles.toggleLabel}>
                  {shipping === 1 ? 'Seller pays shipping' : 'Buyer pays shipping'}
                </span>
              </div>
            </div>

            <button
              type="submit"
              className={styles.submitBtn}
              disabled={loading || !name.trim()}
            >
              {loading ? 'Analyzing...' : 'Predict Price'}
            </button>
          </form>

          {/* Right: Results */}
          <div className={styles.resultPanel}>
            <div className={styles.resultCard}>
              {loading && (
                <div className={styles.loading}>
                  <div className={styles.spinner} />
                  Running inference...
                </div>
              )}

              {error && (
                <div className={styles.error}>{error}</div>
              )}

              {!result && !loading && !error && (
                <div className={styles.emptyState}>
                  <div className={styles.emptyIcon}>💰</div>
                  <div className={styles.emptyText}>
                    Fill in the product details on the left and hit predict to see the estimated price.
                  </div>
                </div>
              )}

              {result && !loading && (
                <>
                  <div className={styles.priceSection}>
                    <div className={styles.priceLabel}>Estimated Price</div>
                    <div className={styles.priceValue}>
                      {formatPrice(result.predicted_price)}
                    </div>
                  </div>

                  <div className={styles.confidenceSection}>
                    <div className={styles.confidenceLabel}>Confidence Range</div>
                    <div className={styles.confidenceBar}>
                      <div
                        className={styles.confidenceFill}
                        style={{
                          width: `${Math.min(100, (result.predicted_price / result.confidence_range.high) * 100)}%`,
                        }}
                      />
                    </div>
                    <div className={styles.confidenceRange}>
                      <span>{formatPrice(result.confidence_range.low)}</span>
                      <span>{formatPrice(result.confidence_range.high)}</span>
                    </div>
                  </div>

                  <div className={styles.summarySection}>
                    {Object.entries(result.input_summary).map(([key, value]) => (
                      <div key={key} className={styles.summaryRow}>
                        <span className={styles.summaryKey}>{key}</span>
                        <span className={styles.summaryValue}>{value}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
