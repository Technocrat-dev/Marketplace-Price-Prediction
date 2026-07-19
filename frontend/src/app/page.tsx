'use client';

import { useState } from 'react';
import { useSession } from 'next-auth/react';
import styles from './page.module.css';
import { predictPrice, analyzeListing, formatPrice, CONDITIONS, type PredictionResponse, type AnalyzeResponse } from '@/lib/api';

export default function PredictPage() {
  const { data: session } = useSession();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('');
  const [brand, setBrand] = useState('');
  const [condition, setCondition] = useState(3);
  const [shipping, setShipping] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [analysisError, setAnalysisError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    setLoading(true);
    setError('');
    setAnalysis(null);
    setAnalysisError('');

    try {
      const response = await predictPrice({
        name,
        item_description: description,
        category_name: category,
        brand_name: brand || 'unknown',
        item_condition_id: condition,
        shipping,
      }, session?.user?.email || undefined);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!name.trim() || analyzing) return;

    setAnalyzing(true);
    setAnalysisError('');

    try {
      const response = await analyzeListing({
        name,
        item_description: description,
        category_name: category,
        brand_name: brand || 'unknown',
        item_condition_id: condition,
        shipping,
      });
      setAnalysis(response);
    } catch (err) {
      setAnalysisError(err instanceof Error ? err.message : 'AI analysis failed');
      setAnalysis(null);
    } finally {
      setAnalyzing(false);
    }
  };

  const agreementLabel: Record<string, string> = {
    close: 'Models agree',
    moderate: 'Some disagreement',
    divergent: 'Models disagree',
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
                  <label htmlFor="product-name" className={styles.label}>What are you selling?</label>
                  <input
                    id="product-name"
                    className={styles.input}
                    type="text"
                    placeholder="e.g. Nike Air Max 90 Classic"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>

                <div className={styles.field}>
                  <label htmlFor="product-description" className={styles.label}>Describe it</label>
                  <textarea
                    id="product-description"
                    className={styles.textarea}
                    placeholder="Condition details, size, color, any flaws..."
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                  />
                </div>

                <div className={styles.fieldRow}>
                  <div className={styles.field}>
                    <label htmlFor="product-category" className={styles.label}>Category</label>
                    <input
                      id="product-category"
                      className={styles.input}
                      type="text"
                      placeholder="e.g. Men/Shoes/Athletic"
                      value={category}
                      onChange={(e) => setCategory(e.target.value)}
                    />
                  </div>
                  <div className={styles.field}>
                    <label htmlFor="product-brand" className={styles.label}>Brand</label>
                    <input
                      id="product-brand"
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
                    className={`${styles.conditionBtn} ${condition === c.value ? styles.conditionBtnActive : ''}`}
                    onClick={() => setCondition(c.value)}
                    aria-label={`Condition: ${c.label}`}
                    aria-pressed={condition === c.value}
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
                role="switch"
                aria-checked={shipping === 1}
                aria-label="Shipping payment toggle"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setShipping(shipping === 1 ? 0 : 1);
                  }
                }}
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
              id="predict-submit"
              className={styles.submitBtn}
              disabled={loading || !name.trim()}
            >
              {loading ? 'Analyzing...' : 'Predict Price'}
            </button>
          </form>

          {/* Right: Results */}
          <div className={styles.resultPanel} aria-live="polite">
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

                  {!analysis && (
                    <button
                      type="button"
                      className={styles.analyzeBtn}
                      onClick={handleAnalyze}
                      disabled={analyzing}
                    >
                      {analyzing ? 'Asking Claude...' : '✦ Get AI Listing Analysis'}
                    </button>
                  )}

                  {analysisError && (
                    <div className={styles.analysisError}>{analysisError}</div>
                  )}
                </>
              )}
            </div>

            {analysis && !loading && (
              <div className={styles.analysisCard} aria-live="polite">
                <div className={styles.analysisHeader}>
                  <span className={styles.analysisTitle}>✦ AI Listing Analysis</span>
                  <span className={styles.analysisScore}>
                    {analysis.ai_analysis.listing_score}/10
                  </span>
                </div>

                <div className={styles.comparisonRow}>
                  <div className={styles.comparisonCol}>
                    <div className={styles.comparisonLabel}>ML Model</div>
                    <div className={styles.comparisonValue}>
                      {formatPrice(analysis.comparison.model_price)}
                    </div>
                  </div>
                  <div className={styles.comparisonCol}>
                    <div className={styles.comparisonLabel}>Claude</div>
                    <div className={styles.comparisonValue}>
                      {formatPrice(analysis.comparison.llm_price)}
                    </div>
                  </div>
                  <span
                    className={`${styles.agreementBadge} ${
                      styles[`agreement_${analysis.comparison.agreement}`] || ''
                    }`}
                  >
                    {agreementLabel[analysis.comparison.agreement] || analysis.comparison.agreement}
                  </span>
                </div>

                <p className={styles.analysisReasoning}>
                  {analysis.ai_analysis.price_reasoning}
                </p>

                {analysis.ai_analysis.strengths.length > 0 && (
                  <div className={styles.analysisSection}>
                    <div className={styles.analysisSectionLabel}>Strengths</div>
                    <ul className={styles.analysisList}>
                      {analysis.ai_analysis.strengths.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {analysis.ai_analysis.improvements.length > 0 && (
                  <div className={styles.analysisSection}>
                    <div className={styles.analysisSectionLabel}>Improve your listing</div>
                    <ul className={styles.analysisList}>
                      {analysis.ai_analysis.improvements.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {analysis.ai_analysis.suggested_title && (
                  <div className={styles.analysisSection}>
                    <div className={styles.analysisSectionLabel}>Suggested title</div>
                    <div className={styles.suggestedTitle}>
                      &ldquo;{analysis.ai_analysis.suggested_title}&rdquo;
                    </div>
                  </div>
                )}

                <div className={styles.analysisFooter}>
                  Powered by {analysis.llm_model}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
