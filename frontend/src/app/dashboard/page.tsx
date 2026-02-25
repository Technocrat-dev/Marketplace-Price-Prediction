'use client';

import { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import styles from './dashboard.module.css';
import {
    fetchModelInfo, fetchRecentPredictions, formatPrice,
    type ModelInfoResponse, type RecentPredictionItem,
} from '@/lib/api';

const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload) return null;
    return (
        <div style={{
            background: '#12121A',
            border: '1px solid #2A2A3A',
            borderRadius: '8px',
            padding: '10px 14px',
            fontSize: '13px',
        }}>
            <div style={{ color: '#E8E8ED', fontWeight: 600, marginBottom: 4 }}>
                Epoch {label}
            </div>
            {payload.map((p: any) => (
                <div key={p.dataKey} style={{ color: p.color, fontFamily: 'JetBrains Mono' }}>
                    {p.name}: {p.value.toFixed(4)}
                </div>
            ))}
        </div>
    );
};

/* Skeleton loader for metric cards */
function MetricSkeleton() {
    return (
        <div className={styles.metricCard}>
            <div className={`${styles.skeleton} ${styles.skeletonLabel}`} />
            <div className={`${styles.skeleton} ${styles.skeletonValue}`} />
            <div className={`${styles.skeleton} ${styles.skeletonSub}`} />
        </div>
    );
}

export default function DashboardPage() {
    const [mounted, setMounted] = useState(false);
    const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
    const [predictions, setPredictions] = useState<RecentPredictionItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        setMounted(true);

        async function loadData() {
            try {
                const [info, recent] = await Promise.allSettled([
                    fetchModelInfo(),
                    fetchRecentPredictions(10),
                ]);

                if (info.status === 'fulfilled') setModelInfo(info.value);
                if (recent.status === 'fulfilled') setPredictions(recent.value.predictions);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load data');
            } finally {
                setLoading(false);
            }
        }

        loadData();
    }, []);

    const metrics = modelInfo ? {
        rmsle: modelInfo.test_metrics.rmsle?.toFixed(3) || '—',
        mae: `$${modelInfo.test_metrics.mae?.toFixed(2) || '—'}`,
        r2: modelInfo.test_metrics.r2?.toFixed(3) || '—',
        params: `${(modelInfo.model_parameters / 1e6).toFixed(1)}M`,
    } : null;

    const config = modelInfo
        ? Object.entries(modelInfo.config_summary)
        : [];

    return (
        <div className={styles.page}>
            <div className={styles.container}>
                <div className={styles.header}>
                    <h1>Model Dashboard</h1>
                    <p>Training performance and architecture overview</p>
                </div>

                {error && (
                    <div className={styles.errorBanner}>{error}</div>
                )}

                {/* Metric Cards */}
                <div className={styles.metricsGrid}>
                    {loading ? (
                        <>
                            <MetricSkeleton />
                            <MetricSkeleton />
                            <MetricSkeleton />
                            <MetricSkeleton />
                        </>
                    ) : metrics ? (
                        <>
                            <div className={styles.metricCard}>
                                <div className={styles.metricLabel}>RMSLE</div>
                                <div className={styles.metricValue}>{metrics.rmsle}</div>
                                <div className={`${styles.metricSub} ${styles.metricGood}`}>
                                    ↓ Lower is better
                                </div>
                            </div>
                            <div className={styles.metricCard}>
                                <div className={styles.metricLabel}>Mean Abs Error</div>
                                <div className={styles.metricValue}>{metrics.mae}</div>
                                <div className={styles.metricSub}>Average prediction error</div>
                            </div>
                            <div className={styles.metricCard}>
                                <div className={styles.metricLabel}>R² Score</div>
                                <div className={styles.metricValue}>{metrics.r2}</div>
                                <div className={styles.metricSub}>Variance explained</div>
                            </div>
                            <div className={styles.metricCard}>
                                <div className={styles.metricLabel}>Parameters</div>
                                <div className={styles.metricValue}>{metrics.params}</div>
                                <div className={styles.metricSub}>Trainable weights</div>
                            </div>
                        </>
                    ) : null}
                </div>

                {/* Loss Curves */}
                <div className={styles.section}>
                    <h3 className={styles.sectionTitle}>Training Progress</h3>
                    <div className={styles.chartCard}>
                        {!mounted || loading ? (
                            <div className={styles.chartPlaceholder}>Loading chart...</div>
                        ) : modelInfo?.loss_curve ? (
                            <ResponsiveContainer width="100%" height={320}>
                                <LineChart data={modelInfo.loss_curve}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2A2A3A" />
                                    <XAxis
                                        dataKey="epoch"
                                        stroke="#555568"
                                        fontSize={12}
                                        tickLine={false}
                                    />
                                    <YAxis
                                        stroke="#555568"
                                        fontSize={12}
                                        tickLine={false}
                                        domain={['auto', 'auto']}
                                    />
                                    <Tooltip content={<CustomTooltip />} />
                                    <Legend
                                        wrapperStyle={{ fontSize: 13, color: '#8888A0' }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="train_loss"
                                        name="Train Loss"
                                        stroke="#6C5CE7"
                                        strokeWidth={2}
                                        dot={{ fill: '#6C5CE7', r: 4 }}
                                        activeDot={{ r: 6 }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="val_loss"
                                        name="Val Loss"
                                        stroke="#a855f7"
                                        strokeWidth={2}
                                        dot={{ fill: '#a855f7', r: 4 }}
                                        activeDot={{ r: 6 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className={styles.chartPlaceholder}>No training data available</div>
                        )}
                    </div>
                </div>

                {/* Recent Predictions */}
                {predictions.length > 0 && (
                    <div className={styles.section}>
                        <h3 className={styles.sectionTitle}>Recent Predictions</h3>
                        <div className={styles.predictionTable}>
                            <div className={styles.predictionHeader}>
                                <span>Product</span>
                                <span>Brand</span>
                                <span>Predicted</span>
                                <span>Range</span>
                                <span>Time</span>
                            </div>
                            {predictions.map((pred, i) => (
                                <div key={i} className={styles.predictionRow}>
                                    <span className={styles.predName}>{pred.product_name}</span>
                                    <span className={styles.predBrand}>{pred.brand}</span>
                                    <span className={styles.predPrice}>{formatPrice(pred.predicted_price)}</span>
                                    <span className={styles.predRange}>
                                        {formatPrice(pred.confidence_low)} – {formatPrice(pred.confidence_high)}
                                    </span>
                                    <span className={styles.predTime}>
                                        {new Date(pred.predicted_at).toLocaleString()}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Architecture */}
                <div className={styles.section}>
                    <h3 className={styles.sectionTitle}>Model Architecture</h3>
                    <div className={styles.archGrid}>
                        <div className={styles.archCard}>
                            <h4>Name Encoder</h4>
                            <p>Processes product titles through embeddings and a bidirectional LSTM to capture naming patterns.</p>
                            <span className={styles.archTag}>256 features</span>
                        </div>
                        <div className={styles.archCard}>
                            <h4>Description Encoder</h4>
                            <p>Processes longer descriptions with the same BiLSTM architecture for contextual understanding.</p>
                            <span className={styles.archTag}>256 features</span>
                        </div>
                        <div className={styles.archCard}>
                            <h4>Tabular Encoder</h4>
                            <p>Embeds categorical features (brand, category, condition) into dense learned representations.</p>
                            <span className={styles.archTag}>64 features</span>
                        </div>
                    </div>
                </div>

                {/* Training Config */}
                {config.length > 0 && (
                    <div className={styles.section}>
                        <h3 className={styles.sectionTitle}>Training Configuration</h3>
                        <div className={styles.configTable}>
                            {config.map(([key, value]) => (
                                <div key={key} className={styles.configRow}>
                                    <div className={styles.configKey}>{key}</div>
                                    <div className={styles.configValue}>{value}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
