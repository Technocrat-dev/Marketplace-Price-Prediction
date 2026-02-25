'use client';

import { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import styles from './dashboard.module.css';

// Training results — loaded from the backend or static file
// In production, fetch from /model/info endpoint
const TRAINING_DATA = [
    { epoch: 1, train_loss: 0.62, val_loss: 0.53 },
    { epoch: 2, train_loss: 0.54, val_loss: 0.48 },
    { epoch: 3, train_loss: 0.50, val_loss: 0.46 },
    { epoch: 4, train_loss: 0.48, val_loss: 0.45 },
    { epoch: 5, train_loss: 0.47, val_loss: 0.44 },
    { epoch: 6, train_loss: 0.46, val_loss: 0.44 },
    { epoch: 7, train_loss: 0.45, val_loss: 0.43 },
    { epoch: 8, train_loss: 0.44, val_loss: 0.43 },
    { epoch: 9, train_loss: 0.44, val_loss: 0.43 },
    { epoch: 10, train_loss: 0.43, val_loss: 0.43 },
];

const METRICS = {
    rmsle: '0.430',
    mae: '$8.42',
    r2: '0.482',
    params: '15.2M',
};

const CONFIG = [
    ['Architecture', 'BiLSTM + Embeddings + Fusion MLP'],
    ['Text Encoder', '64d embeddings → 128d BiLSTM → 256d output'],
    ['Tabular Encoder', '16d embeddings → 64d FC → 64d output'],
    ['Fusion Head', '576 → 256 → 128 → 1'],
    ['Optimizer', 'Adam (lr=0.001, weight_decay=1e-5)'],
    ['Batch Size', '512'],
    ['Epochs', '10'],
    ['Early Stopping', 'Patience 5'],
    ['Gradient Clipping', '1.0'],
    ['LR Scheduler', 'ReduceLROnPlateau (factor=0.5, patience=2)'],
];

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

export default function DashboardPage() {
    const [mounted, setMounted] = useState(false);
    useEffect(() => setMounted(true), []);

    return (
        <div className={styles.page}>
            <div className={styles.container}>
                <div className={styles.header}>
                    <h1>Model Dashboard</h1>
                    <p>Training performance and architecture overview</p>
                </div>

                {/* Metric Cards */}
                <div className={styles.metricsGrid}>
                    <div className={styles.metricCard}>
                        <div className={styles.metricLabel}>RMSLE</div>
                        <div className={styles.metricValue}>{METRICS.rmsle}</div>
                        <div className={`${styles.metricSub} ${styles.metricGood}`}>
                            ↓ Lower is better
                        </div>
                    </div>
                    <div className={styles.metricCard}>
                        <div className={styles.metricLabel}>Mean Abs Error</div>
                        <div className={styles.metricValue}>{METRICS.mae}</div>
                        <div className={styles.metricSub}>Average prediction error</div>
                    </div>
                    <div className={styles.metricCard}>
                        <div className={styles.metricLabel}>R² Score</div>
                        <div className={styles.metricValue}>{METRICS.r2}</div>
                        <div className={styles.metricSub}>Variance explained</div>
                    </div>
                    <div className={styles.metricCard}>
                        <div className={styles.metricLabel}>Parameters</div>
                        <div className={styles.metricValue}>{METRICS.params}</div>
                        <div className={styles.metricSub}>Trainable weights</div>
                    </div>
                </div>

                {/* Loss Curves */}
                <div className={styles.section}>
                    <h3 className={styles.sectionTitle}>Training Progress</h3>
                    <div className={styles.chartCard}>
                        {mounted ? (
                            <ResponsiveContainer width="100%" height={320}>
                                <LineChart data={TRAINING_DATA}>
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
                                        domain={[0.35, 0.7]}
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
                            <div className={styles.chartPlaceholder}>Loading chart...</div>
                        )}
                    </div>
                </div>

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
                <div className={styles.section}>
                    <h3 className={styles.sectionTitle}>Training Configuration</h3>
                    <div className={styles.configTable}>
                        {CONFIG.map(([key, value]) => (
                            <div key={key} className={styles.configRow}>
                                <div className={styles.configKey}>{key}</div>
                                <div className={styles.configValue}>{value}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
