import styles from './about.module.css';

const PIPELINE_STEPS = [
    { icon: '📦', label: 'Raw Data', sub: '1.48M products' },
    { icon: '⚙️', label: 'Preprocess', sub: 'Clean & tokenize' },
    { icon: '🧠', label: 'Train', sub: 'BiLSTM + MLP' },
    { icon: '📊', label: 'Evaluate', sub: 'RMSLE metrics' },
    { icon: '🚀', label: 'Serve', sub: 'FastAPI + ONNX' },
];

const TECH_STACK = [
    { name: 'PyTorch', desc: 'Deep learning framework' },
    { name: 'FastAPI', desc: 'High-performance API' },
    { name: 'MongoDB', desc: 'Product catalog storage' },
    { name: 'ONNX Runtime', desc: 'Optimized inference' },
    { name: 'Next.js', desc: 'React framework' },
    { name: 'Recharts', desc: 'Data visualization' },
];

export default function AboutPage() {
    return (
        <div className={styles.page}>
            <div className={styles.container}>
                <div className={styles.header}>
                    <h1>About PriceScope</h1>
                    <p>
                        A multimodal deep learning system that predicts marketplace product prices
                        by analyzing text descriptions, brand reputation, category context, and
                        item condition — trained on 1.48 million real Mercari listings.
                    </p>
                </div>

                <div className={styles.section}>
                    <h2>How It Works</h2>
                    <p>
                        The model processes each product through three parallel encoders: two
                        bidirectional LSTMs for the product name and description, and an embedding
                        network for categorical features like brand, category, and condition. These
                        representations are fused through a dense MLP that outputs the predicted
                        log-price, which is converted back to dollars.
                    </p>

                    <div className={styles.pipeline}>
                        {PIPELINE_STEPS.map((step) => (
                            <div key={step.label} className={styles.pipelineStep}>
                                <div className={styles.pipelineIcon}>{step.icon}</div>
                                <div className={styles.pipelineLabel}>{step.label}</div>
                                <div className={styles.pipelineSub}>{step.sub}</div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className={styles.section}>
                    <h2>Dataset</h2>
                    <p>
                        Trained on the Mercari Price Suggestion Challenge dataset from Kaggle.
                        The data includes product names, full descriptions, hierarchical categories
                        (3 levels deep), brand names (4,809 unique brands), item condition ratings,
                        and shipping preferences. Prices are log-transformed to handle the heavy
                        right skew in marketplace pricing.
                    </p>
                </div>

                <div className={styles.section}>
                    <h2>Tech Stack</h2>
                    <div className={styles.techGrid}>
                        {TECH_STACK.map((tech) => (
                            <div key={tech.name} className={styles.techCard}>
                                <h4>{tech.name}</h4>
                                <p>{tech.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>

                <div className={styles.section}>
                    <h2>Links</h2>
                    <div className={styles.linkRow}>
                        <a
                            href="https://github.com/Technocrat-dev/Marketplace-Price-Prediction"
                            target="_blank"
                            rel="noopener noreferrer"
                            className={styles.linkBtn}
                        >
                            ⭐ GitHub Repository
                        </a>
                        <a
                            href="https://www.kaggle.com/c/mercari-price-suggestion-challenge"
                            target="_blank"
                            rel="noopener noreferrer"
                            className={styles.linkBtn}
                        >
                            📊 Kaggle Challenge
                        </a>
                    </div>
                </div>
            </div>
        </div>
    );
}
