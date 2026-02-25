"""
SHAP Explainability for the Mercari Price Prediction Model.

Generates feature importance explanations for individual predictions using
SHAP (SHapley Additive exPlanations). Works with the baseline models for
tabular-level explanations.

Usage:
    python scripts/explain.py
    python scripts/explain.py --sample 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def explain_baselines(data_dir: Path, n_samples: int = 50) -> dict:
    """Generate SHAP explanations for baseline models."""
    try:
        import shap
    except ImportError:
        print("[ERROR] SHAP not installed. Run: pip install shap")
        return {}
    
    # Load data
    tabular_test = np.load(data_dir / "test_tabular.npy")
    X_test = tabular_test[:, :-1]
    tabular_test[:, -1]
    
    tabular_train = np.load(data_dir / "train_tabular.npy")
    X_train = tabular_train[:, :-1]
    y_train = tabular_train[:, -1]
    
    feature_names = ["main_cat", "sub_cat1", "sub_cat2", "brand", "condition", "shipping"]
    
    # Sample for speed
    if n_samples < len(X_test):
        idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_explain = X_test[idx]
    else:
        X_explain = X_test[:n_samples]
    
    results = {}
    
    # Try XGBoost SHAP
    try:
        import xgboost as xgb
        
        print("[INFO] Training XGBoost for SHAP...")
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(X_train, y_train)
        
        print("[INFO] Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        
        # Global feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(feature_names, mean_abs_shap.tolist()))
        importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
        
        # Sample explanations
        sample_explanations = []
        for i in range(min(5, len(X_explain))):
            explanation = dict(zip(feature_names, shap_values[i].tolist()))
            sample_explanations.append({
                "features": dict(zip(feature_names, X_explain[i].tolist())),
                "shap_values": explanation,
                "base_value": float(explainer.expected_value),
                "predicted_log_price": float(model.predict(X_explain[i:i+1])[0]),
            })
        
        results["xgboost"] = {
            "global_importance": importance,
            "sample_explanations": sample_explanations,
            "n_samples": n_samples,
        }
        
        print("[SUCCESS] XGBoost SHAP completed")
        print(f"  Top features: {list(importance.items())[:3]}")
        
    except ImportError:
        print("[WARN] XGBoost not available, skipping")
    except Exception as e:
        print(f"[ERROR] XGBoost SHAP failed: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP explanations")
    parser.add_argument("--sample", type=int, default=50, help="Number of samples to explain")
    args = parser.parse_args()
    
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("[ERROR] Processed data not found. Run preprocessing first.")
        return
    
    results = explain_baselines(data_dir, n_samples=args.sample)
    
    if results:
        output_path = Path("outputs/shap_explanations.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SUCCESS] Explanations saved to {output_path}")
    else:
        print("\n[WARN] No explanations generated. Install shap and xgboost.")


if __name__ == "__main__":
    main()
