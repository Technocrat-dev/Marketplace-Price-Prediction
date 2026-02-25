"""
Baseline Model Comparisons for the Mercari Price Prediction Engine.

Trains tabular baselines (XGBoost, LightGBM, Ridge) and compares their
RMSLE against the deep learning model. Uses the same preprocessed data
and train/val/test splits to ensure a fair comparison.

Usage:
    python scripts/train_baselines.py

Results are saved to outputs/baseline_results.json.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================================
# Data Loading
# =========================================================================

def load_split(data_dir: Path, split: str):
    """Load a preprocessed data split. Returns (features, targets)."""
    tabular = np.load(data_dir / f"{split}_tabular.npy")
    
    # Columns: [main_cat, sub_cat1, sub_cat2, brand_name, condition, shipping, log_price]
    features = tabular[:, :-1]  # All columns except log_price
    targets = tabular[:, -1]    # log_price
    
    return features, targets


def rmsle_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Compute RMSLE when inputs are already in log space."""
    return np.sqrt(mean_squared_error(y_true_log, y_pred_log))


def evaluate_model(name: str, y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Evaluate a model and return metrics dict."""
    # Clip predictions to valid range
    y_pred_log = np.clip(y_pred_log, 0, None)
    
    # Convert from log space for MAE
    y_true_price = np.expm1(y_true_log)
    y_pred_price = np.expm1(y_pred_log)
    
    rmsle = rmsle_from_log(y_true_log, y_pred_log)
    mae = mean_absolute_error(y_true_price, y_pred_price)
    r2 = r2_score(y_true_log, y_pred_log)
    median_ae = np.median(np.abs(y_true_price - y_pred_price))
    
    return {
        "model": name,
        "rmsle": round(float(rmsle), 4),
        "mae": round(float(mae), 2),
        "median_ae": round(float(median_ae), 2),
        "r2": round(float(r2), 4),
        "mean_pred_price": round(float(y_pred_price.mean()), 2),
        "mean_actual_price": round(float(y_true_price.mean()), 2),
    }


# =========================================================================
# Baseline Models
# =========================================================================

def train_ridge(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Ridge regression baseline."""
    print("\n[Ridge Regression]")
    start = time.time()
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")
    
    val_metrics = evaluate_model("Ridge", y_val, model.predict(X_val))
    test_metrics = evaluate_model("Ridge", y_test, model.predict(X_test))
    
    print(f"  Val RMSLE:  {val_metrics['rmsle']:.4f}")
    print(f"  Test RMSLE: {test_metrics['rmsle']:.4f}")
    print(f"  Test MAE:   ${test_metrics['mae']:.2f}")
    print(f"  Test R²:    {test_metrics['r2']:.4f}")
    
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_time_seconds": round(elapsed, 1),
    }


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost baseline."""
    try:
        import xgboost as xgb
    except ImportError:
        print("\n[XGBoost] Skipped — install with: pip install xgboost")
        return None
    
    print("\n[XGBoost]")
    start = time.time()
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=20,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    elapsed = time.time() - start
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Best iteration: {best_iteration}")
    
    val_metrics = evaluate_model("XGBoost", y_val, model.predict(X_val))
    test_metrics = evaluate_model("XGBoost", y_test, model.predict(X_test))
    
    # Feature importance
    importance = dict(zip(
        ["main_cat", "sub_cat1", "sub_cat2", "brand", "condition", "shipping"],
        model.feature_importances_.tolist()
    ))
    
    print(f"  Val RMSLE:  {val_metrics['rmsle']:.4f}")
    print(f"  Test RMSLE: {test_metrics['rmsle']:.4f}")
    print(f"  Test MAE:   ${test_metrics['mae']:.2f}")
    print(f"  Test R²:    {test_metrics['r2']:.4f}")
    print(f"  Top features: {sorted(importance.items(), key=lambda x: -x[1])[:3]}")
    
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_time_seconds": round(elapsed, 1),
        "best_iteration": best_iteration,
        "feature_importance": importance,
    }


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train LightGBM baseline."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("\n[LightGBM] Skipped — install with: pip install lightgbm")
        return None
    
    print("\n[LightGBM]")
    start = time.time()
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    
    elapsed = time.time() - start
    best_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Best iteration: {best_iteration}")
    
    val_metrics = evaluate_model("LightGBM", y_val, model.predict(X_val))
    test_metrics = evaluate_model("LightGBM", y_test, model.predict(X_test))
    
    # Feature importance
    importance = dict(zip(
        ["main_cat", "sub_cat1", "sub_cat2", "brand", "condition", "shipping"],
        model.feature_importances_.tolist()
    ))
    
    print(f"  Val RMSLE:  {val_metrics['rmsle']:.4f}")
    print(f"  Test RMSLE: {test_metrics['rmsle']:.4f}")
    print(f"  Test MAE:   ${test_metrics['mae']:.2f}")
    print(f"  Test R²:    {test_metrics['r2']:.4f}")
    print(f"  Top features: {sorted(importance.items(), key=lambda x: -x[1])[:3]}")
    
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_time_seconds": round(elapsed, 1),
        "best_iteration": best_iteration,
        "feature_importance": {k: int(v) for k, v in importance.items()},
    }


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("Mercari Price Prediction — Baseline Comparisons")
    print("=" * 60)
    
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("[ERROR] Processed data not found. Run preprocessing first.")
        return
    
    # Load splits
    print("\n[INFO] Loading data...")
    X_train, y_train = load_split(data_dir, "train")
    X_val, y_val = load_split(data_dir, "val")
    X_test, y_test = load_split(data_dir, "test")
    print(f"  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"  Features: {X_train.shape[1]} (5 categoricals + shipping)")
    
    # Load deep learning results for comparison
    dl_results_path = Path("outputs/training_results.json")
    dl_metrics = None
    if dl_results_path.exists():
        with open(dl_results_path) as f:
            dl_data = json.load(f)
            dl_metrics = dl_data.get("test_metrics", {})
    
    # Train baselines
    results = {}
    
    ridge_result = train_ridge(X_train, y_train, X_val, y_val, X_test, y_test)
    if ridge_result:
        results["Ridge"] = ridge_result
    
    xgb_result = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    if xgb_result:
        results["XGBoost"] = xgb_result
    
    lgb_result = train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
    if lgb_result:
        results["LightGBM"] = lgb_result
    
    # Comparison Table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<20} {'RMSLE':>8} {'MAE':>10} {'R²':>8} {'Time':>8}")
    print("-" * 60)
    
    if dl_metrics:
        print(f"{'BiLSTM+MLP (DL)':<20} {dl_metrics['rmsle']:>8.4f} "
              f"${dl_metrics['mae']:>8.2f} {dl_metrics['r2']:>8.4f} {'—':>8}")
    
    for name, data in results.items():
        m = data["test_metrics"]
        t = data["training_time_seconds"]
        print(f"{name:<20} {m['rmsle']:>8.4f} ${m['mae']:>8.2f} {m['r2']:>8.4f} {t:>7.1f}s")
    
    print("-" * 60)
    
    # Save results
    output = {
        "baselines": results,
        "deep_learning": dl_metrics,
        "data_shape": {
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "num_features": int(X_train.shape[1]),
        },
    }
    
    output_path = Path("outputs/baseline_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[SUCCESS] Results saved to {output_path}")


if __name__ == "__main__":
    main()
