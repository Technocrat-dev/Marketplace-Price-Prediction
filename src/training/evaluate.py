"""
Evaluation metrics for the Mercari Price Prediction model.

Computes:
- RMSLE (Root Mean Squared Logarithmic Error) — primary metric
- MAE (Mean Absolute Error) — in original dollar scale
- R² (Coefficient of Determination) — explained variance
"""

import numpy as np
import torch
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def compute_rmsle(pred: np.ndarray, actual: np.ndarray) -> float:
    """
    Compute RMSLE between predictions and actuals.
    
    Both inputs should be in ORIGINAL price scale (not log-transformed).
    """
    pred = np.maximum(pred, 0)  # Clamp negatives
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2)))


def compute_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Compute Mean Absolute Error in original price scale."""
    return float(np.mean(np.abs(pred - actual)))


def compute_r2(pred: np.ndarray, actual: np.ndarray) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_median_ae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Compute Median Absolute Error — more robust to outliers than MAE."""
    return float(np.median(np.abs(pred - actual)))


def evaluate_predictions(
    pred_log: np.ndarray, 
    actual_log: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        pred_log: Predicted log1p(price) values
        actual_log: Actual log1p(price) values
    
    Returns:
        Dict with all metrics
    """
    # Convert back to original price scale
    pred_price = np.expm1(pred_log)
    actual_price = np.expm1(actual_log)
    
    metrics = {
        "rmsle": compute_rmsle(pred_price, actual_price),
        "mae": compute_mae(pred_price, actual_price),
        "median_ae": compute_median_ae(pred_price, actual_price),
        "r2": compute_r2(pred_price, actual_price),
        "mean_pred_price": float(np.mean(pred_price)),
        "mean_actual_price": float(np.mean(actual_price)),
    }
    
    return metrics


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on entire dataloader and collect all predictions.
    
    Args:
        model: Trained model in eval mode
        dataloader: DataLoader to evaluate on
        device: torch device
    
    Returns:
        (predictions, targets) as numpy arrays in log1p space
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        # Move to device
        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        preds = model(batch_device)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch["target"].numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_targets)


def plot_predictions_vs_actual(
    pred_log: np.ndarray,
    actual_log: np.ndarray,
    save_path: str,
    max_points: int = 10000,
) -> None:
    """
    Scatter plot of predicted vs actual prices.
    
    Args:
        pred_log: Predicted log1p(price) values
        actual_log: Actual log1p(price) values
        save_path: Path to save the plot
        max_points: Max points to plot (for readability)
    """
    # Convert to original scale
    pred_price = np.expm1(pred_log)
    actual_price = np.expm1(actual_log)
    
    # Subsample if too many points
    if len(pred_price) > max_points:
        idx = np.random.choice(len(pred_price), max_points, replace=False)
        pred_price = pred_price[idx]
        actual_price = actual_price[idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Scatter in original scale
    ax1 = axes[0]
    ax1.scatter(actual_price, pred_price, alpha=0.1, s=2, c="#1f77b4")
    max_val = max(actual_price.max(), pred_price.max())
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect")
    ax1.set_xlabel("Actual Price ($)")
    ax1.set_ylabel("Predicted Price ($)")
    ax1.set_title("Predicted vs Actual Price")
    ax1.set_xlim(0, np.percentile(actual_price, 99))
    ax1.set_ylim(0, np.percentile(pred_price, 99))
    ax1.legend()
    
    # Plot 2: Error distribution
    ax2 = axes[1]
    errors = pred_price - actual_price
    ax2.hist(errors, bins=100, alpha=0.7, color="#2ca02c", 
             range=(np.percentile(errors, 1), np.percentile(errors, 99)))
    ax2.axvline(0, color="red", linestyle="--", linewidth=1)
    ax2.set_xlabel("Prediction Error ($)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Error Distribution (Median: ${np.median(np.abs(errors)):.2f})")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved prediction plot to {save_path}")


def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    save_path: str,
) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-o", markersize=4, label="Val Loss", linewidth=2)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("RMSLE Loss", fontsize=12)
    ax.set_title("Training & Validation Loss Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Mark the best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5,
               label=f"Best epoch: {best_epoch}")
    ax.annotate(f"Best: {best_val:.4f}", xy=(best_epoch, best_val),
                xytext=(best_epoch + 0.5, best_val + 0.02),
                fontsize=10, color="green",
                arrowprops=dict(arrowstyle="->", color="green"))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved loss curves to {save_path}")


def print_evaluation_report(metrics: Dict[str, float], split_name: str = "Test") -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 50)
    print(f"  {split_name} Set Evaluation Report")
    print("=" * 50)
    print(f"  RMSLE:             {metrics['rmsle']:.4f}")
    print(f"  MAE:               ${metrics['mae']:.2f}")
    print(f"  Median AE:         ${metrics['median_ae']:.2f}")
    print(f"  R²:                {metrics['r2']:.4f}")
    print(f"  Mean Pred Price:   ${metrics['mean_pred_price']:.2f}")
    print(f"  Mean Actual Price: ${metrics['mean_actual_price']:.2f}")
    print("=" * 50)
