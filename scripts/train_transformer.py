"""
Transformer Comparison for the Mercari Price Prediction Engine.

Fine-tunes a pretrained DistilBERT encoder on "name [SEP] description" fused
with the same tabular embeddings, on the same train/val/test splits as the
BiLSTM model and baselines, so RMSLE numbers are directly comparable.

Usage:
    # Full run (GPU strongly recommended — ~1.5h/epoch on a T4)
    python scripts/train_transformer.py --epochs 2

    # Quick subsample run (works on CPU, for smoke-testing the pipeline)
    python scripts/train_transformer.py --sample 3000 --epochs 1

Results are saved to outputs/transformer_results.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.transformer_model import TransformerPricePredictor


# =========================================================================
# Data
# =========================================================================

class ListingTextDataset(Dataset):
    """Raw text + encoded tabular features for one data split."""

    def __init__(self, names, descs, tabular, targets):
        self.names = names
        self.descs = descs
        self.tabular = tabular      # [N, 6] — 5 encoded categoricals + shipping
        self.targets = targets      # [N] — log1p(price)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "name": self.names[idx],
            "desc": self.descs[idx],
            "categoricals": self.tabular[idx, :5],
            "shipping": self.tabular[idx, 5],
            "target": self.targets[idx],
        }


def make_collate_fn(tokenizer, max_len: int):
    """Tokenize a batch of (name, description) pairs on the fly."""
    def collate(items):
        encoded = tokenizer(
            [it["name"] for it in items],
            [it["desc"] for it in items],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "categoricals": torch.tensor(
                np.stack([it["categoricals"] for it in items]), dtype=torch.long
            ),
            "shipping": torch.tensor(
                [it["shipping"] for it in items], dtype=torch.float32
            ),
            "target": torch.tensor(
                [it["target"] for it in items], dtype=torch.float32
            ),
        }
    return collate


def load_split(data_dir: Path, split: str, sample: int = 0, seed: int = 42):
    """Load raw text (pkl) and encoded tabular features (npy) for a split."""
    df = pd.read_pickle(data_dir / f"{split}.pkl")
    tabular = np.load(data_dir / f"{split}_tabular.npy")

    assert len(df) == len(tabular), f"{split}: pkl/npy row count mismatch"
    # The npy was derived from the same dataframe — verify alignment via target
    assert np.allclose(
        df["log_price"].to_numpy(), tabular[:, -1], atol=1e-5
    ), f"{split}: pkl/npy rows are not aligned"

    if sample and sample < len(df):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=sample, replace=False)
        df = df.iloc[idx]
        tabular = tabular[idx]

    names = df["name_clean"].fillna("").astype(str).tolist()
    descs = df["desc_clean"].fillna("").astype(str).tolist()
    return ListingTextDataset(
        names, descs, tabular[:, :-1].astype(np.int64), tabular[:, -1].astype(np.float32)
    )


# =========================================================================
# Train / Evaluate
# =========================================================================

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Evaluate RMSLE / MAE / R² on a data loader."""
    model.eval()
    preds, targets = [], []
    for batch in loader:
        target = batch.pop("target")
        batch = {k: v.to(device) for k, v in batch.items()}
        preds.append(model(batch).cpu().numpy())
        targets.append(target.numpy())

    y_pred_log = np.clip(np.concatenate(preds), 0, None)
    y_true_log = np.concatenate(targets)

    rmsle = float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_true_log)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot = np.sum((y_true_log - y_true_log.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmsle": round(rmsle, 4), "mae": round(mae, 2), "r2": round(r2, 4)}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer price model")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--sample", type=int, default=0,
                        help="Subsample N training rows (0 = full dataset)")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Train only the head on top of frozen DistilBERT")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Mercari Price Prediction — Transformer Comparison")
    print("=" * 60)
    print(f"Model: {args.model_name} | Device: {device}")
    if device.type == "cpu" and not args.sample:
        print("[WARN] Full-dataset fine-tuning on CPU is impractical. "
              "Use --sample for a quick run, or train on a GPU machine.")

    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("[ERROR] Processed data not found. Run preprocessing first.")
        return

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate = make_collate_fn(tokenizer, args.max_len)

    # Val/test are subsampled proportionally in sample mode to keep eval fast
    eval_sample = max(args.sample // 4, 1000) if args.sample else 0
    print("\n[INFO] Loading data...")
    train_ds = load_split(data_dir, "train", args.sample, args.seed)
    val_ds = load_split(data_dir, "val", eval_sample, args.seed)
    test_ds = load_split(data_dir, "test", eval_sample, args.seed)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2,
                             collate_fn=collate, num_workers=0)

    model = TransformerPricePredictor(
        cat_dims=metadata["cat_sizes"],
        model_name=args.model_name,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    print(f"  Trainable parameters: {model.count_parameters():,}")

    # Discriminative learning rates: small for the pretrained encoder,
    # larger for the randomly initialized head
    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=0.01)

    # MSE on log1p(price) == RMSLE² — same objective as the BiLSTM model
    criterion = nn.MSELoss()

    checkpoint_path = Path("outputs/checkpoints/transformer_best.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_rmsle = float("inf")
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for step, batch in enumerate(train_loader):
            target = batch.pop("target").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss = criterion(model(batch), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            if step % 100 == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}", flush=True)

        val_metrics = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss / max(n_batches, 1), 4),
            "val_rmsle": val_metrics["rmsle"],
        })
        print(f"[Epoch {epoch}] train_loss={history[-1]['train_loss']:.4f} "
              f"val_rmsle={val_metrics['rmsle']:.4f}")

        if val_metrics["rmsle"] < best_val_rmsle:
            best_val_rmsle = val_metrics["rmsle"]
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_rmsle": best_val_rmsle}, checkpoint_path)
            print(f"  Saved best checkpoint (val_rmsle={best_val_rmsle:.4f})")

    elapsed = time.time() - start

    # Evaluate the best checkpoint on the held-out test split
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print(f"{'Split':<10} {'RMSLE':>8} {'MAE':>10} {'R²':>8}")
    print("-" * 60)
    print(f"{'Val':<10} {val_metrics['rmsle']:>8.4f} ${val_metrics['mae']:>8.2f} "
          f"{val_metrics['r2']:>8.4f}")
    print(f"{'Test':<10} {test_metrics['rmsle']:>8.4f} ${test_metrics['mae']:>8.2f} "
          f"{test_metrics['r2']:>8.4f}")
    print("-" * 60)

    # Show BiLSTM result alongside if available
    dl_results_path = Path("outputs/training_results.json")
    if dl_results_path.exists() and not args.sample:
        with open(dl_results_path) as f:
            dl_metrics = json.load(f).get("test_metrics", {})
        if dl_metrics:
            print(f"{'BiLSTM':<10} {dl_metrics.get('rmsle', 0):>8.4f} "
                  f"${dl_metrics.get('mae', 0):>8.2f} {dl_metrics.get('r2', 0):>8.4f}")

    output = {
        "model_name": args.model_name,
        "trainable_parameters": model.count_parameters(),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "freeze_encoder": args.freeze_encoder,
            "sample": args.sample,
        },
        "is_subsample_run": bool(args.sample),
        "train_samples": len(train_ds),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "training_time_seconds": round(elapsed, 1),
        "device": str(device),
    }
    output_path = Path("outputs/transformer_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SUCCESS] Results saved to {output_path}")
    if args.sample:
        print("[NOTE] This was a subsample run — metrics are not comparable to "
              "the full-dataset results table. Re-run without --sample on a GPU.")


if __name__ == "__main__":
    main()
