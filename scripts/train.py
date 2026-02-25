"""
Main training script for the Mercari Price Prediction model.

Usage:
    # Full training
    python scripts/train.py

    # Quick test (2 epochs, small subset)
    python scripts/train.py --quick

    # Custom epochs
    python scripts/train.py --epochs 20
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import create_dataloaders
from src.models.multimodal import MercariPricePredictor
from src.training.trainer import Trainer


def build_model(metadata: dict, cfg: dict) -> MercariPricePredictor:
    """Build the model from metadata and config."""
    model = MercariPricePredictor(
        name_vocab_size=metadata["name_vocab_size"],
        desc_vocab_size=metadata["desc_vocab_size"],
        cat_dims=metadata["cat_sizes"],
        text_embed_dim=cfg["model"]["text_embed_dim"],
        text_hidden_dim=cfg["model"]["text_hidden_dim"],
        text_num_layers=cfg["model"]["text_num_layers"],
        text_dropout=cfg["model"]["text_dropout"],
        text_bidirectional=cfg["model"]["text_bidirectional"],
        use_attention=cfg["model"].get("use_attention", False),
        cat_embed_dim=cfg["model"]["cat_embed_dim"],
        tabular_hidden_dim=cfg["model"]["tabular_hidden_dim"],
        fusion_hidden_dims=cfg["model"]["fusion_hidden_dims"],
        fusion_dropout=cfg["model"]["fusion_dropout"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the Mercari Price Predictor")
    parser.add_argument("--quick", action="store_true", 
                        help="Quick test run (2 epochs)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Quick mode overrides
    if args.quick:
        cfg["training"]["num_epochs"] = 2
        cfg["training"]["batch_size"] = 1024
        print("[INFO] Quick mode: 2 epochs, batch_size=1024")
    
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir=cfg["paths"]["processed_data"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )
    
    # Build model
    model = build_model(metadata, cfg)
    print(model.summary())
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_clip=cfg["training"]["grad_clip"],
        patience=cfg["training"]["patience"],
        scheduler_type=cfg["training"]["scheduler"],
        scheduler_factor=cfg["training"]["scheduler_factor"],
        scheduler_patience=cfg["training"]["scheduler_patience"],
        checkpoint_dir=cfg["paths"]["checkpoints"],
        output_dir=cfg["paths"]["outputs"],
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    history = trainer.fit(num_epochs=cfg["training"]["num_epochs"])
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final evaluation on test set")
    print("=" * 60)
    test_metrics = trainer.evaluate(test_loader, split_name="Test")
    
    # Also evaluate on validation set for comparison
    val_metrics = trainer.evaluate(val_loader, split_name="Validation")
    
    # Save final results
    results = {
        "config": cfg,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "best_epoch": int(torch.tensor(history["val_losses"]).argmin().item()) + 1,
        "total_epochs": len(history["train_losses"]),
        "model_parameters": model.count_parameters(),
    }
    
    results_path = Path(cfg["paths"]["outputs"]) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[INFO] Results saved to {results_path}")


if __name__ == "__main__":
    main()
