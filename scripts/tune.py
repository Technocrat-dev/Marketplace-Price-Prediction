"""
Optuna Hyperparameter Tuning for MercariPricePredictor.

Searches over text encoder dims, tabular encoder dims, fusion architecture,
learning rate, dropout, and attention. Uses pruning (MedianPruner) to
terminate unpromising trials early.

Usage:
    python scripts/tune.py --n-trials 30
    python scripts/tune.py --n-trials 50 --timeout 3600
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import create_dataloaders
from src.models.multimodal import MercariPricePredictor
from src.training.trainer import Trainer


def build_model_from_trial(trial: optuna.Trial, metadata: dict) -> MercariPricePredictor:
    """Build a model with hyperparameters from an Optuna trial."""
    
    text_embed_dim = trial.suggest_categorical("text_embed_dim", [32, 64, 128])
    text_hidden_dim = trial.suggest_categorical("text_hidden_dim", [64, 128, 256])
    text_dropout = trial.suggest_float("text_dropout", 0.1, 0.5, step=0.1)
    use_attention = trial.suggest_categorical("use_attention", [True, False])
    
    cat_embed_dim = trial.suggest_categorical("cat_embed_dim", [8, 16, 32])
    tabular_hidden_dim = trial.suggest_categorical("tabular_hidden_dim", [32, 64, 128])
    
    fusion_depth = trial.suggest_int("fusion_depth", 1, 3)
    fusion_dims = []
    dim = trial.suggest_categorical("fusion_first_dim", [128, 256, 512])
    for i in range(fusion_depth):
        fusion_dims.append(dim)
        dim = max(32, dim // 2)
    fusion_dropout = trial.suggest_float("fusion_dropout", 0.2, 0.5, step=0.1)
    
    model = MercariPricePredictor(
        name_vocab_size=metadata["name_vocab_size"],
        desc_vocab_size=metadata["desc_vocab_size"],
        cat_dims=metadata["cat_sizes"],
        text_embed_dim=text_embed_dim,
        text_hidden_dim=text_hidden_dim,
        text_num_layers=1,
        text_dropout=text_dropout,
        text_bidirectional=True,
        use_attention=use_attention,
        cat_embed_dim=cat_embed_dim,
        tabular_hidden_dim=tabular_hidden_dim,
        fusion_hidden_dims=fusion_dims,
        fusion_dropout=fusion_dropout,
    )
    return model


def objective(trial: optuna.Trial, cfg: dict, metadata: dict, 
              train_loader, val_loader, device: torch.device) -> float:
    """Single Optuna trial: build model, train, return val RMSLE."""
    
    model = build_model_from_trial(trial, metadata)
    
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=lr,
        weight_decay=weight_decay,
        grad_clip=cfg["training"]["grad_clip"],
        patience=5,  # More aggressive early stopping for tuning
        scheduler_type="plateau",
        scheduler_factor=0.5,
        scheduler_patience=2,
        checkpoint_dir=f"checkpoints/trial_{trial.number}",
        output_dir=f"outputs/tuning/trial_{trial.number}",
    )
    
    # Train with pruning callback
    num_epochs = cfg["training"].get("num_epochs", 15)
    for epoch in range(1, num_epochs + 1):
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate_epoch()
        
        # Report to Optuna for pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Check early stopping
        if trainer.early_stop:
            break
    
    return trainer.best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--study-name", type=str, default="mercari-price-prediction")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load data
    train_loader, val_loader, _, metadata = create_dataloaders(
        data_dir=cfg["paths"]["processed_data"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    
    study.optimize(
        lambda trial: objective(trial, cfg, metadata, train_loader, val_loader, device),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    
    # Report results
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 60)
    
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best val RMSLE: {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    output_path = Path("outputs/tuning_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ],
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SUCCESS] Results saved to {output_path}")


if __name__ == "__main__":
    main()
