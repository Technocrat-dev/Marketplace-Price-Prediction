"""
Trainer for the Mercari Price Prediction model.

Handles:
- Training loop with batch progress
- Validation after each epoch
- Learning rate scheduling (ReduceLROnPlateau / CosineAnnealing)
- Gradient clipping
- Early stopping
- Model checkpointing (best + last)
- Loss curve tracking
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.losses import RMSLELoss
from src.training.evaluate import (
    collect_predictions,
    evaluate_predictions,
    plot_loss_curves,
    plot_predictions_vs_actual,
    print_evaluation_report,
)


class Trainer:
    """
    Trainer for the MercariPricePredictor model.
    
    Handles the full training lifecycle: train, validate, checkpoint,
    early stop, and evaluate.
    
    Args:
        model: The MercariPricePredictor model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch device (cuda/cpu)
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        grad_clip: Max gradient norm for clipping
        patience: Early stopping patience (epochs without improvement)
        scheduler_type: "plateau" or "cosine"
        scheduler_factor: LR reduction factor (for plateau)
        scheduler_patience: Epochs to wait before reducing LR (for plateau)
        checkpoint_dir: Directory to save model checkpoints
        output_dir: Directory to save plots and results
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        patience: int = 5,
        scheduler_type: str = "plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
        checkpoint_dir: str = "outputs/checkpoints",
        output_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_clip = grad_clip
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.loss_fn = RMSLELoss(log_space=True)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        if scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience,
            )
        elif scheduler_type == "cosine":
            # We'll set T_max when we know num_epochs in fit()
            self.scheduler = None  
            self._scheduler_type = "cosine"
        else:
            self.scheduler = None
            self._scheduler_type = None
        
        self._scheduler_type_str = scheduler_type
        
        # Tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.current_epoch = 0
    
    def _train_one_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"  Train Epoch {self.current_epoch}",
            leave=False,
            ncols=100,
        )
        
        for batch in pbar:
            # Move batch to device
            batch_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward
            self.optimizer.zero_grad()
            predictions = self.model(batch_device)
            loss = self.loss_fn(predictions, batch_device["target"])
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            batch_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            predictions = self.model(batch_device)
            loss = self.loss_fn(predictions, batch_device["target"])
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        print(f"[INFO] Loaded checkpoint from epoch {self.current_epoch}")
    
    def fit(self, num_epochs: int) -> Dict[str, list]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
        
        Returns:
            Dict with train_losses and val_losses
        """
        # Set up cosine scheduler if needed
        if self._scheduler_type_str == "cosine" and self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        
        print("=" * 60)
        print(f"Training for {num_epochs} epochs on {self.device}")
        print(f"  Batches/epoch: {len(self.train_loader)}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss = self._train_one_epoch()
            
            # Validate
            val_loss = self._validate()
            
            # Track losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if self._scheduler_type_str == "plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch}.pt", is_best=is_best)
            
            # Logging
            epoch_time = time.time() - epoch_start
            best_marker = " ★ BEST" if is_best else ""
            lr_change = f" (LR: {current_lr:.6f}→{new_lr:.6f})" if new_lr != current_lr else ""
            
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f}{best_marker} | "
                f"LR: {new_lr:.6f}{lr_change} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break
        
        total_time = time.time() - start_time
        print(f"\n[INFO] Training complete in {total_time/60:.1f} minutes")
        print(f"[INFO] Best validation loss: {self.best_val_loss:.4f}")
        
        # Save loss curves plot
        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            str(self.output_dir / "loss_curves.png"),
        )
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
    
    def evaluate(
        self, 
        test_loader: DataLoader, 
        split_name: str = "Test",
    ) -> Dict[str, float]:
        """
        Evaluate the model on a test set and generate reports.
        
        Loads the best checkpoint before evaluating.
        
        Args:
            test_loader: Test DataLoader
            split_name: Name for the report header
        
        Returns:
            Dict of evaluation metrics
        """
        # Load best model
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[INFO] Loaded best model from epoch {checkpoint['epoch']}")
        
        # Collect predictions
        pred_log, actual_log = collect_predictions(
            self.model, test_loader, self.device
        )
        
        # Compute metrics
        metrics = evaluate_predictions(pred_log, actual_log)
        
        # Print report
        print_evaluation_report(metrics, split_name)
        
        # Save prediction plot
        plot_predictions_vs_actual(
            pred_log, actual_log,
            str(self.output_dir / f"{split_name.lower()}_predictions.png"),
        )
        
        return metrics
