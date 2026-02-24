"""
Custom loss functions for the Mercari Price Prediction model.

Includes:
- RMSLELoss: Root Mean Squared Logarithmic Error
- SmoothRMSLELoss: RMSLE with Smooth L1 for stability
"""

import torch
import torch.nn as nn


class RMSLELoss(nn.Module):
    """
    Root Mean Squared Logarithmic Error loss.
    
    Formula: sqrt(mean((log(pred+1) - log(actual+1))^2))
    
    Since our model predicts log1p(price) and targets are log1p(price),
    this simplifies to RMSE on the log-transformed values:
        sqrt(mean((pred_log - target_log)^2))
    
    Which is exactly MSELoss → sqrt.
    
    We keep this as a named class for clarity and to allow switching
    between raw and log-space computation.
    """
    
    def __init__(self, log_space: bool = True):
        """
        Args:
            log_space: If True, assumes inputs are already in log1p space.
                       If False, applies log1p transform internally.
        """
        super().__init__()
        self.log_space = log_space
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSLE loss.
        
        Args:
            pred: Predicted values, shape [batch_size]
            target: Target values, shape [batch_size]
        
        Returns:
            Scalar loss value
        """
        if self.log_space:
            # Already in log1p space — just compute RMSE
            return torch.sqrt(self.mse(pred, target) + 1e-8)
        else:
            # Apply log1p transform, clamping pred to avoid log of negative
            pred_log = torch.log1p(pred.clamp(min=0))
            target_log = torch.log1p(target.clamp(min=0))
            return torch.sqrt(self.mse(pred_log, target_log) + 1e-8)


class SmoothRMSLELoss(nn.Module):
    """
    Smooth variant of RMSLE using Huber (Smooth L1) loss instead of MSE.
    
    More robust to outliers than standard RMSLE. Useful for price prediction
    where a few extremely expensive items can dominate MSE-based losses.
    
    When error < beta: behaves like MSE (quadratic)
    When error > beta: behaves like MAE (linear) — reduces outlier impact
    """
    
    def __init__(self, beta: float = 1.0, log_space: bool = True):
        """
        Args:
            beta: Threshold for switching between quadratic and linear behavior
            log_space: If True, assumes inputs are already in log1p space
        """
        super().__init__()
        self.log_space = log_space
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Smooth RMSLE loss.
        
        Args:
            pred: Predicted values, shape [batch_size]
            target: Target values, shape [batch_size]
        
        Returns:
            Scalar loss value
        """
        if self.log_space:
            return torch.sqrt(self.smooth_l1(pred, target) + 1e-8)
        else:
            pred_log = torch.log1p(pred.clamp(min=0))
            target_log = torch.log1p(target.clamp(min=0))
            return torch.sqrt(self.smooth_l1(pred_log, target_log) + 1e-8)
