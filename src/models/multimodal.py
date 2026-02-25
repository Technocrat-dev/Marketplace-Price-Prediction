"""
Multimodal Fusion model for Mercari Price Prediction.

Combines outputs from:
- TextEncoder (product name)
- TextEncoder (product description)  
- TabularEncoder (categories, brand, condition, shipping)

Into a single prediction of log1p(price).
"""

import torch
import torch.nn as nn
from typing import Dict, List

from src.models.text_encoder import TextEncoder
from src.models.tabular_encoder import TabularEncoder


class MercariPricePredictor(nn.Module):
    """
    End-to-end multimodal price prediction model.
    
    Architecture:
        [Name Tokens]  → TextEncoder  → name_features
        [Desc Tokens]  → TextEncoder  → desc_features
        [Cat + Cont]   → TabularEncoder → tab_features
        
        Concat(name_features, desc_features, tab_features)
            → FC + BatchNorm + ReLU + Dropout
            → FC + BatchNorm + ReLU + Dropout
            → Linear → predicted log1p(price)
    
    Args:
        name_vocab_size: Size of product name vocabulary
        desc_vocab_size: Size of description vocabulary
        cat_dims: Dict of {column_name: num_categories} for tabular encoder
        text_embed_dim: Word embedding dimension for text encoders
        text_hidden_dim: LSTM hidden dimension for text encoders
        text_num_layers: Number of LSTM layers
        text_dropout: Dropout for text encoders
        text_bidirectional: Whether to use bidirectional LSTM
        cat_embed_dim: Embedding dimension for categorical features
        tabular_hidden_dim: Hidden dimension of tabular encoder output
        fusion_hidden_dims: List of hidden dimensions for fusion FC layers
        fusion_dropout: Dropout for fusion layers
    """
    
    def __init__(
        self,
        name_vocab_size: int,
        desc_vocab_size: int,
        cat_dims: Dict[str, int],
        text_embed_dim: int = 64,
        text_hidden_dim: int = 128,
        text_num_layers: int = 1,
        text_dropout: float = 0.3,
        text_bidirectional: bool = True,
        use_attention: bool = False,
        cat_embed_dim: int = 16,
        tabular_hidden_dim: int = 64,
        fusion_hidden_dims: List[int] = None,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()
        
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [256, 128]
        
        # Text encoders (separate instances for name and description)
        self.name_encoder = TextEncoder(
            vocab_size=name_vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            dropout=text_dropout,
            bidirectional=text_bidirectional,
            use_attention=use_attention,
        )
        
        self.desc_encoder = TextEncoder(
            vocab_size=desc_vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            dropout=text_dropout,
            bidirectional=text_bidirectional,
            use_attention=use_attention,
        )
        
        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            cat_dims=cat_dims,
            cat_embed_dim=cat_embed_dim,
            num_continuous=1,  # shipping
            hidden_dim=tabular_hidden_dim,
        )
        
        # Calculate fusion input size
        fusion_input_dim = (
            self.name_encoder.output_dim 
            + self.desc_encoder.output_dim 
            + self.tabular_encoder.output_dim
        )
        
        # Build fusion head (variable depth)
        fusion_layers = []
        prev_dim = fusion_input_dim
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
            ])
            prev_dim = hidden_dim
        
        # Final output layer — single scalar (predicted log price)
        fusion_layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion_head = nn.Sequential(*fusion_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for module in self.fusion_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            batch: Dict with keys:
                - name_seq: [batch_size, name_seq_len] (LongTensor)
                - desc_seq: [batch_size, desc_seq_len] (LongTensor)
                - categoricals: [batch_size, 5] (LongTensor)
                - shipping: [batch_size] (FloatTensor)
        
        Returns:
            Predicted log1p(price), shape [batch_size]
        """
        # Encode text branches
        name_features = self.name_encoder(batch["name_seq"])      # [B, text_out]
        desc_features = self.desc_encoder(batch["desc_seq"])      # [B, text_out]
        
        # Encode tabular branch
        # Shipping needs to be [B, 1] for concatenation in TabularEncoder
        shipping = batch["shipping"].unsqueeze(1)                 # [B, 1]
        tab_features = self.tabular_encoder(
            batch["categoricals"], shipping
        )                                                          # [B, tab_out]
        
        # Fuse all features
        combined = torch.cat(
            [name_features, desc_features, tab_features], dim=1
        )                                                          # [B, fusion_in]
        
        # Predict
        output = self.fusion_head(combined)[:, 0]                  # [B]
        
        return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Return a human-readable summary of the model architecture."""
        lines = [
            "=" * 60,
            "MercariPricePredictor — Model Summary",
            "=" * 60,
            f"  Name encoder output:  {self.name_encoder.output_dim}",
            f"  Desc encoder output:  {self.desc_encoder.output_dim}",
            f"  Tabular output:       {self.tabular_encoder.output_dim}",
            f"  Fusion input:         {self.name_encoder.output_dim + self.desc_encoder.output_dim + self.tabular_encoder.output_dim}",
            f"  Total parameters:     {self.count_parameters():,}",
            "=" * 60,
        ]
        return "\n".join(lines)
