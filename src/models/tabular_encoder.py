"""
Tabular Encoder module for the Mercari Price Prediction model.

Encodes categorical features (main_cat, sub_cat1, sub_cat2, brand, condition)
via learned embeddings, and continuous features (shipping) via linear projection.
"""

import torch
import torch.nn as nn
from typing import Dict


class TabularEncoder(nn.Module):
    """
    Encodes tabular features into a fixed-size feature vector.
    
    Pipeline:
        Categorical columns → Individual Embeddings → Concat
        Continuous columns  → Linear projection     → Concat
        All features → Linear → ReLU → output
    
    Args:
        cat_dims: Dict mapping column name to number of categories.
                  Must be in the same order as the categoricals tensor columns.
                  Example: {"main_cat": 12, "sub_cat1": 115, ...}
        cat_embed_dim: Embedding dimension for each categorical feature
        num_continuous: Number of continuous features (e.g., 1 for shipping)
        hidden_dim: Output dimension of the tabular encoder
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        cat_dims: Dict[str, int],
        cat_embed_dim: int = 16,
        num_continuous: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.cat_names = list(cat_dims.keys())
        self.num_continuous = num_continuous
        
        # Create an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_cats, embedding_dim=cat_embed_dim)
            for num_cats in cat_dims.values()
        ])
        
        # Total input size: all cat embeddings + continuous features
        total_embed_dim = len(cat_dims) * cat_embed_dim + num_continuous
        
        # Projection layer
        self.fc = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self._output_dim = hidden_dim
    
    @property
    def output_dim(self) -> int:
        """Size of the output feature vector."""
        return self._output_dim
    
    def forward(
        self, 
        categoricals: torch.Tensor, 
        continuous: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            categoricals: Category indices, shape [batch_size, num_cat_features]
                          Column order must match cat_dims order.
            continuous: Continuous features, shape [batch_size, num_continuous]
                        (e.g., shipping flag)
        
        Returns:
            Feature vector, shape [batch_size, hidden_dim]
        """
        # Embed each categorical feature separately
        cat_embeds = []
        for i, embed_layer in enumerate(self.embeddings):
            cat_embeds.append(embed_layer(categoricals[:, i]))
        
        # Concatenate all categorical embeddings + continuous features
        # Each cat_embed is [batch, cat_embed_dim]
        # continuous is [batch, num_continuous]
        combined = torch.cat(cat_embeds + [continuous], dim=1)
        
        # Project to hidden_dim
        output = self.fc(combined)
        
        return output  # [batch, hidden_dim]
