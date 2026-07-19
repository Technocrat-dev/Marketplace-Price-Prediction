"""
Transformer-based price prediction model.

Replaces the BiLSTM text branches with a pretrained DistilBERT encoder over
"name [SEP] description", fused with the same TabularEncoder used by the
BiLSTM model so the two architectures are directly comparable on identical
splits and targets.
"""

import torch
import torch.nn as nn
from typing import Dict, List

from src.models.tabular_encoder import TabularEncoder


class TransformerPricePredictor(nn.Module):
    """
    Transformer + tabular fusion model for price regression.

    Architecture:
        [Name + Description tokens] → DistilBERT → CLS pooled → projection
        [Cat + Cont]                → TabularEncoder → tab_features

        Concat(text_features, tab_features)
            → FC + BatchNorm + ReLU + Dropout (per fusion layer)
            → Linear → predicted log1p(price)

    Args:
        cat_dims: Dict of {column_name: num_categories} for tabular encoder
        model_name: HuggingFace model identifier for the text encoder
        text_proj_dim: Dimension the transformer CLS output is projected to
        freeze_encoder: If True, the pretrained encoder is not fine-tuned
        cat_embed_dim: Embedding dimension for categorical features
        tabular_hidden_dim: Hidden dimension of tabular encoder output
        fusion_hidden_dims: List of hidden dimensions for fusion FC layers
        fusion_dropout: Dropout for fusion layers
    """

    def __init__(
        self,
        cat_dims: Dict[str, int],
        model_name: str = "distilbert-base-uncased",
        text_proj_dim: int = 256,
        freeze_encoder: bool = False,
        cat_embed_dim: int = 16,
        tabular_hidden_dim: int = 64,
        fusion_hidden_dims: List[int] = None,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()

        from transformers import AutoModel

        if fusion_hidden_dims is None:
            fusion_hidden_dims = [256, 128]

        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(encoder_dim, text_proj_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
        )

        self.tabular_encoder = TabularEncoder(
            cat_dims=cat_dims,
            cat_embed_dim=cat_embed_dim,
            num_continuous=1,  # shipping
            hidden_dim=tabular_hidden_dim,
        )

        fusion_input_dim = text_proj_dim + self.tabular_encoder.output_dim

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
        fusion_layers.append(nn.Linear(prev_dim, 1))
        self.fusion_head = nn.Sequential(*fusion_layers)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for fusion linear layers."""
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
                - input_ids: [batch_size, seq_len] (LongTensor)
                - attention_mask: [batch_size, seq_len] (LongTensor)
                - categoricals: [batch_size, 5] (LongTensor)
                - shipping: [batch_size] (FloatTensor)

        Returns:
            Predicted log1p(price), shape [batch_size]
        """
        encoder_out = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        cls_state = encoder_out.last_hidden_state[:, 0]           # [B, encoder_dim]
        text_features = self.text_proj(cls_state)                 # [B, text_proj_dim]

        shipping = batch["shipping"].unsqueeze(1)                 # [B, 1]
        tab_features = self.tabular_encoder(
            batch["categoricals"], shipping
        )                                                          # [B, tab_out]

        combined = torch.cat([text_features, tab_features], dim=1)
        return self.fusion_head(combined)[:, 0]                   # [B]

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
