"""
Text Encoder module for the Mercari Price Prediction model.

Architecture: Embedding → Bidirectional LSTM → Final hidden state
Used for both product names and descriptions (separate instances).
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextEncoder(nn.Module):
    """
    Encodes tokenized text sequences into fixed-size feature vectors.
    
    Pipeline:
        Token IDs → Embedding → BiLSTM → Concat(fwd_hidden, bwd_hidden)
    
    Args:
        vocab_size: Number of words in vocabulary (incl. PAD and UNK)
        embed_dim: Dimension of word embeddings
        hidden_dim: LSTM hidden state dimension (per direction)
        num_layers: Number of stacked LSTM layers
        dropout: Dropout rate applied to embeddings and between LSTM layers
        bidirectional: Whether to use bidirectional LSTM
        pad_idx: Index of the PAD token (for zero-masking in embedding)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer — PAD tokens get zero vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
    
    @property
    def output_dim(self) -> int:
        """Size of the output feature vector."""
        return self.hidden_dim * self.num_directions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Token indices, shape [batch_size, seq_len] (LongTensor)
        
        Returns:
            Feature vector, shape [batch_size, hidden_dim * num_directions]
        """
        # Compute actual sequence lengths (non-PAD positions)
        # PAD index is 0, so count non-zero elements
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        
        # Embed tokens: [batch, seq_len] → [batch, seq_len, embed_dim]
        embedded = self.dropout(self.embedding(x))
        
        # Pack for efficient LSTM processing (ignores padding)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward: outputs all hidden states, (h_n, c_n) are final states
        _, (h_n, _) = self.lstm(packed)
        # h_n shape: [num_layers * num_directions, batch, hidden_dim]
        
        if self.bidirectional:
            # Concatenate final forward and backward hidden states
            # Take the last layer: h_n[-2] is forward, h_n[-1] is backward
            forward_h = h_n[-2]   # [batch, hidden_dim]
            backward_h = h_n[-1]  # [batch, hidden_dim]
            hidden = torch.cat([forward_h, backward_h], dim=1)
        else:
            # Take the last layer's hidden state
            hidden = h_n[-1]  # [batch, hidden_dim]
        
        return hidden  # [batch, hidden_dim * num_directions]
