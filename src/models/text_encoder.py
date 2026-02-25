"""
Text Encoder module for the Mercari Price Prediction model.

Architecture:
    Without attention: Embedding → BiLSTM → Final hidden state
    With attention:    Embedding → BiLSTM → Self-Attention → Weighted sum

Used for both product names and descriptions (separate instances).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionLayer(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over LSTM hidden states.
    
    Learns which tokens are most important for price prediction, producing
    a weighted sum of all hidden states instead of just using the final one.
    
    Attention(H) = softmax(W₂ · tanh(W₁ · H + b₁) + b₂) · H
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, attention_dim)
        self.W2 = nn.Linear(attention_dim, 1)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim] — all LSTM outputs
            mask: [batch, seq_len] — True for real tokens, False for padding
        
        Returns:
            context: [batch, hidden_dim] — attention-weighted sum
            weights: [batch, seq_len] — attention weights (for interpretability)
        """
        # Compute attention scores: [batch, seq_len, 1]
        energy = self.W2(torch.tanh(self.W1(hidden_states)))
        energy = energy.squeeze(-1)  # [batch, seq_len]
        
        # Mask padding positions with -inf so softmax gives them 0 weight
        energy = energy.masked_fill(~mask, float('-inf'))
        
        # Softmax over sequence dimension
        weights = F.softmax(energy, dim=1)  # [batch, seq_len]
        
        # Handle edge case: all-padding sequences produce NaN from softmax(-inf)
        weights = weights.nan_to_num(0.0)
        
        # Weighted sum of hidden states
        context = torch.bmm(weights.unsqueeze(1), hidden_states)  # [batch, 1, hidden_dim]
        context = context.squeeze(1)  # [batch, hidden_dim]
        
        return context, weights


class TextEncoder(nn.Module):
    """
    Encodes tokenized text sequences into fixed-size feature vectors.
    
    Pipeline (no attention):
        Token IDs → Embedding → BiLSTM → Concat(fwd_hidden, bwd_hidden)
    
    Pipeline (with attention):
        Token IDs → Embedding → BiLSTM → Attention(all_hidden) → context
    
    Args:
        vocab_size: Number of words in vocabulary (incl. PAD and UNK)
        embed_dim: Dimension of word embeddings
        hidden_dim: LSTM hidden state dimension (per direction)
        num_layers: Number of stacked LSTM layers
        dropout: Dropout rate applied to embeddings and between LSTM layers
        bidirectional: Whether to use bidirectional LSTM
        pad_idx: Index of the PAD token (for zero-masking in embedding)
        use_attention: If True, use self-attention over all hidden states
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
        use_attention: bool = False,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = use_attention
        
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
        
        # Optional attention layer
        if use_attention:
            self.attention = AttentionLayer(
                hidden_dim=hidden_dim * self.num_directions,
                attention_dim=hidden_dim,
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
        mask = (x != 0)  # [batch, seq_len]
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        
        # Embed tokens: [batch, seq_len] → [batch, seq_len, embed_dim]
        embedded = self.dropout(self.embedding(x))
        
        # Pack for efficient LSTM processing (ignores padding)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        if self.use_attention:
            # Unpack all hidden states for attention
            packed_output, (h_n, _) = self.lstm(packed)
            hidden_states, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=x.size(1)
            )
            # hidden_states: [batch, seq_len, hidden_dim * num_directions]
            
            # Apply attention
            hidden, _ = self.attention(hidden_states, mask)
        else:
            # Original behavior: use only final hidden states
            _, (h_n, _) = self.lstm(packed)
            # h_n shape: [num_layers * num_directions, batch, hidden_dim]
            
            if self.bidirectional:
                forward_h = h_n[-2]   # [batch, hidden_dim]
                backward_h = h_n[-1]  # [batch, hidden_dim]
                hidden = torch.cat([forward_h, backward_h], dim=1)
            else:
                hidden = h_n[-1]  # [batch, hidden_dim]
        
        return hidden  # [batch, hidden_dim * num_directions]

