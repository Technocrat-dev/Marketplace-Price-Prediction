"""
Unit tests for the model architecture.

Tests:
- TextEncoder forward pass and output shapes
- TabularEncoder forward pass and output shapes
- MercariPricePredictor full forward pass
- Loss functions produce valid gradients
- Model can overfit a single batch (sanity check)
- Integration with real DataLoader
"""

import json
from pathlib import Path

import pytest
import torch

from src.models.text_encoder import TextEncoder
from src.models.tabular_encoder import TabularEncoder
from src.models.multimodal import MercariPricePredictor
from src.models.losses import RMSLELoss, SmoothRMSLELoss


# =========================================================================
# TextEncoder Tests
# =========================================================================

class TestTextEncoder:
    """Tests for the TextEncoder module."""
    
    @pytest.fixture
    def encoder(self):
        return TextEncoder(
            vocab_size=1000,
            embed_dim=32,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
        )
    
    def test_output_shape(self, encoder):
        x = torch.randint(0, 1000, (8, 10))  # batch=8, seq_len=10
        out = encoder(x)
        # BiLSTM → 64 * 2 = 128
        assert out.shape == (8, 128)
    
    def test_output_dim_property(self, encoder):
        assert encoder.output_dim == 128  # 64 * 2 for bidirectional
    
    def test_unidirectional(self):
        encoder = TextEncoder(
            vocab_size=500, embed_dim=16, hidden_dim=32,
            bidirectional=False,
        )
        x = torch.randint(0, 500, (4, 5))
        out = encoder(x)
        assert out.shape == (4, 32)
        assert encoder.output_dim == 32
    
    def test_handles_all_padding(self, encoder):
        """All-PAD input (idx=0) should not crash."""
        x = torch.zeros(4, 10, dtype=torch.long)
        out = encoder(x)
        assert out.shape == (4, 128)
        assert not torch.isnan(out).any()
    
    def test_handles_single_token(self, encoder):
        """Input with just one non-PAD token."""
        x = torch.zeros(4, 10, dtype=torch.long)
        x[:, 0] = 42
        out = encoder(x)
        assert out.shape == (4, 128)
    
    def test_gradient_flows(self, encoder):
        x = torch.randint(0, 1000, (4, 10))
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        # Check that embedding gradients exist
        assert encoder.embedding.weight.grad is not None


# =========================================================================
# TabularEncoder Tests
# =========================================================================

class TestTabularEncoder:
    """Tests for the TabularEncoder module."""
    
    @pytest.fixture
    def encoder(self):
        cat_dims = {
            "main_cat": 12,
            "sub_cat1": 115,
            "sub_cat2": 871,
            "brand_name": 4809,
            "item_condition_id": 6,
        }
        return TabularEncoder(
            cat_dims=cat_dims,
            cat_embed_dim=16,
            num_continuous=1,
            hidden_dim=64,
        )
    
    def test_output_shape(self, encoder):
        cats = torch.randint(0, 6, (8, 5))
        cont = torch.randn(8, 1)
        out = encoder(cats, cont)
        assert out.shape == (8, 64)
    
    def test_output_dim_property(self, encoder):
        assert encoder.output_dim == 64
    
    def test_gradient_flows(self, encoder):
        cats = torch.randint(0, 6, (4, 5))
        cont = torch.randn(4, 1)
        out = encoder(cats, cont)
        loss = out.sum()
        loss.backward()
        # Check first embedding has gradients
        assert encoder.embeddings[0].weight.grad is not None


# =========================================================================
# MercariPricePredictor Tests
# =========================================================================

class TestMercariPricePredictor:
    """Tests for the full multimodal model."""
    
    @pytest.fixture
    def model(self):
        cat_dims = {
            "main_cat": 12,
            "sub_cat1": 115,
            "sub_cat2": 871,
            "brand_name": 4809,
            "item_condition_id": 6,
        }
        return MercariPricePredictor(
            name_vocab_size=1000,
            desc_vocab_size=2000,
            cat_dims=cat_dims,
            text_embed_dim=32,
            text_hidden_dim=64,
            text_num_layers=1,
            text_dropout=0.0,
            cat_embed_dim=8,
            tabular_hidden_dim=32,
            fusion_hidden_dims=[128, 64],
            fusion_dropout=0.0,
        )
    
    @pytest.fixture
    def sample_batch(self):
        return {
            "name_seq": torch.randint(0, 1000, (8, 10)),
            "desc_seq": torch.randint(0, 2000, (8, 75)),
            "categoricals": torch.randint(0, 6, (8, 5)),
            "shipping": torch.randint(0, 2, (8,)).float(),
            "target": torch.randn(8).abs() + 1,  # Positive targets
        }
    
    def test_forward_shape(self, model, sample_batch):
        out = model(sample_batch)
        assert out.shape == (8,)
    
    def test_forward_no_nan(self, model, sample_batch):
        out = model(sample_batch)
        assert not torch.isnan(out).any()
    
    def test_count_parameters(self, model):
        count = model.count_parameters()
        assert count > 0
        # Should be in the thousands to millions range for this config
        assert count > 10_000
    
    def test_summary(self, model):
        summary = model.summary()
        assert "MercariPricePredictor" in summary
        assert "parameters" in summary.lower()
    
    def test_gradient_flows_end_to_end(self, model, sample_batch):
        """Full backward pass should produce gradients everywhere."""
        out = model(sample_batch)
        loss = out.sum()
        loss.backward()
        
        # Check gradients in each sub-module
        assert model.name_encoder.embedding.weight.grad is not None
        assert model.desc_encoder.embedding.weight.grad is not None
        assert model.tabular_encoder.embeddings[0].weight.grad is not None
    
    def test_overfit_single_batch(self, model, sample_batch):
        """
        Sanity check: model should be able to overfit one batch.
        If loss doesn't decrease, something is fundamentally broken.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = RMSLELoss(log_space=True)
        
        model.train()
        initial_loss = None
        
        for step in range(50):
            optimizer.zero_grad()
            pred = model(sample_batch)
            loss = loss_fn(pred, sample_batch["target"])
            loss.backward()
            optimizer.step()
            
            if step == 0:
                initial_loss = loss.item()
        
        final_loss = loss.item()
        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, (
            f"Model failed to overfit: initial={initial_loss:.4f}, "
            f"final={final_loss:.4f}"
        )


# =========================================================================
# Loss Function Tests
# =========================================================================

class TestRMSLELoss:
    """Tests for the RMSLELoss function."""
    
    def test_zero_loss_for_perfect_prediction(self):
        loss_fn = RMSLELoss(log_space=True)
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = loss_fn(pred, target)
        assert loss.item() < 1e-3
    
    def test_positive_loss_for_wrong_prediction(self):
        loss_fn = RMSLELoss(log_space=True)
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([4.0, 5.0, 6.0])
        loss = loss_fn(pred, target)
        assert loss.item() > 0
    
    def test_gradient_exists(self):
        loss_fn = RMSLELoss(log_space=True)
        pred = torch.tensor([2.0, 3.0], requires_grad=True)
        target = torch.tensor([3.0, 4.0])
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
    
    def test_raw_space_mode(self):
        """Test with raw prices (not log-transformed)."""
        loss_fn = RMSLELoss(log_space=False)
        pred = torch.tensor([10.0, 20.0, 30.0])
        target = torch.tensor([10.0, 20.0, 30.0])
        loss = loss_fn(pred, target)
        assert loss.item() < 1e-3
    
    def test_symmetric(self):
        """RMSLE should be symmetric: error(pred, target) == error(target, pred)."""
        loss_fn = RMSLELoss(log_space=True)
        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        assert abs(loss_fn(a, b).item() - loss_fn(b, a).item()) < 1e-6


class TestSmoothRMSLELoss:
    """Tests for the SmoothRMSLELoss function."""
    
    def test_zero_loss_for_perfect_prediction(self):
        loss_fn = SmoothRMSLELoss(log_space=True)
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = loss_fn(pred, target)
        assert loss.item() < 1e-3
    
    def test_less_sensitive_to_outliers(self):
        """
        SmoothRMSLE should produce smaller loss than RMSLE
        when there's a large outlier, since it uses L1 beyond beta.
        """
        rmsle = RMSLELoss(log_space=True)
        smooth = SmoothRMSLELoss(beta=0.5, log_space=True)
        
        # Create data with one huge outlier
        pred = torch.tensor([2.0, 3.0, 20.0])   # 20 is way off
        target = torch.tensor([2.0, 3.0, 3.0])
        
        rmsle_loss = rmsle(pred, target).item()
        smooth_loss = smooth(pred, target).item()
        
        # Smooth should give lower loss due to linear behavior on outlier
        assert smooth_loss < rmsle_loss


# =========================================================================
# Integration Test — Model with real DataLoader
# =========================================================================

class TestModelIntegration:
    """Integration tests with actual processed data."""
    
    @pytest.fixture
    def metadata(self):
        path = Path("data/processed/metadata.json")
        if not path.exists():
            pytest.skip("Processed data not found.")
        with open(path) as f:
            return json.load(f)
    
    def test_model_with_real_metadata(self, metadata):
        """Build model using actual vocabulary and category sizes."""
        model = MercariPricePredictor(
            name_vocab_size=metadata["name_vocab_size"],
            desc_vocab_size=metadata["desc_vocab_size"],
            cat_dims=metadata["cat_sizes"],
        )
        
        batch = {
            "name_seq": torch.randint(0, 100, (4, 10)),
            "desc_seq": torch.randint(0, 100, (4, 75)),
            "categoricals": torch.randint(0, 6, (4, 5)),
            "shipping": torch.randint(0, 2, (4,)).float(),
        }
        
        out = model(batch)
        assert out.shape == (4,)
        print(f"\nModel parameter count: {model.count_parameters():,}")
        print(model.summary())
