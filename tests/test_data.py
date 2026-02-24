"""
Unit tests for the data preprocessing pipeline and dataset.

Tests:
- Text cleaning edge cases
- Category parsing
- Vocabulary building and encoding
- CategoricalEncoder behavior
- Dataset tensor shapes and types
- DataLoader batch integrity
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocess import (
    Vocabulary,
    CategoricalEncoder,
    clean_text,
    parse_category,
    preprocess_dataframe,
)
from src.data.dataset import MercariDataset


# =========================================================================
# Text Cleaning Tests
# =========================================================================

class TestCleanText:
    """Tests for the clean_text function."""
    
    def test_basic_cleaning(self):
        assert clean_text("Hello World!") == "hello world!"
    
    def test_rm_token_removal(self):
        """Kaggle redacts prices as [rm] — these should be removed."""
        assert clean_text("Price was [rm] dollars") == "price was dollars"
    
    def test_html_removal(self):
        assert clean_text("Nice <b>bold</b> text") == "nice bold text"
    
    def test_url_removal(self):
        result = clean_text("Check http://example.com for details")
        assert "http" not in result
        assert "example" not in result
    
    def test_special_chars(self):
        result = clean_text("Hello @world #2024 $$$")
        # Should only keep alphanumeric, spaces, and basic punctuation
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
    
    def test_whitespace_normalization(self):
        assert clean_text("too   many    spaces") == "too many spaces"
    
    def test_empty_string(self):
        assert clean_text("") == ""
    
    def test_none_input(self):
        assert clean_text(None) == ""
    
    def test_nan_input(self):
        assert clean_text(float("nan")) == ""
    
    def test_numeric_input(self):
        assert clean_text(12345) == ""
    
    def test_only_special_chars(self):
        result = clean_text("@#$%^&*()")
        assert result == ""
    
    def test_preserves_apostrophe(self):
        assert "don't" in clean_text("Don't stop")
    
    def test_preserves_hyphen(self):
        assert "t-shirt" in clean_text("T-SHIRT")


# =========================================================================
# Category Parsing Tests
# =========================================================================

class TestParseCategory:
    """Tests for the parse_category function."""
    
    def test_full_hierarchy(self):
        result = parse_category("Women/Tops & Blouses/Blouse")
        assert result == ("women", "tops & blouses", "blouse")
    
    def test_two_levels(self):
        result = parse_category("Electronics/Computers")
        assert result == ("electronics", "computers", "missing")
    
    def test_one_level(self):
        result = parse_category("Books")
        assert result == ("books", "missing", "missing")
    
    def test_empty_string(self):
        result = parse_category("")
        assert result == ("missing", "missing", "missing")
    
    def test_none_input(self):
        result = parse_category(None)
        assert result == ("missing", "missing", "missing")
    
    def test_extra_levels_ignored(self):
        # Only first 3 levels should be used
        result = parse_category("A/B/C/D/E")
        assert result == ("a", "b", "c")
    
    def test_whitespace_handling(self):
        result = parse_category(" Electronics / Phones / Cases ")
        assert result == ("electronics", "phones", "cases")


# =========================================================================
# Vocabulary Tests
# =========================================================================

class TestVocabulary:
    """Tests for the Vocabulary class."""
    
    def test_special_tokens(self):
        vocab = Vocabulary(min_freq=1)
        assert vocab.PAD_IDX == 0
        assert vocab.UNK_IDX == 1
        assert len(vocab) == 2  # PAD + UNK
    
    def test_build_basic(self):
        vocab = Vocabulary(min_freq=1)
        vocab.build(["hello world", "hello there"])
        assert "hello" in vocab.word2idx
        assert "world" in vocab.word2idx
        assert len(vocab) > 2  # PAD + UNK + actual words
    
    def test_min_freq_filtering(self):
        vocab = Vocabulary(min_freq=2)
        vocab.build(["hello world", "hello there"])
        # "hello" appears 2x, "world" and "there" appear 1x each
        assert "hello" in vocab.word2idx
        assert "world" not in vocab.word2idx
        assert "there" not in vocab.word2idx
    
    def test_encode_known_words(self):
        vocab = Vocabulary(min_freq=1)
        vocab.build(["hello world"])
        result = vocab.encode("hello world", max_len=5)
        assert len(result) == 5
        assert result[0] == vocab.word2idx["hello"]
        assert result[1] == vocab.word2idx["world"]
        # Remaining should be PAD
        assert result[2] == vocab.PAD_IDX
    
    def test_encode_unknown_words(self):
        vocab = Vocabulary(min_freq=1)
        vocab.build(["hello world"])
        result = vocab.encode("unknown words", max_len=3)
        assert result[0] == vocab.UNK_IDX
        assert result[1] == vocab.UNK_IDX
    
    def test_encode_truncation(self):
        vocab = Vocabulary(min_freq=1)
        vocab.build(["a b c d e f"])
        result = vocab.encode("a b c d e f", max_len=3)
        assert len(result) == 3
    
    def test_encode_empty_string(self):
        vocab = Vocabulary(min_freq=1)
        vocab.build(["hello"])
        result = vocab.encode("", max_len=5)
        assert result == [vocab.PAD_IDX] * 5
    
    def test_save_and_load(self, tmp_path):
        vocab = Vocabulary(min_freq=1)
        vocab.build(["hello world test"])
        
        save_path = str(tmp_path / "vocab.json")
        vocab.save(save_path)
        
        loaded = Vocabulary.load(save_path)
        assert len(loaded) == len(vocab)
        assert loaded.word2idx == vocab.word2idx


# =========================================================================
# CategoricalEncoder Tests
# =========================================================================

class TestCategoricalEncoder:
    """Tests for the CategoricalEncoder class."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "color": ["red", "blue", "red", "green"],
            "size": ["S", "M", "L", "S"],
        })
    
    def test_fit_basic(self, sample_df):
        encoder = CategoricalEncoder()
        encoder.fit(sample_df, ["color", "size"])
        
        assert "color" in encoder.encoders
        assert "size" in encoder.encoders
    
    def test_transform_known_values(self, sample_df):
        encoder = CategoricalEncoder()
        encoder.fit(sample_df, ["color"])
        
        result = encoder.transform(sample_df, ["color"])
        # All values should be > 0 (0 is reserved for unknown)
        assert (result["color"] > 0).all()
    
    def test_transform_unknown_values(self, sample_df):
        encoder = CategoricalEncoder()
        encoder.fit(sample_df, ["color"])
        
        new_df = pd.DataFrame({"color": ["purple", "orange"]})
        result = encoder.transform(new_df, ["color"])
        # Unknown values should map to 0
        assert (result["color"] == 0).all()
    
    def test_num_classes(self, sample_df):
        encoder = CategoricalEncoder()
        encoder.fit(sample_df, ["color"])
        # 3 unique colors + 1 unknown = 4
        assert encoder.get_num_classes("color") == 4
    
    def test_save_and_load(self, sample_df, tmp_path):
        encoder = CategoricalEncoder()
        encoder.fit(sample_df, ["color", "size"])
        
        save_path = str(tmp_path / "encoder.json")
        encoder.save(save_path)
        
        loaded = CategoricalEncoder.load(save_path)
        assert loaded.encoders == encoder.encoders


# =========================================================================
# Preprocess DataFrame Tests
# =========================================================================

class TestPreprocessDataframe:
    """Tests for the preprocess_dataframe function."""
    
    @pytest.fixture
    def raw_df(self):
        """Create a minimal raw DataFrame mimicking Mercari data."""
        return pd.DataFrame({
            "train_id": [1, 2, 3, 4, 5],
            "name": ["Nike Shoes", "Vintage Dress", "iPhone Case", "Free Item", "Broken Toy"],
            "item_condition_id": [1, 3, 2, 1, 5],
            "category_name": [
                "Men/Shoes/Athletic",
                "Women/Dresses/Above Knee",
                "Electronics/Cell Phones/Cases",
                "Women/Other/Other",
                None,  # Missing category
            ],
            "brand_name": ["Nike", "Vintage", None, None, "Unknown"],
            "price": [55.0, 25.0, 10.0, 0.0, 15.0],  # One zero price
            "shipping": [1, 0, 1, 0, 1],
            "item_description": [
                "Great shoes, barely worn",
                "Beautiful [rm] dress",
                "Fits iPhone <b>12</b>",
                "Free stuff",
                None,  # Missing description
            ],
        })
    
    def test_removes_zero_prices(self, raw_df):
        result = preprocess_dataframe(raw_df)
        assert (result["price"] > 0).all()
        assert len(result) == 4  # One removed
    
    def test_log_price_created(self, raw_df):
        result = preprocess_dataframe(raw_df)
        assert "log_price" in result.columns
        # log1p(55) ≈ 4.025
        np.testing.assert_almost_equal(
            result.iloc[0]["log_price"], np.log1p(55.0), decimal=3
        )
    
    def test_text_cleaned(self, raw_df):
        result = preprocess_dataframe(raw_df)
        assert "name_clean" in result.columns
        assert "desc_clean" in result.columns
        # Check [rm] was removed
        assert "[rm]" not in result["desc_clean"].iloc[1]
    
    def test_categories_parsed(self, raw_df):
        result = preprocess_dataframe(raw_df)
        assert "main_cat" in result.columns
        assert "sub_cat1" in result.columns
        assert "sub_cat2" in result.columns
        assert result["main_cat"].iloc[0] == "men"
    
    def test_missing_brand_filled(self, raw_df):
        result = preprocess_dataframe(raw_df)
        assert result["brand_name"].notna().all()
        # NaN brands should become "unknown"
        assert "unknown" in result["brand_name"].values
    
    def test_missing_category_handled(self, raw_df):
        result = preprocess_dataframe(raw_df)
        # The row with None category should have "missing"
        assert "missing" in result["main_cat"].values


# =========================================================================
# MercariDataset Tests
# =========================================================================

class TestMercariDataset:
    """Tests for the PyTorch Dataset class."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a minimal Dataset from a small DataFrame."""
        df = pd.DataFrame({
            "name_seq": [[1, 2, 3, 0, 0]] * 10,
            "desc_seq": [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]] * 10,
            "main_cat": [1] * 10,
            "sub_cat1": [2] * 10,
            "sub_cat2": [3] * 10,
            "brand_name": [4] * 10,
            "item_condition_id": [1] * 10,
            "shipping": [1.0] * 10,
            "log_price": [3.5] * 10,
        })
        return MercariDataset(df)
    
    def test_len(self, sample_dataset):
        assert len(sample_dataset) == 10
    
    def test_getitem_keys(self, sample_dataset):
        batch = sample_dataset[0]
        expected_keys = {"name_seq", "desc_seq", "categoricals", "shipping", "target"}
        assert set(batch.keys()) == expected_keys
    
    def test_getitem_dtypes(self, sample_dataset):
        batch = sample_dataset[0]
        assert batch["name_seq"].dtype == torch.int64
        assert batch["desc_seq"].dtype == torch.int64
        assert batch["categoricals"].dtype == torch.int64
        assert batch["shipping"].dtype == torch.float32
        assert batch["target"].dtype == torch.float32
    
    def test_getitem_shapes(self, sample_dataset):
        batch = sample_dataset[0]
        assert batch["name_seq"].shape == (5,)
        assert batch["desc_seq"].shape == (10,)
        assert batch["categoricals"].shape == (5,)
        assert batch["shipping"].shape == ()
        assert batch["target"].shape == ()
    
    def test_target_value(self, sample_dataset):
        batch = sample_dataset[0]
        assert batch["target"].item() == pytest.approx(3.5)


# =========================================================================
# Integration Test — Full Pipeline on Real Data
# =========================================================================

class TestIntegration:
    """Integration tests using the actual processed data."""
    
    @pytest.fixture
    def processed_data_dir(self):
        path = Path("data/processed")
        if not path.exists():
            pytest.skip("Processed data not found. Run preprocessing first.")
        return path
    
    def test_metadata_exists(self, processed_data_dir):
        metadata_path = processed_data_dir / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["train_size"] > 0
        assert metadata["val_size"] > 0
        assert metadata["test_size"] > 0
        assert metadata["name_vocab_size"] > 100
        assert metadata["desc_vocab_size"] > 100
    
    def test_train_data_loadable(self, processed_data_dir):
        df = pd.read_pickle(processed_data_dir / "train.pkl")
        assert len(df) > 0
        assert "name_seq" in df.columns
        assert "log_price" in df.columns
    
    def test_vocab_loadable(self, processed_data_dir):
        vocab = Vocabulary.load(str(processed_data_dir / "name_vocab.json"))
        assert len(vocab) > 100
    
    def test_encoder_loadable(self, processed_data_dir):
        encoder = CategoricalEncoder.load(str(processed_data_dir / "cat_encoder.json"))
        assert "main_cat" in encoder.encoders
        assert "brand_name" in encoder.encoders
    
    def test_dataset_from_processed(self, processed_data_dir):
        """End-to-end: load processed data → create Dataset → get a batch."""
        df = pd.read_pickle(processed_data_dir / "train.pkl")
        dataset = MercariDataset(df)
        
        assert len(dataset) > 0
        
        batch = dataset[0]
        assert batch["name_seq"].shape[0] == 10   # max_name_len
        assert batch["desc_seq"].shape[0] == 75    # max_desc_len
        assert batch["categoricals"].shape[0] == 5  # 5 cat features
        assert batch["target"].item() > 0           # log price > 0
