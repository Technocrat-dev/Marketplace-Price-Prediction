"""
Data preprocessing pipeline for the Mercari Price Prediction Engine.

Handles:
- Loading raw TSV data
- Text cleaning (name, description)
- Category hierarchy parsing (3 levels)
- Missing value imputation
- Label encoding for categoricals
- Vocabulary building for text fields
- Log-transform of price target
- Train/val/test splitting
- Saving processed artifacts
"""

import re
import json
import pickle
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean a single text string: lowercase, remove special chars, normalize spaces."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    text = text.lower().strip()
    
    # Remove [rm] tokens (Kaggle price redaction)
    text = re.sub(r"\[rm\]", "", text)
    
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    
    # Keep only alphanumeric, spaces, and basic punctuation
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# ---------------------------------------------------------------------------
# Category Parsing
# ---------------------------------------------------------------------------

def parse_category(category: str) -> Tuple[str, str, str]:
    """
    Parse hierarchical category string into 3 levels.
    
    Example: 'Women/Tops & Blouses/Blouse' -> ('women', 'tops & blouses', 'blouse')
    """
    if not isinstance(category, str) or category.strip() == "":
        return ("missing", "missing", "missing")
    
    parts = category.lower().split("/")
    
    main_cat = parts[0].strip() if len(parts) > 0 else "missing"
    sub_cat1 = parts[1].strip() if len(parts) > 1 else "missing"
    sub_cat2 = parts[2].strip() if len(parts) > 2 else "missing"
    
    return (main_cat, sub_cat1, sub_cat2)


# ---------------------------------------------------------------------------
# Vocabulary Builder
# ---------------------------------------------------------------------------

class Vocabulary:
    """Word-to-index mapping for text tokenization."""
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1
    
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        self.word_freq: Counter = Counter()
    
    def build(self, texts: List[str]) -> "Vocabulary":
        """Build vocabulary from a list of cleaned text strings."""
        for text in texts:
            if isinstance(text, str):
                self.word_freq.update(text.split())
        
        idx = len(self.word2idx)
        for word, freq in self.word_freq.most_common():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        return self
    
    def encode(self, text: str, max_len: int) -> List[int]:
        """Convert text to list of token indices, padded/truncated to max_len."""
        if not isinstance(text, str) or text.strip() == "":
            return [self.PAD_IDX] * max_len
        
        tokens = text.split()[:max_len]
        indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        
        # Pad to max_len
        indices += [self.PAD_IDX] * (max_len - len(indices))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def save(self, path: str) -> None:
        """Save vocabulary to JSON."""
        data = {
            "word2idx": self.word2idx,
            "min_freq": self.min_freq,
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        
        vocab = cls(min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab


# ---------------------------------------------------------------------------
# Label Encoder (for categorical features)
# ---------------------------------------------------------------------------

class CategoricalEncoder:
    """Maps categorical string values to integer indices with an 'unknown' fallback."""
    
    UNKNOWN_TOKEN = "<UNK>"
    UNKNOWN_IDX = 0
    
    def __init__(self):
        self.encoders: Dict[str, Dict[str, int]] = {}
    
    def fit(self, df: pd.DataFrame, columns: List[str]) -> "CategoricalEncoder":
        """Build label mappings for specified columns."""
        for col in columns:
            unique_vals = sorted(df[col].dropna().unique().tolist())
            mapping = {self.UNKNOWN_TOKEN: self.UNKNOWN_IDX}
            for i, val in enumerate(unique_vals, start=1):
                mapping[str(val)] = i
            self.encoders[col] = mapping
        
        return self
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply label encoding to specified columns."""
        df = df.copy()
        for col in columns:
            if col not in self.encoders:
                raise ValueError(f"Column '{col}' has not been fitted.")
            mapping = self.encoders[col]
            df[col] = df[col].apply(
                lambda x: mapping.get(str(x), self.UNKNOWN_IDX)
            )
        return df
    
    def get_num_classes(self, col: str) -> int:
        """Return number of unique categories for a column (incl. unknown)."""
        return len(self.encoders[col])
    
    def save(self, path: str) -> None:
        """Save encoders to JSON."""
        with open(path, "w") as f:
            json.dump(self.encoders, f)
    
    @classmethod
    def load(cls, path: str) -> "CategoricalEncoder":
        """Load encoders from JSON."""
        encoder = cls()
        with open(path, "r") as f:
            encoder.encoders = json.load(f)
        return encoder


# ---------------------------------------------------------------------------
# Main Preprocessing Pipeline
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: str) -> pd.DataFrame:
    """Load raw training data from TSV."""
    path = Path(data_dir) / "train.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}")
    
    df = pd.read_csv(path, sep="\t")
    print(f"[INFO] Loaded {len(df):,} rows from {path}")
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning and feature engineering to raw dataframe."""
    print("[INFO] Preprocessing dataframe...")
    initial_len = len(df)
    
    # 1. Remove zero/negative prices
    df = df[df["price"] > 0].copy()
    removed = initial_len - len(df)
    if removed > 0:
        print(f"  Removed {removed} zero/negative price rows")
    
    # 2. Log-transform price (target)
    df["log_price"] = np.log1p(df["price"])
    
    # 3. Clean text fields
    print("  Cleaning text fields...")
    df["name_clean"] = df["name"].apply(clean_text)
    df["desc_clean"] = df["item_description"].apply(clean_text)
    
    # 4. Parse category hierarchy
    print("  Parsing categories...")
    cat_parsed = df["category_name"].apply(parse_category)
    df["main_cat"] = cat_parsed.apply(lambda x: x[0])
    df["sub_cat1"] = cat_parsed.apply(lambda x: x[1])
    df["sub_cat2"] = cat_parsed.apply(lambda x: x[2])
    
    # 5. Handle missing brand names
    df["brand_name"] = df["brand_name"].fillna("unknown").str.lower().str.strip()
    
    # 6. Ensure item_condition_id is in expected range (1-5)
    df["item_condition_id"] = df["item_condition_id"].clip(1, 5)
    
    print(f"  Final shape: {df.shape}")
    return df


def build_artifacts(
    df: pd.DataFrame,
    min_word_freq: int = 2,
    max_name_len: int = 10,
    max_desc_len: int = 75,
) -> Tuple[Vocabulary, Vocabulary, CategoricalEncoder]:
    """
    Build vocabularies and categorical encoders from the training data.
    
    Returns:
        name_vocab: Vocabulary for product names
        desc_vocab: Vocabulary for descriptions  
        cat_encoder: CategoricalEncoder for categorical columns
    """
    print("[INFO] Building vocabularies...")
    
    # Build text vocabularies
    name_vocab = Vocabulary(min_freq=min_word_freq)
    name_vocab.build(df["name_clean"].tolist())
    print(f"  Name vocabulary: {len(name_vocab):,} words")
    
    desc_vocab = Vocabulary(min_freq=min_word_freq)
    desc_vocab.build(df["desc_clean"].tolist())
    print(f"  Description vocabulary: {len(desc_vocab):,} words")
    
    # Build categorical encoders
    cat_columns = ["main_cat", "sub_cat1", "sub_cat2", "brand_name", "item_condition_id"]
    cat_encoder = CategoricalEncoder()
    cat_encoder.fit(df, cat_columns)
    
    for col in cat_columns:
        print(f"  {col}: {cat_encoder.get_num_classes(col)} categories")
    
    return name_vocab, desc_vocab, cat_encoder


def encode_features(
    df: pd.DataFrame,
    name_vocab: Vocabulary,
    desc_vocab: Vocabulary,
    cat_encoder: CategoricalEncoder,
    max_name_len: int = 10,
    max_desc_len: int = 75,
) -> pd.DataFrame:
    """Encode all features into model-ready format."""
    print("[INFO] Encoding features...")
    
    # Encode text to token indices
    df["name_seq"] = df["name_clean"].apply(
        lambda x: name_vocab.encode(x, max_name_len)
    )
    df["desc_seq"] = df["desc_clean"].apply(
        lambda x: desc_vocab.encode(x, max_desc_len)
    )
    
    # Encode categoricals
    cat_columns = ["main_cat", "sub_cat1", "sub_cat2", "brand_name", "item_condition_id"]
    df = cat_encoder.transform(df, cat_columns)
    
    print("  Encoding complete.")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test sets."""
    print("[INFO] Splitting data...")
    
    # Create price bins for stratified splitting
    df["price_bin"] = pd.qcut(df["log_price"], q=10, labels=False, duplicates="drop")
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_seed,
        stratify=df["price_bin"]
    )
    
    # Second split: train vs val
    adjusted_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=adjusted_val_size, random_state=random_seed,
        stratify=train_val["price_bin"]
    )
    
    # Drop temporary price_bin column
    train = train.drop(columns=["price_bin"])
    val = val.drop(columns=["price_bin"])
    test = test.drop(columns=["price_bin"])
    
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


def _save_split(df: pd.DataFrame, output_path: Path, prefix: str) -> None:
    """
    Save a single data split as numpy arrays (memory-efficient).
    
    Saves:
      - {prefix}_name_seq.npy   — [N, max_name_len] int32
      - {prefix}_desc_seq.npy   — [N, max_desc_len] int32
      - {prefix}_tabular.npy    — [N, 7] float32 (cats + shipping + log_price)
    """
    # Convert sequence lists to numpy arrays
    name_seqs = np.array(df["name_seq"].tolist(), dtype=np.int32)
    desc_seqs = np.array(df["desc_seq"].tolist(), dtype=np.int32)
    
    # Pack scalar columns together: 5 categoricals + shipping + log_price
    cat_cols = ["main_cat", "sub_cat1", "sub_cat2", "brand_name", "item_condition_id"]
    tabular = np.column_stack([
        df[cat_cols].values.astype(np.int32),
        df["shipping"].values.astype(np.float32),
        df["log_price"].values.astype(np.float32),
    ])
    
    np.save(str(output_path / f"{prefix}_name_seq.npy"), name_seqs)
    np.save(str(output_path / f"{prefix}_desc_seq.npy"), desc_seqs)
    np.save(str(output_path / f"{prefix}_tabular.npy"), tabular)
    
    print(f"  {prefix}: {len(df):,} rows → "
          f"name_seq {name_seqs.shape}, desc_seq {desc_seqs.shape}, "
          f"tabular {tabular.shape}")


def save_processed_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    name_vocab: Vocabulary,
    desc_vocab: Vocabulary,
    cat_encoder: CategoricalEncoder,
    output_dir: str = "data/processed",
) -> None:
    """Save all processed data and artifacts to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Saving processed data to {output_path}...")
    
    # Save each split as numpy arrays (much more memory-efficient than pickle)
    _save_split(train, output_path, "train")
    _save_split(val, output_path, "val")
    _save_split(test, output_path, "test")
    
    # Save vocabularies
    name_vocab.save(str(output_path / "name_vocab.json"))
    desc_vocab.save(str(output_path / "desc_vocab.json"))
    
    # Save categorical encoder
    cat_encoder.save(str(output_path / "cat_encoder.json"))
    
    # Save metadata
    metadata = {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "name_vocab_size": len(name_vocab),
        "desc_vocab_size": len(desc_vocab),
        "cat_columns": list(cat_encoder.encoders.keys()),
        "cat_sizes": {col: cat_encoder.get_num_classes(col) 
                      for col in cat_encoder.encoders},
    }
    with open(str(output_path / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("[SUCCESS] All processed data saved!")
    for k, v in metadata.items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    raw_data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    min_word_freq: int = 2,
    max_name_len: int = 10,
    max_desc_len: int = 75,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> None:
    """Run the full preprocessing pipeline end-to-end."""
    print("=" * 60)
    print("Mercari Price Prediction — Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Load
    df = load_raw_data(raw_data_dir)
    
    # Preprocess
    df = preprocess_dataframe(df)
    
    # Build artifacts (vocabularies and encoders) — fit on ALL data before split
    # (so val/test tokens can be encoded, even if rare)
    name_vocab, desc_vocab, cat_encoder = build_artifacts(
        df, min_word_freq, max_name_len, max_desc_len
    )
    
    # Encode features
    df = encode_features(
        df, name_vocab, desc_vocab, cat_encoder, max_name_len, max_desc_len
    )
    
    # Split
    train, val, test = split_data(df, test_size, val_size, random_seed)
    
    # Save
    save_processed_data(
        train, val, test, name_vocab, desc_vocab, cat_encoder, output_dir
    )


if __name__ == "__main__":
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        run_preprocessing_pipeline(
            raw_data_dir=cfg["paths"]["raw_data"],
            output_dir=cfg["paths"]["processed_data"],
            min_word_freq=cfg["data"]["min_word_freq"],
            max_name_len=cfg["data"]["max_name_len"],
            max_desc_len=cfg["data"]["max_desc_len"],
            test_size=cfg["data"]["test_size"],
            val_size=cfg["data"]["val_size"],
            random_seed=cfg["data"]["random_seed"],
        )
    else:
        run_preprocessing_pipeline()
