"""
Feature Engineering Utilities for the Mercari Price Prediction Pipeline.

Adds derived features to improve model quality:
- Text statistics (word counts, char counts, caps ratio)
- Price-relevant signals (description quality, brand presence)
- Category depth features
"""

from typing import Dict

import numpy as np
import pandas as pd


# =========================================================================
# Text Statistics
# =========================================================================

def compute_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add text-derived features to the DataFrame."""
    
    # Name features
    df["name_word_count"] = df["name"].fillna("").str.split().str.len().fillna(0).astype(np.float32)
    df["name_char_count"] = df["name"].fillna("").str.len().astype(np.float32)
    df["name_caps_ratio"] = (
        df["name"].fillna("").str.count(r"[A-Z]") /
        df["name_char_count"].clip(lower=1)
    ).astype(np.float32)
    
    # Description features  
    desc = df["item_description"].fillna("")
    df["desc_word_count"] = desc.str.split().str.len().fillna(0).astype(np.float32)
    df["desc_char_count"] = desc.str.len().astype(np.float32)
    df["desc_has_content"] = (df["desc_word_count"] > 0).astype(np.float32)
    
    # Punctuation and special chars (signal of listing quality)
    df["desc_exclaim_count"] = desc.str.count("!").astype(np.float32)
    df["desc_question_count"] = desc.str.count(r"\?").astype(np.float32)
    
    return df


# =========================================================================
# Brand and Category Features
# =========================================================================

def compute_brand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add brand-derived features."""
    
    # Binary: known brand vs unknown
    df["has_brand"] = (
        ~df["brand_name"].fillna("").str.lower().isin(["", "unknown", "no brand"])
    ).astype(np.float32)
    
    return df


def compute_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add category depth features."""
    
    cat = df["category_name"].fillna("")
    df["category_depth"] = cat.str.count("/").astype(np.float32) + (cat.str.len() > 0).astype(np.float32)
    
    return df


# =========================================================================
# Pipeline
# =========================================================================

ENGINEERED_FEATURES = [
    "name_word_count", "name_char_count", "name_caps_ratio",
    "desc_word_count", "desc_char_count", "desc_has_content",
    "desc_exclaim_count", "desc_question_count",
    "has_brand", "category_depth",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = compute_text_stats(df)
    df = compute_brand_features(df)
    df = compute_category_features(df)
    return df


def get_engineered_features_array(df: pd.DataFrame) -> np.ndarray:
    """Extract the engineered features as a numpy array."""
    df = engineer_features(df)
    return df[ENGINEERED_FEATURES].values.astype(np.float32)


def engineer_single_item(
    name: str, 
    description: str, 
    brand_name: str, 
    category_name: str,
) -> Dict[str, float]:
    """Engineer features for a single prediction request."""
    
    name = name or ""
    description = description or ""
    brand_name = brand_name or ""
    category_name = category_name or ""
    
    words_name = name.split()
    words_desc = description.split()
    name_len = max(len(name), 1)
    
    return {
        "name_word_count": float(len(words_name)),
        "name_char_count": float(len(name)),
        "name_caps_ratio": sum(1 for c in name if c.isupper()) / name_len,
        "desc_word_count": float(len(words_desc)),
        "desc_char_count": float(len(description)),
        "desc_has_content": float(len(words_desc) > 0),
        "desc_exclaim_count": float(description.count("!")),
        "desc_question_count": float(description.count("?")),
        "has_brand": float(brand_name.lower() not in ("", "unknown", "no brand")),
        "category_depth": float(category_name.count("/") + (len(category_name) > 0)),
    }
