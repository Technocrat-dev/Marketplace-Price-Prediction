"""
PyTorch Dataset and DataLoader for the Mercari Price Prediction Engine.

Loads preprocessed data from numpy .npy files (memory-efficient) and serves
batches of:
- name_seq: tokenized product name (LongTensor)
- desc_seq: tokenized description (LongTensor)
- categoricals: [main_cat, sub_cat1, sub_cat2, brand, condition] (LongTensor)
- shipping: binary shipping flag (FloatTensor)
- target: log1p(price) (FloatTensor)
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MercariDataset(Dataset):
    """PyTorch Dataset for Mercari price prediction.
    
    Loads data from numpy arrays for memory efficiency.
    """
    
    def __init__(
        self,
        name_seqs: np.ndarray,
        desc_seqs: np.ndarray,
        categoricals: np.ndarray,
        shipping: np.ndarray,
        targets: np.ndarray,
    ):
        """
        Args:
            name_seqs: [N, max_name_len] int32
            desc_seqs: [N, max_desc_len] int32
            categoricals: [N, 5] int32
            shipping: [N] float32
            targets: [N] float32 (log1p price)
        """
        self.name_seqs = name_seqs
        self.desc_seqs = desc_seqs
        self.categoricals = categoricals
        self.shipping = shipping
        self.targets = targets
        self._len = len(targets)
    
    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "name_seq": torch.tensor(self.name_seqs[idx], dtype=torch.long),
            "desc_seq": torch.tensor(self.desc_seqs[idx], dtype=torch.long),
            "categoricals": torch.tensor(self.categoricals[idx], dtype=torch.long),
            "shipping": torch.tensor(self.shipping[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


def _load_split(data_path: Path, prefix: str) -> MercariDataset:
    """Load a single data split from numpy files."""
    name_seqs = np.load(data_path / f"{prefix}_name_seq.npy")
    desc_seqs = np.load(data_path / f"{prefix}_desc_seq.npy")
    tabular = np.load(data_path / f"{prefix}_tabular.npy")
    
    # Unpack tabular: columns 0-4 are categoricals, 5 is shipping, 6 is log_price
    categoricals = tabular[:, :5].astype(np.int64)
    shipping = tabular[:, 5].astype(np.float32)
    targets = tabular[:, 6].astype(np.float32)
    
    return MercariDataset(
        name_seqs=name_seqs.astype(np.int64),
        desc_seqs=desc_seqs.astype(np.int64),
        categoricals=categoricals,
        shipping=shipping,
        targets=targets,
    )


def create_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load processed data and create train/val/test DataLoaders.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for training
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory for GPU transfer
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    data_path = Path(data_dir)
    
    # Load metadata
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"[INFO] Loading processed data from {data_path}...")
    
    # Load splits from numpy arrays (memory-efficient)
    train_dataset = _load_split(data_path, "train")
    val_dataset = _load_split(data_path, "val")
    test_dataset = _load_split(data_path, "test")
    
    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"  Batches per epoch — Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    # Quick smoke test
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        batch_size=32, num_workers=0
    )
    
    batch = next(iter(train_loader))
    print("\n[SMOKE TEST] Sample batch shapes:")
    for key, val in batch.items():
        print(f"  {key}: {val.shape} ({val.dtype})")
    
    print(f"\n  Target range: [{batch['target'].min():.2f}, {batch['target'].max():.2f}]")
    print(f"  Metadata: {json.dumps(metadata, indent=2)}")
