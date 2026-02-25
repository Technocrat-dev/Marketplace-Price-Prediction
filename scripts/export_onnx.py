"""
ONNX export for the Mercari Price Prediction model.

Exports the trained PyTorch model to ONNX format for production inference.
Includes validation that the ONNX model produces identical outputs to PyTorch.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint outputs/checkpoints/best_model.pt
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.multimodal import MercariPricePredictor


class ONNXWrapper(torch.nn.Module):
    """
    Wrapper that unpacks flat tensor inputs into the batch dict
    expected by MercariPricePredictor.
    
    ONNX doesn't support dict inputs, so we accept individual tensors.
    """
    
    def __init__(self, model: MercariPricePredictor):
        super().__init__()
        self.model = model
    
    def forward(
        self,
        name_seq: torch.Tensor,
        desc_seq: torch.Tensor,
        categoricals: torch.Tensor,
        shipping: torch.Tensor,
    ) -> torch.Tensor:
        batch = {
            "name_seq": name_seq,
            "desc_seq": desc_seq,
            "categoricals": categoricals,
            "shipping": shipping,
        }
        return self.model(batch)


def export_to_onnx(
    checkpoint_path: str,
    metadata_path: str,
    output_path: str,
    max_name_len: int = 10,
    max_desc_len: int = 75,
) -> None:
    """
    Export a trained model checkpoint to ONNX.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint
        metadata_path: Path to metadata.json
        output_path: Path to save the .onnx file
        max_name_len: Max token length for names
        max_desc_len: Max token length for descriptions
    """
    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Build model
    model = MercariPricePredictor(
        name_vocab_size=metadata["name_vocab_size"],
        desc_vocab_size=metadata["desc_vocab_size"],
        cat_dims=metadata["cat_sizes"],
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"[INFO] Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Wrap for ONNX (flat tensor inputs)
    wrapper = ONNXWrapper(model)
    wrapper.eval()
    
    # Create dummy inputs
    batch_size = 1
    dummy_name = torch.randint(0, 100, (batch_size, max_name_len))
    dummy_desc = torch.randint(0, 100, (batch_size, max_desc_len))
    dummy_cats = torch.randint(0, 5, (batch_size, 5))
    dummy_ship = torch.tensor([1.0]).unsqueeze(0) if batch_size == 1 else torch.ones(batch_size)
    dummy_ship = torch.ones(batch_size)
    
    # Export
    print(f"\n[INFO] Exporting to ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy_name, dummy_desc, dummy_cats, dummy_ship),
        output_path,
        input_names=["name_seq", "desc_seq", "categoricals", "shipping"],
        output_names=["predicted_log_price"],
        dynamic_axes={
            "name_seq": {0: "batch_size"},
            "desc_seq": {0: "batch_size"},
            "categoricals": {0: "batch_size"},
            "shipping": {0: "batch_size"},
            "predicted_log_price": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"  ONNX file size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    
    # Validate
    _validate_onnx(wrapper, output_path, dummy_name, dummy_desc, dummy_cats, dummy_ship)


def _validate_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    name_seq: torch.Tensor,
    desc_seq: torch.Tensor,
    categoricals: torch.Tensor,
    shipping: torch.Tensor,
) -> None:
    """Validate that ONNX model produces identical outputs to PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime not installed — skipping ONNX validation")
        print("  Install with: pip install onnxruntime")
        return
    
    # PyTorch prediction
    with torch.no_grad():
        pytorch_out = pytorch_model(name_seq, desc_seq, categoricals, shipping)
    pytorch_out = pytorch_out.numpy()
    
    # ONNX prediction
    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(
        None,
        {
            "name_seq": name_seq.numpy(),
            "desc_seq": desc_seq.numpy(),
            "categoricals": categoricals.numpy(),
            "shipping": shipping.numpy(),
        },
    )[0]
    
    # Compare
    max_diff = np.max(np.abs(pytorch_out - onnx_out))
    print(f"\n[VALIDATION] PyTorch vs ONNX:")
    print(f"  PyTorch output: {pytorch_out.flatten()[:3]}")
    print(f"  ONNX output:    {onnx_out.flatten()[:3]}")
    print(f"  Max difference: {max_diff:.8f}")
    
    if max_diff < 1e-4:
        print("  ✅ ONNX export validated — outputs match!")
    else:
        print("  ⚠️  WARNING: Outputs differ by more than 1e-4")


def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: best_model.pt)")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    checkpoint_path = args.checkpoint or str(
        Path(cfg["paths"]["checkpoints"]) / "best_model.pt"
    )
    metadata_path = str(Path(cfg["paths"]["processed_data"]) / "metadata.json")
    output_path = cfg["paths"]["onnx_model"]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    export_to_onnx(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        output_path=output_path,
        max_name_len=cfg["data"]["max_name_len"],
        max_desc_len=cfg["data"]["max_desc_len"],
    )


if __name__ == "__main__":
    main()
