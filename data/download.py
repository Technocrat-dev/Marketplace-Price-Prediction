"""
Download the Mercari Price Suggestion Challenge dataset from Kaggle.

Usage:
    python data/download.py

Requirements:
    - KAGGLE_API_TOKEN environment variable set, OR
    - kaggle.json in ~/.kaggle/
    - Competition rules accepted at:
      https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/rules
"""

import os
import sys
import zipfile
from pathlib import Path


def download_mercari_dataset(output_dir: str = "data/raw") -> None:
    """Download and extract the Mercari dataset using Kaggle API."""
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    train_file = output_path / "train.tsv"
    if train_file.exists():
        print(f"[INFO] Dataset already exists at {output_path}")
        print(f"  train.tsv: {train_file.stat().st_size / 1e6:.1f} MB")
        return
    
    # Verify Kaggle credentials
    kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_token and not kaggle_json.exists():
        print("[ERROR] No Kaggle credentials found!")
        print("  Set KAGGLE_API_TOKEN env var, or place kaggle.json in ~/.kaggle/")
        sys.exit(1)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)
    
    print("[INFO] Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    competition = "mercari-price-suggestion-challenge"
    
    print(f"[INFO] Downloading dataset from '{competition}'...")
    print(f"[INFO] Saving to: {output_path.resolve()}")
    print("[INFO] This may take a few minutes (~600MB compressed)...")
    
    try:
        api.competition_download_files(
            competition=competition,
            path=str(output_path),
            quiet=False
        )
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "must accept" in error_msg.lower():
            print("\n[ERROR] You must accept the competition rules first!")
            print("  Visit: https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/rules")
            print("  Click 'I Understand and Accept', then re-run this script.")
        else:
            print(f"\n[ERROR] Failed to download: {e}")
        sys.exit(1)
    
    # Extract zip files
    print("[INFO] Extracting files...")
    for zip_file in output_path.glob("*.zip"):
        print(f"  Extracting: {zip_file.name}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(output_path)
        zip_file.unlink()  # Remove zip after extraction
    
    # Also handle .7z files if present (Mercari uses 7z sometimes)
    seven_z_files = list(output_path.glob("*.7z"))
    if seven_z_files:
        print("[WARN] Found .7z files. You may need to extract them manually.")
        print("  Install 7-Zip and run: 7z x <file>")
        for f in seven_z_files:
            print(f"  - {f.name}")
    
    # Verify extraction
    print("\n[INFO] Extracted files:")
    for f in sorted(output_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name}: {size_mb:.1f} MB")
    
    print("\n[SUCCESS] Dataset download complete!")


if __name__ == "__main__":
    # Run from project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    download_mercari_dataset()
