"""
Script to acquire, verify, and validate the ULB Credit Card Fraud dataset.

Usage:
    python scripts/acquire_data.py [--sample]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ml.preprocessing.validator import (
    generate_synthetic_benchmark,
    load_dataset,
    validate_dataframe,
)

EXPECTED_SHA256 = "9222dbacda44f6974cb3eab4b82d96c9c6f2a64c4805c31766a01490226ea542"
RAW_DATA_PATHS = [
    Path("ml/data/raw/creditcard.csv"),
    Path("ml/data/creditcard.csv"),
]


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file efficiently in chunks."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(1024 * 1024):  # 1MB buffer
            sha256.update(chunk)
    return sha256.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and validate credit card fraud data.")
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a local synthetic development dataset if raw file is not present.",
    )
    args = parser.parse_args()

    # Search for raw dataset in known locations
    dataset_path: Path | None = None
    for p in RAW_DATA_PATHS:
        if p.is_file():
            dataset_path = p
            break

    if dataset_path:
        print(f"📁 Found existing dataset at: {dataset_path}")
        print("🔐 Computing SHA-256 checksum...")
        actual_hash = calculate_sha256(dataset_path)
        print(f"Calculated Hash: {actual_hash}")
        if actual_hash == EXPECTED_SHA256:
            print("✅ SHA-256 checksum matches official Kaggle release!")
        else:
            print("ℹ️ Checksum differs (file might be re-exported or sample). Continuing validation...")

        df = load_dataset(dataset_path)
        summary = validate_dataframe(df)
        summary.print_report()
        return

    print("⚠️ No raw dataset found at 'ml/data/raw/creditcard.csv'.")
    print("\nHow to acquire the full dataset:")
    print("1. Download 'creditcard.csv' from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("2. Place the uncompressed CSV at: ml/data/raw/creditcard.csv\n")

    if args.generate_sample:
        target_dir = Path("ml/data/raw")
        target_dir.mkdir(parents=True, exist_ok=True)
        sample_path = target_dir / "creditcard_sample.csv"
        print(f"🔨 Generating synthetic development dataset at: {sample_path} (5,000 records)...")
        df_sample = generate_synthetic_benchmark(n_samples=5000, fraud_rate=0.002)
        df_sample.to_csv(sample_path, index=False)
        print("✅ Sample created successfully!")
        summary = validate_dataframe(df_sample)
        summary.print_report()
    else:
        print("💡 Tip: To generate a local synthetic dataset for immediate testing, run:")
        print("   python scripts/acquire_data.py --generate-sample")


if __name__ == "__main__":
    main()
