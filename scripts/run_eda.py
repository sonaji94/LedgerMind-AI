"""
Run Exploratory Data Analysis (EDA) on the available dataset.

Usage:
    python scripts/run_eda.py
"""

from pathlib import Path

from ml.preprocessing.eda import generate_eda_figures, profile_dataset
from ml.preprocessing.validator import load_dataset

DATA_CANDIDATES = [
    Path("ml/data/raw/creditcard.csv"),
    Path("ml/data/creditcard.csv"),
    Path("ml/data/raw/creditcard_sample.csv"),
]


def main() -> None:
    data_path: Path | None = None
    for p in DATA_CANDIDATES:
        if p.is_file():
            data_path = p
            break

    if not data_path:
        print("❌ No dataset found. Please run 'python scripts/acquire_data.py --generate-sample' first.")
        return

    print(f"📊 Loading dataset from: {data_path}...")
    df = load_dataset(data_path)

    print("🔍 Computing statistical profile...")
    profile = profile_dataset(df)
    profile.print_summary()

    print("\n🎨 Generating visual reports in 'reports/figures'...")
    figures = generate_eda_figures(df)
    for fig in figures:
        print(f"  • Generated: {fig}")

    print("\n✅ EDA execution complete!")


if __name__ == "__main__":
    main()
