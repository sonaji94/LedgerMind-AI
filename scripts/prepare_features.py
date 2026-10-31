"""
Prepare and fit feature engineering pipeline on training split.

Usage:
    python scripts/prepare_features.py
"""

from pathlib import Path

import joblib

from ml.preprocessing.pipeline import create_feature_pipeline
from ml.preprocessing.time_split import split_by_time
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

    print(f"📂 Loading dataset from: {data_path}...")
    df = load_dataset(data_path)

    print("⏱️ Splitting dataset chronologically (70% Train, 15% Val, 15% Test)...")
    split = split_by_time(df, train_ratio=0.70, val_ratio=0.15)

    print(f"  • Train Set : {len(split.train_df):,} records | Time: {split.train_time_range[0]:.0f}s -> {split.train_time_range[1]:.0f}s")
    print(f"  • Val Set   : {len(split.val_df):,} records | Time: {split.val_time_range[0]:.0f}s -> {split.val_time_range[1]:.0f}s")
    print(f"  • Test Set  : {len(split.test_df):,} records | Time: {split.test_time_range[0]:.0f}s -> {split.test_time_range[1]:.0f}s")

    print("\n⚙️ Fitting feature pipeline strictly on Training Set (preventing leakage)...")
    preprocessor = create_feature_pipeline()

    # Separate features and labels
    X_train = split.train_df.drop(columns=["Class"])
    y_train = split.train_df["Class"]

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(split.val_df.drop(columns=["Class"]))
    X_test_transformed = preprocessor.transform(split.test_df.drop(columns=["Class"]))

    print(
        f"  • Transformed Shapes: Train={X_train_transformed.shape}, "
        f"Val={X_val_transformed.shape}, Test={X_test_transformed.shape}"
    )
    print(f"  • Train Fraud Count         : {int(y_train.sum())} ({y_train.mean() * 100:.3f}%)")
    print(f"  • Val Fraud Count           : {int(split.val_df['Class'].sum())}")
    print(f"  • Test Fraud Count          : {int(split.test_df['Class'].sum())}")

    # Save fitted preprocessor
    models_dir = Path("ml/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = models_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, pipeline_path)
    print(f"\n💾 Saved fitted preprocessor to: {pipeline_path}")
    print("✅ Feature preparation complete!")


if __name__ == "__main__":
    main()
