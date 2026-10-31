"""
Train, evaluate, and save the baseline Logistic Regression model.

Usage:
    python scripts/train_baseline.py
"""

from pathlib import Path

import joblib

from ml.preprocessing.pipeline import create_feature_pipeline
from ml.preprocessing.time_split import split_by_time
from ml.preprocessing.validator import load_dataset
from ml.training.baseline import (
    evaluate_classification,
    train_logistic_regression,
)

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
        print("❌ No dataset found. Run 'python scripts/acquire_data.py --generate-sample' first.")
        return

    print(f"📂 Loading dataset from: {data_path}...")
    df = load_dataset(data_path)

    print("⏱️ Splitting dataset chronologically (70% Train, 15% Val, 15% Test)...")
    split = split_by_time(df, train_ratio=0.70, val_ratio=0.15)

    X_train_raw = split.train_df.drop(columns=["Class"])
    y_train = split.train_df["Class"].to_numpy()

    X_val_raw = split.val_df.drop(columns=["Class"])
    y_val = split.val_df["Class"].to_numpy()

    X_test_raw = split.test_df.drop(columns=["Class"])
    y_test = split.test_df["Class"].to_numpy()

    print("⚙️ Fitting feature pipeline strictly on Training Set...")
    preprocessor = create_feature_pipeline()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    print("🧠 Training Baseline Model: Class-Weighted Logistic Regression...")
    model = train_logistic_regression(X_train, y_train, class_weight="balanced")

    # Evaluate on Validation set (used for threshold tuning)
    val_metrics = evaluate_classification(model, X_val, y_val, threshold=0.50)
    val_metrics.print_report(split_name="Validation Split (Threshold Tuning)")

    # Evaluate on Test set (held-out final verification)
    test_metrics = evaluate_classification(model, X_test, y_test, threshold=0.50)
    test_metrics.print_report(split_name="Test Split (Held-Out Final)")

    # Save model and preprocessor artifacts
    models_dir = Path("ml/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "baseline_logistic_regression.joblib"
    joblib.dump(model, model_path)
    print(f"\n💾 Model artifact saved to: {model_path}")

    preprocessor_path = models_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"💾 Preprocessor artifact saved to: {preprocessor_path}")

    print("\n✅ Baseline model training and evaluation complete!")


if __name__ == "__main__":
    main()
