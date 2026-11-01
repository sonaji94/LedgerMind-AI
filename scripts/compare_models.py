"""
Compare Logistic Regression, Random Forest, and Isolation Forest.
Calibrate optimal decision threshold and save winning production model.

Usage:
    python scripts/compare_models.py
"""

from pathlib import Path

import joblib

from ml.preprocessing.pipeline import create_feature_pipeline
from ml.preprocessing.time_split import split_by_time
from ml.preprocessing.validator import load_dataset
from ml.training.baseline import evaluate_classification, train_logistic_regression
from ml.training.comparison import (
    evaluate_isolation_forest,
    find_optimal_threshold,
    train_isolation_forest,
    train_random_forest,
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

    print("⚙️ Transforming features with preprocessor...")
    preprocessor = create_feature_pipeline()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    print("\n" + "=" * 65)
    print("🏆 MODEL EXPERIMENT 1: Class-Weighted Logistic Regression")
    print("=" * 65)
    lr_model = train_logistic_regression(X_train, y_train, class_weight="balanced")
    lr_val_metrics = evaluate_classification(lr_model, X_val, y_val)
    lr_val_metrics.print_report("Logistic Regression (Validation)")

    print("\n" + "=" * 65)
    print("🏆 MODEL EXPERIMENT 2: Random Forest Classifier (50 Trees)")
    print("=" * 65)
    rf_model = train_random_forest(X_train, y_train, n_estimators=50, max_depth=8)
    rf_val_metrics = evaluate_classification(rf_model, X_val, y_val)
    rf_val_metrics.print_report("Random Forest (Validation)")

    print("\n" + "=" * 65)
    print("🏆 MODEL EXPERIMENT 3: Isolation Forest (Unsupervised Anomaly Detection)")
    print("=" * 65)
    iso_model = train_isolation_forest(X_train)
    iso_val_metrics = evaluate_isolation_forest(iso_model, X_val, y_val)
    iso_val_metrics.print_report("Isolation Forest (Validation)")

    # Threshold calibration on validation set
    print("\n" + "=" * 65)
    print("🎯 COST-SENSITIVE THRESHOLD CALIBRATION (Random Forest)")
    print("=" * 65)
    rf_val_probas = rf_model.predict_proba(X_val)[:, 1]
    best_cal = find_optimal_threshold(
        y_val, rf_val_probas, cost_fp=10.0, cost_fn=250.0
    )
    print(f"Optimal Threshold        : {best_cal.threshold:.3f}")
    print(f"Calibrated Precision     : {best_cal.precision:.4f}")
    print(f"Calibrated Recall        : {best_cal.recall:.4f}")
    print(f"Calibrated F1-Score      : {best_cal.f1:.4f}")
    print(f"Total Financial Loss     : ${best_cal.total_financial_loss:,.2f}")
    print(f"False Alarms (FP @ $10)  : {best_cal.false_positives}")
    print(f"Missed Fraud (FN @ $250) : {best_cal.false_negatives}")

    # Evaluate winning model on test set using optimal threshold
    print("\n" + "=" * 65)
    print("🏁 FINAL HELD-OUT TEST EVALUATION (Winning Model: Random Forest)")
    print("=" * 65)
    test_metrics = evaluate_classification(
        rf_model, X_test, y_test, threshold=best_cal.threshold
    )
    test_metrics.print_report("Random Forest (Final Test Set)")

    # Save production model
    models_dir = Path("ml/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    prod_model_path = models_dir / "fraud_model.joblib"
    joblib.dump(
        {
            "model": rf_model,
            "optimal_threshold": best_cal.threshold,
            "model_type": "RandomForestClassifier",
            "features_count": 31,
        },
        prod_model_path,
    )
    print(f"\n💾 Production model bundle saved to: {prod_model_path}")
    print("✅ Model comparison & threshold calibration complete!")


if __name__ == "__main__":
    main()
