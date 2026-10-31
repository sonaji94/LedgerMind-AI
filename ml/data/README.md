# ULB Credit Card Fraud Detection Dataset

## Provenance & Attribution
- **Source**: Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB) & Worldline.
- **Published on**: Kaggle ([Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) & OpenML (Dataset ID: 1597).
- **Authors**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.
- **License**: Open Database License (ODbL) v1.0.

## Citation
> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. In IEEE Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

---

## Dataset Description

The dataset contains credit card transactions made by European cardholders in September 2013 across two days:
- **Total Transactions**: 284,807
- **Total Fraudulent Transactions**: 492
- **Class Imbalance**: 0.172% positive class (492 / 284,807)

### Data Dictionary

| Column | Type | Description |
|:---|:---:|:---|
| `Time` | `float64` | Seconds elapsed between this transaction and the first transaction in the dataset. |
| `V1` – `V28` | `float64` | Principal components obtained using PCA to protect sensitive customer identities and card information. |
| `Amount` | `float64` | Transaction amount in Euros. |
| `Class` | `int64` | Ground-truth binary target: `1` for fraudulent transactions, `0` otherwise. |

---

## File Integrity & Placement

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the uncompressed file at:
   `ml/data/raw/creditcard.csv` (or `ml/data/creditcard.csv`)
3. Expected SHA-256 Checksum:
   `9222dbacda44f6974cb3eab4b82d96c9c6f2a64c4805c31766a01490226ea542` (uncompressed CSV)
