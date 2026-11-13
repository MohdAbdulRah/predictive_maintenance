"""
Train all models from the AI_Industrial_Maintenance_Model_Training notebook
on the UCI AI4I 2020 Predictive Maintenance dataset.

Saves individual model files and an ensemble bundle to artifacts/.
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "Machine failure"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load UCI AI4I 2020 dataset, with fallback to local CSV."""
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=601)
        X_raw = dataset.data.features.copy()
        y_raw = dataset.data.targets.copy()
        print("Downloaded AI4I 2020 dataset from UCI.")
    except Exception as exc:
        print(f"UCI loader failed: {repr(exc)}")
        # Try local CSV fallback
        local_csv = ARTIFACTS_DIR / "ai4i2020.csv"
        if local_csv.exists():
            print(f"Loading from local CSV: {local_csv}")
            df = pd.read_csv(local_csv)
            target_candidates = [c for c in df.columns if c.strip().lower() == "machine failure"]
            if not target_candidates:
                raise ValueError(f"Could not find 'Machine failure'. Columns: {df.columns.tolist()}")
            target_col = target_candidates[0]
            y_raw = df[[target_col]].rename(columns={target_col: TARGET})
            X_raw = df.drop(columns=[target_col])
        else:
            raise RuntimeError(
                "UCI loader failed and no local CSV found. "
                f"Please download ai4i2020.csv and place it in {ARTIFACTS_DIR}"
            )
    return X_raw, y_raw


def prepare_features(X_raw, y_raw):
    """Clean and prepare features, excluding leakage columns."""
    if isinstance(y_raw, pd.DataFrame):
        target_matches = [c for c in y_raw.columns if c.strip().lower() == TARGET.lower()]
        if target_matches:
            y = y_raw[target_matches[0]].copy()
        elif y_raw.shape[1] == 1:
            y = y_raw.iloc[:, 0].copy()
        else:
            raise ValueError(f"Unexpected target columns: {y_raw.columns.tolist()}")
    else:
        y = pd.Series(y_raw, name=TARGET)

    y = pd.to_numeric(y, errors="raise").astype(int).reset_index(drop=True)
    X_all = X_raw.reset_index(drop=True).copy()

    leakage_or_id_columns = [
        c for c in X_all.columns
        if c.strip().lower() in {"udi", "product id", "twf", "hdf", "pwf", "osf", "rnf"}
    ]
    X = X_all.drop(columns=leakage_or_id_columns, errors="ignore").copy()

    print(f"Feature shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Excluded leakage columns: {leakage_or_id_columns}")
    print(f"Model input columns: {X.columns.tolist()}")
    return X, y


def make_preprocessor(X, scale_numeric=False):
    """Build sklearn ColumnTransformer for the given dataframe."""
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ], remainder="drop"), categorical_features, numeric_features


def build_candidates(X):
    """Build all 6 candidate model pipelines from the notebook."""
    preprocessor_base, _, _ = make_preprocessor(X, scale_numeric=False)
    preprocessor_scaled, _, _ = make_preprocessor(X, scale_numeric=True)

    candidates = {
        "Dummy baseline": Pipeline([
            ("preprocess", make_preprocessor(X)[0]),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),
        "Logistic Regression": Pipeline([
            ("preprocess", make_preprocessor(X, scale_numeric=True)[0]),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE
            ))
        ]),
        "Decision Tree": Pipeline([
            ("preprocess", make_preprocessor(X)[0]),
            ("model", DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=8,
                random_state=RANDOM_STATE
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocess", make_preprocessor(X)[0]),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "Extra Trees": Pipeline([
            ("preprocess", make_preprocessor(X)[0]),
            ("model", ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("preprocess", make_preprocessor(X)[0]),
            ("model", HistGradientBoostingClassifier(
                max_iter=300,
                random_state=RANDOM_STATE
            ))
        ]),
    }
    return candidates


def evaluate_model(name, estimator, X_val, y_val):
    """Evaluate a fitted model on validation data."""
    val_pred = estimator.predict(X_val)
    row = {
        "model": name,
        "precision_failure": precision_score(y_val, val_pred, zero_division=0),
        "recall_failure": recall_score(y_val, val_pred, zero_division=0),
        "f1_failure": f1_score(y_val, val_pred, zero_division=0),
        "accuracy": accuracy_score(y_val, val_pred),
    }
    if hasattr(estimator, "predict_proba"):
        val_prob = estimator.predict_proba(X_val)[:, 1]
        row["average_precision"] = average_precision_score(y_val, val_prob)
        row["roc_auc"] = roc_auc_score(y_val, val_prob)
    else:
        row["average_precision"] = np.nan
        row["roc_auc"] = np.nan
    return row


def find_best_threshold(model, X_val, y_val):
    """Find optimal decision threshold based on F1 score."""
    if not hasattr(model, "predict_proba"):
        return 0.5
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold_rows = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        pred = (val_prob >= threshold).astype(int)
        threshold_rows.append({
            "threshold": round(float(threshold), 2),
            "f1_failure": f1_score(y_val, pred, zero_division=0),
            "recall_failure": recall_score(y_val, pred, zero_division=0),
        })
    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        ["f1_failure", "recall_failure"], ascending=False
    ).reset_index(drop=True)
    return float(threshold_df.iloc[0]["threshold"])


def train_all():
    """Main training function."""
    print("=" * 60)
    print("AI Industrial Maintenance — Training all models")
    print("=" * 60)

    # Load & prepare data
    X_raw, y_raw = load_dataset()
    X, y = prepare_features(X_raw, y_raw)

    # Train/val/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full
    )
    print(f"\nTrain: {len(X_train):,} rows | Val: {len(X_val):,} rows | Test: {len(X_test):,} rows")

    # Build and train all candidates
    candidates = build_candidates(X)
    fitted_candidates = {}
    results = []

    print("\nTraining candidates...")
    for name, pipeline in candidates.items():
        print(f"  Training: {name}...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        fitted_candidates[name] = pipeline
        row = evaluate_model(name, pipeline, X_val, y_val)
        results.append(row)
        print(f"AP={row.get('average_precision', float('nan')):.3f}")

    # Sort by average precision
    results_df = pd.DataFrame(results).sort_values(
        ["average_precision", "recall_failure", "f1_failure"],
        ascending=False, na_position="last"
    ).reset_index(drop=True)

    print("\nValidation results:")
    print(results_df[["model", "average_precision", "roc_auc", "f1_failure"]].to_string(index=False))

    # Select best model
    eligible = results_df.dropna(subset=["average_precision"])
    best_name = eligible.iloc[0]["model"]
    best_model = fitted_candidates[best_name]
    print(f"\nBest model: {best_name}")

    # Find best threshold
    best_threshold = find_best_threshold(best_model, X_val, y_val)
    print(f"Best threshold: {best_threshold}")

    # Retrain best model on full train+val data
    print(f"\nRetraining {best_name} on full train+val data...")
    best_candidates = build_candidates(X)
    final_model = best_candidates[best_name]
    final_model.fit(X_train_full, y_train_full)

    # Test evaluation
    if hasattr(final_model, "predict_proba"):
        test_prob = final_model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= best_threshold).astype(int)
        test_ap = average_precision_score(y_test, test_prob)
        test_roc = roc_auc_score(y_test, test_prob)
    else:
        test_pred = final_model.predict(X_test)
        test_ap = np.nan
        test_roc = np.nan

    test_metrics = {
        "precision_failure": precision_score(y_test, test_pred, zero_division=0),
        "recall_failure": recall_score(y_test, test_pred, zero_division=0),
        "f1_failure": f1_score(y_test, test_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, test_pred),
        "average_precision": test_ap,
        "roc_auc": test_roc,
    }
    print(f"Test metrics: {json.dumps({k: round(v, 4) for k, v in test_metrics.items() if not np.isnan(v)}, indent=2)}")

    # Save all individual models
    print("\nSaving models to artifacts/...")
    save_meta = {}
    for name, model in fitted_candidates.items():
        safe_name = name.lower().replace(" ", "_")
        path = ARTIFACTS_DIR / f"nb_{safe_name}.joblib"
        joblib.dump(model, path)
        threshold = find_best_threshold(model, X_val, y_val)
        metrics = evaluate_model(name, model, X_val, y_val)
        save_meta[name] = {
            "artifact": str(path),
            "threshold": threshold,
            "val_metrics": {k: (round(v, 4) if not np.isnan(v) else None) for k, v in metrics.items() if k != "model"},
        }
        print(f"  Saved: {path.name}")

    # Save the best (final) model
    best_path = ARTIFACTS_DIR / "nb_best_model.joblib"
    bundle = {
        "model": final_model,
        "best_model_name": best_name,
        "threshold": best_threshold,
        "feature_columns": X.columns.tolist(),
        "test_metrics": {k: (round(v, 4) if not np.isnan(v) else None) for k, v in test_metrics.items()},
        "all_models_meta": save_meta,
    }
    joblib.dump(bundle, best_path)
    print(f"  Saved best model bundle: {best_path.name}")

    # Save metadata JSON
    meta_path = ARTIFACTS_DIR / "nb_models_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "best_model_name": best_name,
            "threshold": best_threshold,
            "feature_columns": X.columns.tolist(),
            "test_metrics": {k: (round(v, 4) if not np.isnan(v) else None) for k, v in test_metrics.items()},
            "all_models": save_meta,
        }, f, indent=2)
    print(f"  Saved metadata: {meta_path.name}")

    print("\n✅ All notebook models trained and saved!")
    return bundle


if __name__ == "__main__":
    train_all()

