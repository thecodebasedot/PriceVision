"""Train the PriceVision Gradient Boosting model.

Builds a scikit-learn Pipeline that one-hot encodes the categorical features
and fits a GradientBoostingRegressor. The fitted pipeline is persisted to
``models/pricevision_model.joblib`` together with evaluation metrics.

Run:
    python -m src.train
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config
from .generate_data import generate


def load_data() -> pd.DataFrame:
    """Load the dataset, generating it on first run if it does not exist."""
    if not config.DATA_PATH.exists():
        print("Dataset not found — generating a fresh one...")
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate().to_csv(config.DATA_PATH, index=False)
    return pd.read_csv(config.DATA_PATH)


def build_pipeline() -> Pipeline:
    """Create the preprocessing + Gradient Boosting pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", config.NUMERIC_FEATURES),
        ]
    )
    model = GradientBoostingRegressor(**config.GB_PARAMS)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train() -> Pipeline:
    df = load_data()
    X = df[config.FEATURES]
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    pipeline = build_pipeline()
    print("Training Gradient Boosting model...")
    pipeline.fit(X_train, y_train)

    # --- evaluation -------------------------------------------------------
    preds = pipeline.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # 5-fold cross-validated R^2 for a more robust estimate
    cv_r2 = cross_val_score(
        pipeline, X, y, cv=5, scoring="r2", n_jobs=-1
    )

    metrics = {
        "rmse": round(float(rmse), 2),
        "mae": round(float(mae), 2),
        "r2": round(float(r2), 4),
        "cv_r2_mean": round(float(np.mean(cv_r2)), 4),
        "cv_r2_std": round(float(np.std(cv_r2)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print("\n=== Evaluation on hold-out test set ===")
    print(f"RMSE      : {metrics['rmse']:,.0f}")
    print(f"MAE       : {metrics['mae']:,.0f}")
    print(f"R^2       : {metrics['r2']:.4f}")
    print(f"CV R^2    : {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']:.4f})")

    # --- persist ----------------------------------------------------------
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model to   {config.MODEL_PATH}")
    print(f"Saved metrics to {config.METRICS_PATH}")

    return pipeline


if __name__ == "__main__":
    train()
