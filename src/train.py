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
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
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


def build_pipeline() -> TransformedTargetRegressor:
    """Create the preprocessing + Histogram Gradient Boosting pipeline.

    The categorical columns are one-hot encoded (unknown categories — e.g. a
    city not in the training set — are safely ignored, so the model falls back
    to the country + city_tier signal). The whole regressor is wrapped in a
    ``TransformedTargetRegressor`` that trains on ``log1p(price)`` and inverts
    with ``expm1`` at predict time, which handles the huge price range across
    countries far better than a raw target.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", config.NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingRegressor(**config.HGB_PARAMS)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1
    )


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
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # 5-fold cross-validated R^2 for a more robust estimate
    cv_r2 = cross_val_score(
        pipeline, X, y, cv=5, scoring="r2", n_jobs=-1
    )

    from . import geography

    metrics = {
        "rmse": round(float(rmse), 2),
        "mae": round(float(mae), 2),
        "mape": round(float(mape), 4),
        "r2": round(float(r2), 4),
        "cv_r2_mean": round(float(np.mean(cv_r2)), 4),
        "cv_r2_std": round(float(np.std(cv_r2)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_countries": geography.n_countries(),
        "n_cities": geography.n_cities(),
        "model": "HistGradientBoostingRegressor (log-target)",
    }

    print("\n=== Evaluation on hold-out test set ===")
    print(f"RMSE      : {metrics['rmse']:,.0f}")
    print(f"MAE       : {metrics['mae']:,.0f}")
    print(f"MAPE      : {metrics['mape'] * 100:.2f}%")
    print(f"R^2       : {metrics['r2']:.4f}")
    print(f"CV R^2    : {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']:.4f})")
    print(f"Coverage  : {metrics['n_countries']} countries, "
          f"{metrics['n_cities']} cities")

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
