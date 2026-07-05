"""Predict house prices with a trained PriceVision model.

Provides a reusable ``predict_price`` helper (used by the web app) and a small
command-line interface.

Run:
    python -m src.predict --area 2000 --bedrooms 3 --bathrooms 2 \
        --stories 2 --parking 1 --age 5 \
        --location urban --mainroad yes --furnishingstatus semi-furnished
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import pandas as pd

from . import config, geography


@lru_cache(maxsize=1)
def load_model():
    """Load the persisted pipeline (cached across calls)."""
    import joblib

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. "
            "Train it first with:  python -m src.train"
        )
    return joblib.load(config.MODEL_PATH)


def predict_price(features: dict) -> float:
    """Predict a single house price from a feature dict.

    ``features`` must contain every key in ``config.FEATURES`` — except
    ``city_tier``, which is auto-derived from the (country, city) pair when
    absent. This lets callers price *any* city: known cities get their tier
    from the geography table, unknown ones fall back to the default tier.
    """
    features = dict(features)  # don't mutate the caller's dict
    if "city_tier" not in features:
        features["city_tier"] = geography.tier_of(
            features.get("country"), features.get("city")
        )

    missing = [f for f in config.FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing feature(s): {missing}")

    model = load_model()
    row = pd.DataFrame([{k: features[k] for k in config.FEATURES}])
    return float(model.predict(row)[0])


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict a house price")
    p.add_argument("--area", type=float, required=True)
    p.add_argument("--bedrooms", type=int, required=True)
    p.add_argument("--bathrooms", type=int, required=True)
    p.add_argument("--stories", type=int, required=True)
    p.add_argument("--parking", type=int, required=True)
    p.add_argument("--age", type=int, required=True)
    p.add_argument("--country", choices=config.COUNTRY_LEVELS, required=True)
    p.add_argument(
        "--city",
        default="",
        help="city name (any city; known cities use their price tier)",
    )
    p.add_argument("--location", choices=config.LOCATION_LEVELS, required=True)
    p.add_argument("--mainroad", choices=config.YESNO_LEVELS, required=True)
    p.add_argument(
        "--furnishingstatus", choices=config.FURNISHING_LEVELS, required=True
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    features = vars(args)
    price = predict_price(features)
    print(f"Estimated price: ${price:,.0f} USD")


if __name__ == "__main__":
    main()
