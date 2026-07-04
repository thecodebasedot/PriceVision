"""Generate a realistic synthetic housing dataset for PriceVision.

Real house-price datasets cannot be redistributed with the repo, so this
module builds a reproducible synthetic dataset whose prices follow sensible,
non-linear relationships with the features. The relationships are intentionally
non-linear and include interactions so that a Gradient Boosting model has
something meaningful to learn.

Run:
    python -m src.generate_data --rows 5000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config


def generate(n_rows: int = 5000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Create a synthetic housing DataFrame with a realistic price column."""
    rng = np.random.default_rng(seed)

    # --- raw features -----------------------------------------------------
    area = rng.normal(1800, 650, n_rows).clip(350, 6000).round(0)
    bedrooms = rng.integers(1, 6, n_rows)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n_rows), 1, 4)
    stories = rng.integers(1, 5, n_rows)
    parking = rng.integers(0, 4, n_rows)
    age = rng.integers(0, 60, n_rows)

    country = rng.choice(config.COUNTRY_LEVELS, n_rows)
    location = rng.choice(
        config.LOCATION_LEVELS, n_rows, p=[0.15, 0.40, 0.30, 0.15]
    )
    mainroad = rng.choice(config.YESNO_LEVELS, n_rows, p=[0.7, 0.3])
    furnishingstatus = rng.choice(
        config.FURNISHING_LEVELS, n_rows, p=[0.3, 0.4, 0.3]
    )

    # --- price model (ground truth we want the ML model to recover) -------
    location_mult = np.select(
        [location == "prime", location == "urban",
         location == "suburban", location == "rural"],
        [1.9, 1.35, 1.0, 0.7],
    )
    furnish_bonus = np.select(
        [furnishingstatus == "furnished",
         furnishingstatus == "semi-furnished",
         furnishingstatus == "unfurnished"],
        [9000.0, 4000.0, 0.0],
    )

    # per-country market factor (prices reported in USD)
    country_factor = np.array(
        [config.COUNTRY_PRICE_FACTOR[c] for c in country]
    )

    base = 15000.0
    price = (
        base
        + area * 30 * location_mult            # area matters more in good areas
        + bedrooms * 6000
        + bathrooms * 4000
        + stories * 3000
        + parking * 2500
        + furnish_bonus
        + np.where(mainroad == "yes", 6000.0, 0.0)
    )
    # depreciation with age (non-linear) and a mild premium for brand-new
    price *= 1.0 - (age / 120.0)
    price += np.where(age <= 2, 5000.0, 0.0)

    # scale by country market level
    price *= country_factor

    # multiplicative noise so the target is not perfectly learnable
    price *= rng.normal(1.0, 0.08, n_rows).clip(0.7, 1.35)
    price = price.clip(8000, None).round(-2)  # floor + round to nearest 100

    df = pd.DataFrame(
        {
            "area": area.astype(int),
            "bedrooms": bedrooms.astype(int),
            "bathrooms": bathrooms.astype(int),
            "stories": stories.astype(int),
            "parking": parking.astype(int),
            "age": age.astype(int),
            "country": country,
            "location": location,
            "mainroad": mainroad,
            "furnishingstatus": furnishingstatus,
            "price": price.astype(int),
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PriceVision dataset")
    parser.add_argument("--rows", type=int, default=5000, help="number of rows")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(config.DATA_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {config.DATA_PATH}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
