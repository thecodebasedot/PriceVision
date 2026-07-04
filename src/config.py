"""Central configuration for the PriceVision project.

Keeps file paths, column definitions and model hyper-parameters in one place
so that data generation, training, prediction and the web app all stay in
sync.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

DATA_PATH = DATA_DIR / "houses.csv"
MODEL_PATH = MODEL_DIR / "pricevision_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
TARGET = "price"

NUMERIC_FEATURES = [
    "area",         # living area in square feet
    "bedrooms",     # number of bedrooms
    "bathrooms",    # number of bathrooms
    "stories",      # number of floors
    "parking",      # number of parking spots
    "age",          # age of the property in years
]

CATEGORICAL_FEATURES = [
    "country",        # country the property is in
    "location",       # neighbourhood tier
    "mainroad",       # attached to a main road (yes/no)
    "furnishingstatus",  # furnished / semi-furnished / unfurnished
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed categorical values (used for validation and the web UI)
LOCATION_LEVELS = ["prime", "urban", "suburban", "rural"]
YESNO_LEVELS = ["yes", "no"]
FURNISHING_LEVELS = ["furnished", "semi-furnished", "unfurnished"]

# Supported countries and their relative price levels (rough market factor,
# normalised so that "Bangladesh" ~ 1.0). Prices are reported in USD so the
# model is comparable across countries.
COUNTRY_PRICE_FACTOR = {
    "Bangladesh": 1.0,
    "India": 1.1,
    "Pakistan": 0.9,
    "UAE": 3.2,
    "UK": 5.5,
    "USA": 6.0,
    "Canada": 4.8,
    "Australia": 5.2,
}
COUNTRY_LEVELS = list(COUNTRY_PRICE_FACTOR.keys())

# ---------------------------------------------------------------------------
# Model hyper-parameters (Gradient Boosting)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

GB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.9,
    "min_samples_leaf": 15,
    "random_state": RANDOM_STATE,
}
