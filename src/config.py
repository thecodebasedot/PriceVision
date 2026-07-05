"""Central configuration for the PriceVision project.

Keeps file paths, column definitions and model hyper-parameters in one place
so that data generation, training, prediction and the web app all stay in
sync.
"""
from pathlib import Path

from . import geography

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
    "city_tier",    # 1 (megacity) … 5 (small town); derived from the city
]

CATEGORICAL_FEATURES = [
    "country",        # country the property is in
    "city",           # city within the country (high cardinality)
    "location",       # neighbourhood tier
    "mainroad",       # attached to a main road (yes/no)
    "furnishingstatus",  # furnished / semi-furnished / unfurnished
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed categorical values (used for validation and the web UI)
LOCATION_LEVELS = ["prime", "urban", "suburban", "rural"]
YESNO_LEVELS = ["yes", "no"]
FURNISHING_LEVELS = ["furnished", "semi-furnished", "unfurnished"]

# Countries & cities live in geography.py. Prices are reported in USD so the
# model is comparable across every country.
COUNTRY_LEVELS = geography.COUNTRY_LEVELS

# ---------------------------------------------------------------------------
# Model hyper-parameters (Histogram-based Gradient Boosting)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# HistGradientBoostingRegressor: faster and stronger than the classic
# GradientBoostingRegressor, scales to large data and many categories, and
# supports built-in early stopping. The target is log-transformed at train
# time (see train.py) because prices span several orders of magnitude across
# countries.
HGB_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 900,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 25,
    "l2_regularization": 0.15,
    "max_bins": 255,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 40,
    "random_state": RANDOM_STATE,
}
