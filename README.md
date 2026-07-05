# 🏠 PriceVision

**Predict house prices with a Gradient Boosting model.**

PriceVision is an end-to-end machine-learning project that estimates the price
of a house from its features (area, bedrooms, bathrooms, location, age, …). It
covers the full pipeline: dataset generation → model training & evaluation →
command-line prediction → an interactive web app.

| | |
|---|---|
| **Problem** | House price prediction (বাড়ির দাম পূর্বাভাস) |
| **Algorithm** | Gradient Boosting (`HistGradientBoostingRegressor`, log-target) |
| **Coverage** | 48 countries · 325+ cities worldwide (extensible to any city) |
| **Stack** | Python · scikit-learn · pandas · Streamlit |

---

## Project structure

```
PriceVision/
├── app.py                 # Streamlit web app
├── requirements.txt
├── data/
│   └── houses.csv         # dataset (generated, reproducible)
├── models/
│   ├── metrics.json       # saved evaluation metrics
│   └── pricevision_model.joblib   # trained model (git-ignored, rebuild locally)
└── src/
    ├── config.py          # paths, schema & hyper-parameters
    ├── geography.py       # world countries + cities + price tiers
    ├── generate_data.py   # synthetic dataset generator
    ├── train.py           # train + evaluate + save the model
    └── predict.py         # reusable prediction helper + CLI
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the dataset — 5000 rows by default
python -m src.generate_data --rows 5000

# 3. Train the Gradient Boosting model (auto-generates data if missing)
python -m src.train

# 4a. Predict from the command line
python -m src.predict \
    --area 2000 --bedrooms 3 --bathrooms 2 --stories 2 \
    --parking 1 --age 5 --country USA --location urban \
    --mainroad yes --furnishingstatus semi-furnished

# 4b. …or launch the web app
streamlit run app.py
```

---

## The dataset

Real house-price datasets can't be redistributed here, so
`src/generate_data.py` builds a **reproducible synthetic dataset** whose prices
follow realistic, non-linear relationships (location multipliers, age
depreciation, furnishing bonuses, feature interactions, plus noise). Each house
has:

| Feature | Type | Description |
|---|---|---|
| `area` | numeric | living area in square feet |
| `bedrooms` | numeric | number of bedrooms |
| `bathrooms` | numeric | number of bathrooms |
| `stories` | numeric | number of floors |
| `parking` | numeric | parking spots |
| `age` | numeric | age of the property in years |
| `city_tier` | numeric | 1 (megacity) … 5 (small town) — auto-derived from the city |
| `country` | categorical | 48 countries across every continent |
| `city` | categorical | 325+ major cities (any city works — see below) |
| `location` | categorical | `prime` / `urban` / `suburban` / `rural` |
| `mainroad` | categorical | `yes` / `no` |
| `furnishingstatus` | categorical | `furnished` / `semi-furnished` / `unfurnished` |
| `price` | **target** | sale price **in USD** (comparable across countries) |

### Worldwide coverage & "any city"

`src/geography.py` maps **48 countries** to **325+ major cities**, each tagged
with a price *tier* (1 = most expensive megacity … 5 = small town). A house
price is driven by `country_factor × tier_multiplier`, so the same house is
valued very differently in, say, Dhaka vs. New York.

Enumerating literally every settlement on Earth isn't feasible, so the model is
built to **generalise to any city**:

- the transferable signal lives in the numeric `city_tier` feature, and
- unknown city names are safely ignored by the one-hot encoder, so the model
  falls back to the `country` + `city_tier` signal.

To price a city that isn't in the table, just pass its name and (optionally) a
tier — or add it to `COUNTRIES` in `src/geography.py`. Adding a country/city is
a one-line edit.

---

## The model

A scikit-learn `Pipeline` that one-hot encodes the categorical columns and
feeds everything into a **`HistGradientBoostingRegressor`** — a faster, stronger
histogram-based Gradient Boosting that scales to large data and many categories
and supports built-in early stopping. Because prices span several orders of
magnitude across countries, the whole regressor is wrapped in a
`TransformedTargetRegressor` that trains on `log1p(price)` and inverts with
`expm1` at predict time.

Evaluation on a held-out test set (metrics saved to `models/metrics.json`):

- **R²** ≈ 0.97
- 5-fold cross-validated R² ≈ 0.97
- **MAPE** (mean absolute % error) reported per run

Tune the hyper-parameters in `src/config.py` (`HGB_PARAMS`).

---

## License

See [LICENSE](LICENSE).
