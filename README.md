# 🏠 PriceVision

**Predict house prices with a Gradient Boosting model.**

PriceVision is an end-to-end machine-learning project that estimates the price
of a house from its features (area, bedrooms, bathrooms, location, age, …). It
covers the full pipeline: dataset generation → model training & evaluation →
command-line prediction → an interactive web app.

| | |
|---|---|
| **Problem** | House price prediction (বাড়ির দাম পূর্বাভাস) |
| **Algorithm** | Gradient Boosting (`GradientBoostingRegressor`) |
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
| `country` | categorical | `Bangladesh` / `India` / `Pakistan` / `UAE` / `UK` / `USA` / `Canada` / `Australia` |
| `location` | categorical | `prime` / `urban` / `suburban` / `rural` |
| `mainroad` | categorical | `yes` / `no` |
| `furnishingstatus` | categorical | `furnished` / `semi-furnished` / `unfurnished` |
| `price` | **target** | sale price **in USD** (comparable across countries) |

> **Multi-country support:** prices are modelled in USD and scaled by a
> per-country market factor, so the same house is valued differently in, say,
> Bangladesh vs. the USA. Add or tune countries in `COUNTRY_PRICE_FACTOR`
> (`src/config.py`).

---

## The model

A scikit-learn `Pipeline` that one-hot encodes the categorical columns and
feeds everything into a `GradientBoostingRegressor`. Gradient Boosting builds an
ensemble of shallow decision trees, where each new tree corrects the errors of
the ones before it — a great fit for the non-linear, interaction-heavy nature of
housing data.

Evaluation on a held-out test set (metrics saved to `models/metrics.json`):

- **R²** ≈ 0.93
- 5-fold cross-validated R² ≈ 0.93

Tune the hyper-parameters in `src/config.py` (`GB_PARAMS`).

---

## License

See [LICENSE](LICENSE).
