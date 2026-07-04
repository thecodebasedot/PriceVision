"""PriceVision — house price prediction web app.

A small Streamlit UI on top of the trained Gradient Boosting model.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json

import streamlit as st

from src import config
from src.predict import load_model, predict_price

st.set_page_config(page_title="PriceVision", page_icon="🏠", layout="centered")

st.title("🏠 PriceVision")
st.caption("House price prediction powered by Gradient Boosting")


@st.cache_resource
def _get_model():
    return load_model()


# Fail gracefully if the model has not been trained yet.
try:
    _get_model()
    model_ready = True
except FileNotFoundError as exc:
    model_ready = False
    st.error(str(exc))

with st.form("prediction_form"):
    st.subheader("Property details")

    country = st.selectbox("Country", config.COUNTRY_LEVELS, index=0)

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (sq ft)", 300, 8000, 1800, step=50)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 2)
        stories = st.slider("Stories", 1, 4, 2)
    with col2:
        parking = st.slider("Parking spots", 0, 4, 1)
        age = st.slider("Age (years)", 0, 60, 5)
        location = st.selectbox("Location", config.LOCATION_LEVELS, index=1)
        furnishingstatus = st.selectbox(
            "Furnishing", config.FURNISHING_LEVELS, index=1
        )

    mainroad = st.radio("On a main road?", config.YESNO_LEVELS, horizontal=True)

    submitted = st.form_submit_button("Predict price", use_container_width=True)

if submitted and model_ready:
    features = {
        "country": country,
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "age": age,
        "location": location,
        "mainroad": mainroad,
        "furnishingstatus": furnishingstatus,
    }
    price = predict_price(features)
    st.success(f"### Estimated price: ${price:,.0f} USD")

# Show model quality if metrics are available.
if config.METRICS_PATH.exists():
    metrics = json.loads(config.METRICS_PATH.read_text())
    with st.expander("Model performance"):
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{metrics['r2']:.3f}")
        c2.metric("RMSE", f"{metrics['rmse']:,.0f}")
        c3.metric("MAE", f"{metrics['mae']:,.0f}")
        st.caption(
            f"5-fold CV R² = {metrics['cv_r2_mean']:.3f} "
            f"± {metrics['cv_r2_std']:.3f} · "
            f"trained on {metrics['n_train']:,} rows."
        )
