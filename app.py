import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("models/house_price_model.pkl")

# Load dataset (for dashboard)
df = pd.read_csv("data/kc_house_data.csv")

# Page config
st.set_page_config(page_title="House Price AI", layout="wide")

# Title
st.title("🏡 AI House Price Predictor")
st.write("Predict house prices + explore data insights")

# ---------------- INPUT SECTION ----------------
st.header("🔮 Predict House Price")

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.slider("Bedrooms", 1, 10, 3)

with col2:
    bathrooms = st.slider("Bathrooms", 1, 10, 2)

with col3:
    yr_built = st.slider("Year Built", 1900, 2025, 2000)

col4, col5 = st.columns(2)

with col4:
    sqft_living = st.number_input("Living Area (sqft)", 500, 10000, 1500)

with col5:
    sqft_lot = st.number_input("Lot Size (sqft)", 500, 50000, 5000)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, yr_built]])
    prediction = model.predict(input_data)
    st.success(f"🏠 Estimated Price: ${prediction[0]:,.2f}")

# ---------------- DASHBOARD SECTION ----------------
st.markdown("---")
st.header("📊 Data Insights Dashboard")

# 1. Price distribution
st.subheader("🏠 Price Distribution")
st.bar_chart(df["price"].head(100))

# 2. Correlation insights
st.subheader("📈 Feature Correlation")
corr = df[["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot"]].corr()
st.write(corr)

# 3. Average price by bedrooms
st.subheader("🏡 Avg Price by Bedrooms")
avg_price = df.groupby("bedrooms")["price"].mean()
st.line_chart(avg_price)