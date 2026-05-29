import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load model
model = joblib.load("app/house_price_model.pkl")

# Load dataset
df = pd.read_csv("app/housing.csv")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🏠 House Price Prediction")

st.sidebar.info(
    """
    This Machine Learning app predicts
    house prices based on:

    ✅ Area  
    ✅ Bedrooms  
    ✅ Bathrooms  
    ✅ floors  
    ✅ Parking

    Built using:
    - Python
    - Pandas
    - Scikit-Learn
    - Streamlit
    """
)

# =========================
# MAIN TITLE
# =========================

st.title("🏡 House Price Prediction System")

st.write(
    "Enter house details below to predict estimated house price."
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Houses", len(df))

col2.metric(
    "Average Price",
    f"₹ {int(df['price'].mean()):,}"
)

col3.metric(
    "Max Price",
    f"₹ {int(df['price'].max()):,}"
)

st.subheader("📊 Housing Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.subheader("💰 Price Distribution")

st.line_chart(df["price"])

st.subheader("🏠 Area vs Price")

chart_data = df[["area", "price"]]

st.line_chart(chart_data)

st.subheader("📈 Correlation Table")

correlation = df.corr(numeric_only=True)

st.dataframe(correlation)

st.subheader("🌈 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# =========================
# USER INPUTS
# =========================

area = st.number_input(
    "Area (sqft)",
    min_value=0,
    value=1000
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    value=2
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    value=1
)

floors = st.number_input(
    "floors",
    min_value=1,
    value=1
)

parking = st.number_input(
    "Parking",
    min_value=0,
    value=1
)

# =========================
# PREDICTION
# =========================

if st.button("🔍 Predict House Price"):

    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'flor': [floors],
        'parking': [parking]
    })

    prediction = model.predict(input_data)

    st.success(
        f"🏠 Estimated House Price: ₹ {prediction[0]:,.2f}"
    )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Machine Learning House Price Prediction Project"
)