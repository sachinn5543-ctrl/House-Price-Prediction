import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# =========================
# BACKGROUND COLOR
# =========================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD DATASET & MODEL
# =========================

df = pd.read_csv("app/Housing (2).csv")

model = joblib.load("app/house_price_model.pkl")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🏠 House Price Prediction")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    ### 📌 About Project

    This Machine Learning app predicts
    house prices based on:

    ✅ Area  
    ✅ Bedrooms  
    ✅ Bathrooms  
    ✅ Floors  
    ✅ Parking  

    ---

    ### 🛠 Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-Learn
    - Streamlit
    - Matplotlib
    - Seaborn

    ---

    ### 👨‍💻 Developed By

    Sachin
    """
)

# =========================
# MAIN TITLE
# =========================

st.title("🏡 House Price Prediction System")

st.write(
    "Enter house details below to predict estimated house price."
)

# =========================
# METRICS SECTION
# =========================

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

# =========================
# DATASET PREVIEW
# =========================

st.subheader("📊 Housing Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# =========================
# PRICE DISTRIBUTION
# =========================

st.subheader("💰 Price Distribution")

st.line_chart(df["price"])

# =========================
# AREA VS PRICE
# =========================

st.subheader("🏠 Area vs Price")

chart_data = df[["area", "price"]]

st.line_chart(chart_data)

# =========================
# CORRELATION TABLE
# =========================

st.subheader("📈 Correlation Table")

correlation = df.corr(numeric_only=True)

st.dataframe(
    correlation,
    use_container_width=True
)

# =========================
# HEATMAP
# =========================

st.subheader("🌈 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))

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

st.subheader("🏠 Enter House Details")

area = st.number_input(
    "Area (sqft)",
    min_value=0,
    value=1000
)

bedrooms = st.slider(
    "Bedrooms",
    1,
    10,
    2
)

bathrooms = st.slider(
    "Bathrooms",
    1,
    10,
    1
)

floors = st.slider(
    "Floors",
    1,
    5,
    1
)

parking = st.slider(
    "Parking",
    0,
    5,
    1
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

    st.markdown("---")

    st.subheader("📢 Prediction Result")

    st.success(
        f"🏠 Estimated House Price: ₹ {prediction[0]:,.2f}"
    )

    st.info(
        "Prediction generated using Linear Regression Model"
    )

    st.balloons()

# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    """
    <center>
    Developed with ❤️ using Streamlit & Machine Learning
    </center>
    """,
    unsafe_allow_html=True
)