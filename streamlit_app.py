"""
Product Demand Prediction - Streamlit App
"""

import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Product Demand Prediction",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Product Demand Prediction")
st.markdown("---")

st.info("Enter product sales details to estimate expected demand.")

st.write("""
This app predicts **product demand (Quantity)** based on:

• Sales amount  
• Discount offered  
• Profit margin  
""")

st.markdown("---")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("Model not found! Please train the model first.")


if model_loaded:

    st.subheader("Enter Product Details")

    col1, col2 = st.columns(2)

    with col1:

        sales = st.number_input(
            "Sales ($)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )

        discount = st.number_input(
            "Discount (0 - 1)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )

    with col2:

        profit = st.number_input(
            "Profit ($)",
            min_value=-1000.0,
            max_value=5000.0,
            value=10.0,
            step=5.0
        )

    st.markdown("---")

 if st.button("🔮 Predict Demand", type="primary", use_container_width=True):

    # Calculate profit_ratio (same as training)
    profit_ratio = profit / sales if sales != 0 else 0

    # Send 4 features to model
    input_features = np.array([[sales, discount, profit, profit_ratio]])

    prediction = model.predict(input_features)[0]

    predicted_quantity = max(1, round(prediction))

    st.success("Prediction Complete!")

    st.markdown(f"""
    <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;text-align:center">
    <h3>Predicted Demand</h3>
    <h1>{predicted_quantity}</h1>
    <p>Units</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")
st.caption("Built with ❤️ using Python, scikit-learn and Streamlit")
