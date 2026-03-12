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

# Title and description
st.title("📦 Product Demand Prediction")
st.markdown("---")

st.info("Enter product sales details to estimate expected demand.")

st.write("""
This app predicts **product demand (Quantity)** based on:
- Sales amount
- Discount offered
- Profit margin

Simply enter the values below and click **Predict**!
""")

st.markdown("---")

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model


try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model not found! Please run the training script first to create model.pkl.")


# Input section
if model_loaded:

    st.subheader("Enter Product Details:")

    col1, col2 = st.columns(2)

    with col1:
        sales = st.number_input(
            "Sales ($)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help="Total sales amount for the product"
        )

        discount = st.number_input(
            "Discount (0 - 1)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Discount as a decimal (e.g., 0.2 = 20%)"
        )

    with col2:
        profit = st.number_input(
            "Profit ($)",
            min_value=-1000.0,
            max_value=5000.0,
            value=10.0,
            step=5.0,
            help="Profit margin for the product"
        )

    st.markdown("---")

    # Prediction button
    if st.button("🔮 Predict Demand", type="primary", use_container_width=True):

        # Prepare input features (same as training)
        # [Sales, Discount, Profit]
        input_features = np.array([[sales, discount, profit]])

        # Make prediction
        prediction = model.predict(input_features)[0]

        st.markdown("---")
        st.success("✅ Prediction Complete!")

        predicted_quantity = max(1, round(prediction))

        # Display result
        st.markdown(f"""
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #1e90ff; margin: 0;">Predicted Demand</h3>
            <h1 style="color: #333; margin: 10px 0; font-size: 48px;">{predicted_quantity}</h1>
            <p style="color: #666; margin: 0;">units</p>
        </div>
        """, unsafe_allow_html=True)

        # Input summary
        st.markdown("---")
        st.subheader("📊 Input Summary:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Sales", f"${sales:.2f}")

        with col2:
            st.metric("Discount", f"{discount*100:.0f}%")

        with col3:
            st.metric("Profit", f"${profit:.2f}")


# Footer
st.markdown("---")
st.caption("Built with ❤️ using Python, scikit-learn, and Streamlit")
