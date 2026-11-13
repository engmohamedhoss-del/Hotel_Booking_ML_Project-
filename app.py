import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Hotel Booking Prediction",
    page_icon="🏨",
    layout="wide"
)

model = joblib.load("hotel_cancellation_model.pkl")
app_data = joblib.load("hotel_app_data.pkl")

st.title("Hotel Booking Cancellation Prediction")
st.write("Predict whether a hotel booking will be canceled or not.")

st.divider()

numeric_features = app_data["numeric_features"]
categorical_features = app_data["categorical_features"]

input_data = {}

st.subheader("Booking Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Numerical Features")

    for feature in numeric_features:
        input_data[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=float(app_data["medians"][feature])
        )

with col2:
    st.subheader("Categorical Features")

    for feature in categorical_features:
        categories = app_data["categories"][feature]

        input_data[feature] = st.selectbox(
            feature.replace("_", " ").title(),
            categories
        )

st.divider()

if st.button("Predict", use_container_width=True):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("This booking is likely to be canceled.")
    else:
        st.success("This booking is not likely to be canceled.")

    st.metric(
        "Cancellation Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))