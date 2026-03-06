import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction & Retention Recommendation System")

st.write("Enter customer details to predict churn risk")

# User Inputs
tenure = st.number_input("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0)
total_charges = st.number_input("Total Charges", 0.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# Convert categorical values
contract_map = {
    "Month-to-month":0,
    "One year":1,
    "Two year":2
}

internet_map = {
    "DSL":0,
    "Fiber optic":1,
    "No":2
}

contract_value = contract_map[contract]
internet_value = internet_map[internet_service]

# Create dataframe
input_data = pd.DataFrame({
    "tenure":[tenure],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges],
    "Contract":[contract_value],
    "InternetService":[internet_value]
})

# Fix feature mismatch
input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale data
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Churn"):

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    churn_prob = probability[0][1] * 100
    stay_prob = probability[0][0] * 100

    st.subheader("Prediction Result")

    st.write(f"Churn Probability: {churn_prob:.2f}%")
    st.write(f"Stay Probability: {stay_prob:.2f}%")

    # Risk meter
    st.progress(int(churn_prob))

    # Risk level
    if churn_prob > 70:
        st.error("High Risk Customer")
    elif churn_prob > 40:
        st.warning("Medium Risk Customer")
    else:
        st.success("Low Risk Customer")

    # Prediction result
    if prediction[0] == 1:

        st.error("⚠️ Customer likely to CHURN")

        st.subheader("Recommended Retention Strategy")

        if contract_value == 0:
            st.write("• Offer discount for long-term contract")

        if internet_value == 1:
            st.write("• Provide fiber internet upgrade benefits")

        if monthly_charges > 80:
            st.write("• Offer loyalty discount")

        st.write("• Provide priority customer support")

    else:

        st.success("✅ Customer likely to STAY")

        st.write("Customer satisfaction is good. Maintain service quality.")
