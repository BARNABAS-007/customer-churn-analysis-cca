import streamlit as st
import pandas as pd
import sklearn
import joblib

# Load model and scaler
model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction System")

st.write("Enter customer details to predict churn risk")

# User Inputs
tenure = st.number_input("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0)
total_charges = st.number_input("Total Charges", 0.0)
contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

# Convert categorical values
contract_map = {"Month-to-month":0,"One year":1,"Two year":2}
internet_map = {"DSL":0,"Fiber optic":1,"No":2}

contract = contract_map[contract]
internet_service = internet_map[internet_service]

# Create dataframe
input_data = pd.DataFrame({
    "tenure":[tenure],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges],
    "Contract":[contract],
    "InternetService":[internet_service]
})

# Scale data
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Churn"):

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Customer likely to CHURN")

        st.subheader("Recommended Retention Strategy")

        st.write("""
        • Offer loyalty discounts  
        • Provide better customer support  
        • Upgrade internet plan offers  
        • Provide long-term contract benefits
        """)

    else:

        st.success("✅ Customer likely to STAY")
