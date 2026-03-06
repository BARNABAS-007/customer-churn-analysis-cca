import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Customer Churn AI Dashboard", layout="wide")

st.title("📊 Customer Churn Analysis & Retention AI System")

# ------------------------------
# LOAD MODEL FILES
# ------------------------------

model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------
# USER INPUT
# ------------------------------

st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)",0,72,12)
monthly_charges = st.sidebar.slider("Monthly Charges",0,150,70)
total_charges = st.sidebar.slider("Total Charges",0,10000,2000)
contract = st.sidebar.selectbox("Contract Type",[0,1,2])
internet = st.sidebar.selectbox("Internet Service",[0,1,2])
tech_support = st.sidebar.selectbox("Tech Support",[0,1])

# ------------------------------
# CREATE INPUT DATAFRAME
# ------------------------------

input_dict = {
    "tenure":[tenure],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges],
    "Contract":[contract],
    "InternetService":[internet],
    "TechSupport":[tech_support]
}

X = pd.DataFrame(input_dict)

# ------------------------------
# SCALE INPUT
# ------------------------------

try:
    X_scaled = scaler.transform(X)
except:
    st.error("⚠ Feature mismatch between scaler and input. Check training features.")
    st.stop()

# ------------------------------
# PREDICTION
# ------------------------------

prediction = model.predict(X_scaled)[0]
probability = model.predict_proba(X_scaled)[0][1]

st.subheader("🔮 Churn Prediction")

if prediction == 1:
    st.error(f"Customer likely to churn (Probability {probability:.2f})")
else:
    st.success(f"Customer likely to stay (Probability {probability:.2f})")

# ------------------------------
# FEATURE IMPORTANCE
# ------------------------------

st.subheader("📊 Feature Importance")

try:
    importance = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature":X.columns,
        "Importance":importance[:len(X.columns)]
    })

    fig = px.bar(feat_df,
                 x="Importance",
                 y="Feature",
                 orientation="h")

    st.plotly_chart(fig,use_container_width=True)

except:
    st.info("Feature importance not available for this model")

# ------------------------------
# CUSTOMER SEGMENTATION
# ------------------------------

st.subheader("👥 Customer Segmentation")

if monthly_charges > 80:
    segment = "High Value Customer"
elif tenure < 12:
    segment = "New Customer"
else:
    segment = "Regular Customer"

st.info(f"Segment: {segment}")

# ------------------------------
# AI CHATBOT
# ------------------------------

st.subheader("🤖 Churn Explanation Bot")

user_question = st.text_input("Ask why customer might churn")

if user_question:

    if prediction == 1:
        response = """
Customer may churn because:

• High monthly charges  
• Short tenure  
• Lack of technical support  
• Contract type risk
"""
    else:
        response = """
Customer retention looks strong because:

• Long tenure  
• Stable contract  
• Balanced charges
"""

    st.write(response)

# ------------------------------
# EXECUTIVE DASHBOARD
# ------------------------------

st.subheader("📊 Executive Business Dashboard")

col1,col2,col3 = st.columns(3)

col1.metric("Churn Probability",round(probability,2))
col2.metric("Customer Tenure",tenure)
col3.metric("Monthly Revenue",monthly_charges)

st.success("✅ AI Dashboard Running Successfully")
