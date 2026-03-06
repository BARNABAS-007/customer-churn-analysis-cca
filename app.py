import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.cluster import KMeans

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn AI System", layout="wide")

st.title("📊 Customer Churn Analysis & Retention AI System")

st.write("AI powered churn prediction and retention recommendation system")

# -----------------------------
# Get training features
# -----------------------------
features = scaler.feature_names_in_

st.sidebar.header("Customer Inputs")

input_data = {}

for feature in features:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

df = pd.DataFrame([input_data])

# -----------------------------
# Scale Data
# -----------------------------
X_scaled = scaler.transform(df)

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict(X_scaled)
prob = model.predict_proba(X_scaled)

churn_prob = prob[0][1] * 100
stay_prob = prob[0][0] * 100

# -----------------------------
# Dashboard Layout
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Churn Probability", f"{churn_prob:.2f}%")
col2.metric("Stay Probability", f"{stay_prob:.2f}%")
col3.metric("Prediction", "Churn" if prediction[0] == 1 else "Stay")

st.progress(int(churn_prob))

# -----------------------------
# Risk Level
# -----------------------------
if churn_prob > 70:
    st.error("🔴 High Risk Customer")
elif churn_prob > 40:
    st.warning("🟡 Medium Risk Customer")
else:
    st.success("🟢 Low Risk Customer")

# -----------------------------
# Retention Strategy
# -----------------------------
st.subheader("🎯 Recommended Retention Strategy")

if prediction[0] == 1:

    strategies = []

    if "MonthlyCharges" in df.columns and df["MonthlyCharges"][0] > 80:
        strategies.append("Offer loyalty discount")

    if "tenure" in df.columns and df["tenure"][0] < 12:
        strategies.append("Provide onboarding support")

    strategies.append("Provide premium customer support")
    strategies.append("Offer long-term contract benefits")

    for s in strategies:
        st.write("•", s)

else:
    st.write("Customer likely to stay. Maintain service quality.")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📈 Feature Importance")

if hasattr(model, "feature_importances_"):

    importance = model.feature_importances_

    fig, ax = plt.subplots()

    ax.barh(features, importance)
    ax.set_title("Feature Importance")

    st.pyplot(fig)

# -----------------------------
# SHAP Explainable AI
# -----------------------------
st.subheader("🤖 SHAP Explainable AI")

try:

    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)

    fig2 = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)

    st.pyplot(fig2)

except:
    st.write("SHAP visualization not supported for this model")

# -----------------------------
# Customer Segmentation
# -----------------------------
st.subheader("👥 Customer Segmentation")

try:

    kmeans = KMeans(n_clusters=3, random_state=42)

    cluster = kmeans.fit_predict(X_scaled)

    st.write("Customer Segment:", int(cluster[0]))

except:
    st.write("Segmentation unavailable")

# -----------------------------
# Retention Report Download
# -----------------------------
st.subheader("📄 Download Retention Report")

report = f"""
Customer Churn Analysis Report

Churn Probability: {churn_prob:.2f}%
Stay Probability: {stay_prob:.2f}%

Prediction:
{"Customer likely to churn" if prediction[0]==1 else "Customer likely to stay"}

Recommended Strategy:
Improve service quality
Offer discounts
Provide customer support
"""

st.download_button(
    label="Download Report",
    data=report,
    file_name="churn_report.txt"
)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("AI Powered Customer Retention System")
