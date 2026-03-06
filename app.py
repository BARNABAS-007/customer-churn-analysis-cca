import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import shap

# -----------------------------
# Load Model & Scaler
# -----------------------------

model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Customer Churn Analysis & Retention AI System")

# -----------------------------
# Upload Dataset
# -----------------------------

uploaded_file = st.file_uploader("Upload Customer Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Select Features
    # -----------------------------

    features = scaler.feature_names_in_

    X = df[features]

    # Scale data
    X_scaled = scaler.transform(X)

    # -----------------------------
    # Predict Churn
    # -----------------------------

    df["Churn Prediction"] = model.predict(X_scaled)

    df["Churn Probability"] = model.predict_proba(X_scaled)[:,1]

    st.subheader("Prediction Results")
    st.dataframe(df.head())

    # -----------------------------
    # Churn Probability Dashboard
    # -----------------------------

    st.subheader("📊 Churn Probability Distribution")

    fig, ax = plt.subplots()
    ax.hist(df["Churn Probability"], bins=20)
    st.pyplot(fig)

    # -----------------------------
    # Customer Segmentation
    # -----------------------------

    st.subheader("👥 Customer Segmentation")

    kmeans = KMeans(n_clusters=3)
    df["Segment"] = kmeans.fit_predict(X_scaled)

    fig2, ax2 = plt.subplots()

    ax2.scatter(
        df[features[0]],
        df[features[1]],
        c=df["Segment"]
    )

    ax2.set_xlabel(features[0])
    ax2.set_ylabel(features[1])

    st.pyplot(fig2)

    st.write("Segment 0 = Loyal Customers")
    st.write("Segment 1 = At Risk")
    st.write("Segment 2 = New Customers")

    # -----------------------------
    # Feature Importance
    # -----------------------------

    st.subheader("📈 Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature":features,
            "Importance":importance
        }).sort_values("Importance", ascending=False)

        fig3, ax3 = plt.subplots()

        ax3.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        st.pyplot(fig3)

    # -----------------------------
    # SHAP Explainable AI
    # -----------------------------

    st.subheader("🤖 AI Explanation (SHAP)")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_scaled)

    fig4 = plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    st.pyplot(fig4)

    # -----------------------------
    # Retention Strategy Generator
    # -----------------------------

    st.subheader("💡 Retention Strategy Recommendations")

    high_risk = df[df["Churn Probability"] > 0.7]

    st.write("High Risk Customers:", len(high_risk))

    st.write("""
    Recommended Actions:

    • Offer loyalty discounts  
    • Provide long-term contract benefits  
    • Improve customer support  
    • Offer bundled service plans
    """)

    # -----------------------------
    # Download Report
    # -----------------------------

    st.subheader("⬇ Download Report")

    csv = df.to_csv(index=False)

    st.download_button(
        "Download Prediction Report",
        csv,
        "churn_report.csv",
        "text/csv"
    )
