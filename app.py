import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Churn Intelligence Platform", layout="wide")

# -------------------------------
# LOAD MODEL
# -------------------------------

model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler (1).pkl")

st.title("Customer Churn Analysis & Retention Intelligence System")

# -------------------------------
# DATASET UPLOAD
# -------------------------------

st.sidebar.header("Upload Dataset")

file = st.sidebar.file_uploader("Upload churn dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # DATA CLEANING
    # -------------------------------

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df["Contract"] = df["Contract"].map({
        "Month-to-month":0,
        "One year":1,
        "Two year":2
    })

    df["InternetService"] = df["InternetService"].map({
        "DSL":0,
        "Fiber optic":1,
        "No":2
    })

    features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService"
    ]

    X = df[features]

    # -------------------------------
    # SCALE DATA
    # -------------------------------

    X_scaled = scaler.transform(X)

    # -------------------------------
    # PREDICT CHURN
    # -------------------------------

    df["Prediction"] = model.predict(X_scaled)
    df["ChurnProbability"] = model.predict_proba(X_scaled)[:,1]*100

    # -------------------------------
    # EXECUTIVE DASHBOARD
    # -------------------------------

    st.header("Executive Business Dashboard")

    total_customers = len(df)
    churn_customers = df[df["Prediction"] == 1].shape[0]
    churn_rate = (churn_customers / total_customers) * 100
    revenue_risk = df[df["Prediction"] == 1]["MonthlyCharges"].sum()

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("Predicted Churn Customers", churn_customers)
    col3.metric("Churn Rate (%)", f"{churn_rate:.2f}")
    col4.metric("Revenue at Risk ($)", f"{revenue_risk:.2f}")

    # -------------------------------
    # INTERACTIVE FILTER
    # -------------------------------

    st.sidebar.header("Filters")

    risk = st.sidebar.slider("Churn Risk Threshold",0,100,50)

    high_risk = df[df["ChurnProbability"] > risk]

    st.subheader("High Risk Customers")

    st.dataframe(high_risk)

    # -------------------------------
    # CHURN DISTRIBUTION
    # -------------------------------

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Churn Probability Distribution")

        fig,ax = plt.subplots()

        sns.histplot(df["ChurnProbability"], bins=20, ax=ax)

        st.pyplot(fig)

    with col2:

        st.subheader("Monthly Charges vs Churn Risk")

        fig2,ax2 = plt.subplots()

        sns.scatterplot(
            x=df["MonthlyCharges"],
            y=df["ChurnProbability"],
            ax=ax2
        )

        st.pyplot(fig2)

    # -------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature":features,
        "Importance":importance
    })

    fig3,ax3 = plt.subplots()

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        ax=ax3
    )

    st.pyplot(fig3)

    # -------------------------------
    # SHAP AI EXPLANATION
    # -------------------------------

    st.subheader("AI Explanation (SHAP)")

    explainer = shap.Explainer(model)

    shap_values = explainer(X_scaled)

    fig4 = plt.figure()

    shap.summary_plot(shap_values, X, show=False)

    st.pyplot(fig4)

    # -------------------------------
    # CUSTOMER SEGMENTATION
    # -------------------------------

    st.subheader("Customer Segmentation")

    kmeans = KMeans(n_clusters=3, random_state=42)

    df["Segment"] = kmeans.fit_predict(X_scaled)

    fig5,ax5 = plt.subplots()

    sns.scatterplot(
        x=df["MonthlyCharges"],
        y=df["tenure"],
        hue=df["Segment"],
        palette="viridis",
        ax=ax5
    )

    st.pyplot(fig5)

    # -------------------------------
    # BUSINESS INSIGHTS
    # -------------------------------

    st.subheader("Churn vs Stay")

    fig6,ax6 = plt.subplots()

    sns.countplot(x="Prediction", data=df, ax=ax6)

    ax6.set_xticklabels(["Stay","Churn"])

    st.pyplot(fig6)

    # -------------------------------
    # RETENTION STRATEGY
    # -------------------------------

    st.subheader("Retention Strategy")

    st.write("""
    🔴 High Risk Customers
    • Offer loyalty discounts  
    • Encourage yearly contracts  
    • Improve customer support  

    🟡 Medium Risk
    • Offer bundled services  

    🟢 Low Risk
    • Maintain service quality
    """)

    # -------------------------------
    # AI CHATBOT
    # -------------------------------

    st.header("AI Churn Assistant")

    question = st.text_input("Ask something about churn")

    if question:

        if "why churn" in question.lower():

            st.write("""
Main churn reasons:

• High monthly charges
• Month-to-month contracts
• Low tenure customers
• Fiber internet dissatisfaction
""")

        elif "reduce churn" in question.lower():

            st.write("""
Recommended strategies:

• Offer contract discounts
• Improve customer support
• Provide bundle offers
• Upgrade internet services
""")

        elif "high risk" in question.lower():

            st.dataframe(high_risk.head())

        else:

            st.write("Ask about churn causes, strategies, or high risk customers.")
