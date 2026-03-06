import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# ------------------------------------------------
# Page Config
# ------------------------------------------------

st.set_page_config(
    page_title="AI Customer Churn Dashboard",
    layout="wide"
)

st.title("📊 AI Customer Churn & Retention Dashboard")
st.write("Upload your dataset to predict churn and generate retention strategies.")

# ------------------------------------------------
# Load Model
# ------------------------------------------------

model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

features = scaler.feature_names_in_

# ------------------------------------------------
# Upload Dataset
# ------------------------------------------------

uploaded_file = st.file_uploader("Upload Customer Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------------------------
    # Prepare Data
    # ------------------------------------------------

    X = df[features]

    X_scaled = scaler.transform(X)

    # ------------------------------------------------
    # Predict Churn
    # ------------------------------------------------

    pred = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled)

    df["Churn Prediction"] = pred
    df["Churn Probability"] = prob[:,1] * 100

    # ------------------------------------------------
    # Customer Lifetime Value
    # ------------------------------------------------

    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["CLV"] = df["MonthlyCharges"] * df["tenure"] * (1 - prob[:,1])

    # ------------------------------------------------
    # Retention Strategy Generator
    # ------------------------------------------------

    def retention_strategy(row):

        strategies = []

        if row["Churn Probability"] > 70:
            strategies.append("Offer loyalty discount")

        if "tenure" in row and row["tenure"] < 12:
            strategies.append("Provide onboarding support")

        if "MonthlyCharges" in row and row["MonthlyCharges"] > 80:
            strategies.append("Offer cheaper plan")

        if len(strategies) == 0:
            strategies.append("Maintain service quality")

        return ", ".join(strategies)

    df["Retention Strategy"] = df.apply(retention_strategy, axis=1)

    # ------------------------------------------------
    # Customer Segmentation
    # ------------------------------------------------

    kmeans = KMeans(n_clusters=3, random_state=42)

    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # ------------------------------------------------
    # Dashboard Metrics
    # ------------------------------------------------

    churn_rate = df["Churn Prediction"].mean() * 100
    avg_clv = df["CLV"].mean()

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{churn_rate:.2f}%")
    col3.metric("Average CLV", f"${avg_clv:.2f}")

    # ------------------------------------------------
    # Gauge Meter
    # ------------------------------------------------

    st.subheader("Overall Churn Risk")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_rate,
        title={'text': "Churn Risk"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "red"},
            'steps': [
                {'range':[0,40],'color':"green"},
                {'range':[40,70],'color':"yellow"},
                {'range':[70,100],'color':"red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # Tabs
    # ------------------------------------------------

    tab1,tab2,tab3,tab4 = st.tabs([
        "📊 Executive Dashboard",
        "🤖 Predictions",
        "🎯 Retention Strategies",
        "👥 Customer Segmentation"
    ])

    # ------------------------------------------------
    # Executive Dashboard
    # ------------------------------------------------

    with tab1:

        st.subheader("Churn Distribution")

        fig = px.pie(
            df,
            names="Churn Prediction",
            title="Customer Churn Breakdown"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Charges vs Churn")

        fig = px.box(
            df,
            x="Churn Prediction",
            y="MonthlyCharges",
            title="Monthly Charges Impact"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tenure Distribution")

        fig = px.histogram(
            df,
            x="tenure",
            color="Churn Prediction",
            nbins=30
        )

        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # Predictions Table
    # ------------------------------------------------

    with tab2:

        st.subheader("Customer Predictions")

        st.dataframe(df[[
            "Churn Prediction",
            "Churn Probability",
            "CLV"
        ]])

    # ------------------------------------------------
    # Retention Strategy
    # ------------------------------------------------

    with tab3:

        st.subheader("AI Retention Strategies")

        st.dataframe(df[[
            "Churn Probability",
            "Retention Strategy"
        ]])

    # ------------------------------------------------
    # Segmentation
    # ------------------------------------------------

    with tab4:

        st.subheader("Customer Segmentation")

        fig = px.scatter(
            df,
            x="MonthlyCharges",
            y="tenure",
            color="Cluster",
            title="Customer Segments"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # High Risk Customers
    # ------------------------------------------------

    st.subheader("🔴 High Risk Customers")

    risky = df[df["Churn Probability"] > 70]

    st.dataframe(risky)

    # ------------------------------------------------
    # Download Results
    # ------------------------------------------------

    csv = df.to_csv(index=False)

    st.download_button(
        "Download Prediction Results",
        csv,
        "churn_predictions.csv"
    )

else:

    st.info("Upload a dataset to start the churn analysis.")
