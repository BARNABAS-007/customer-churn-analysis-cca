import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Analysis and Retention Strategy Recommendation System",
    layout="wide"
)

st.title("📊 Customer Churn Analysis and Retention Strategy Recommendation System")

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

features = scaler.feature_names_in_

# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------

df = pd.read_csv("customer_data.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------

st.sidebar.header("Dashboard Filters")

if "tenure" in df.columns:

    tenure_filter = st.sidebar.slider(
        "Customer Tenure",
        int(df["tenure"].min()),
        int(df["tenure"].max()),
        (int(df["tenure"].min()), int(df["tenure"].max()))
    )

    df = df[(df["tenure"] >= tenure_filter[0]) & (df["tenure"] <= tenure_filter[1])]

if "MonthlyCharges" in df.columns:

    charge_filter = st.sidebar.slider(
        "Monthly Charges",
        float(df["MonthlyCharges"].min()),
        float(df["MonthlyCharges"].max()),
        (float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()))
    )

    df = df[(df["MonthlyCharges"] >= charge_filter[0]) & (df["MonthlyCharges"] <= charge_filter[1])]

# ------------------------------------------------
# CONVERT CATEGORICAL
# ------------------------------------------------

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = pd.factorize(df[col])[0]

# ------------------------------------------------
# MATCH MODEL FEATURES
# ------------------------------------------------

for col in features:
    if col not in df.columns:
        df[col] = 0

X = df[features]

# ------------------------------------------------
# SCALE DATA
# ------------------------------------------------

X_scaled = scaler.transform(X)

# ------------------------------------------------
# MODEL PREDICTIONS
# ------------------------------------------------

pred = model.predict(X_scaled)
prob = model.predict_proba(X_scaled)

df["Churn Prediction"] = pred
df["Churn Probability"] = prob[:,1] * 100

# ------------------------------------------------
# CUSTOMER LIFETIME VALUE
# ------------------------------------------------

if "MonthlyCharges" in df.columns and "tenure" in df.columns:
    df["CLV"] = df["MonthlyCharges"] * df["tenure"] * (1 - prob[:,1])
else:
    df["CLV"] = 0

# ------------------------------------------------
# RISK LEVEL
# ------------------------------------------------

def risk_level(p):

    if p > 70:
        return "High Risk"
    elif p > 40:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk Level"] = df["Churn Probability"].apply(risk_level)

# ------------------------------------------------
# RETENTION STRATEGY
# ------------------------------------------------

def retention_strategy(row):

    strategies = []

    if row["Churn Probability"] > 70:
        strategies.append("Offer loyalty discount")

    if "tenure" in row and row["tenure"] < 12:
        strategies.append("Provide onboarding support")

    if "MonthlyCharges" in row and row["MonthlyCharges"] > 80:
        strategies.append("Suggest lower price plan")

    if len(strategies) == 0:
        strategies.append("Maintain service quality")

    return ", ".join(strategies)

df["Retention Strategy"] = df.apply(retention_strategy, axis=1)

# ------------------------------------------------
# CUSTOMER SEGMENTATION
# ------------------------------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ------------------------------------------------
# KPI DASHBOARD
# ------------------------------------------------

st.subheader("Executive KPI Dashboard")

churn_rate = df["Churn Prediction"].mean() * 100
avg_clv = df["CLV"].mean()
high_risk_count = len(df[df["Risk Level"] == "High Risk"])

c1,c2,c3,c4 = st.columns(4)

c1.metric("Total Customers", len(df))
c2.metric("Churn Rate", f"{churn_rate:.2f}%")
c3.metric("Average CLV", f"${avg_clv:.2f}")
c4.metric("High Risk Customers", high_risk_count)

# ------------------------------------------------
# CHURN RISK GAUGE
# ------------------------------------------------

st.subheader("Overall Churn Risk")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=churn_rate,
    title={'text': "Churn Risk"},
    gauge={
        'axis': {'range':[0,100]},
        'steps':[
            {'range':[0,40],'color':"green"},
            {'range':[40,70],'color':"yellow"},
            {'range':[70,100],'color':"red"}
        ]
    }
))

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------

st.subheader("AI Model Feature Importance")

if hasattr(model,"feature_importances_"):

    importance = pd.DataFrame({
        "Feature":features,
        "Importance":model.feature_importances_
    }).sort_values("Importance",ascending=False)

    fig = px.bar(
        importance.head(10),
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# TABS
# ------------------------------------------------

tab1,tab2,tab3,tab4 = st.tabs([
    "Executive Dashboard",
    "Churn Predictions",
    "Retention Strategies",
    "Customer Segmentation"
])

# ------------------------------------------------
# EXECUTIVE DASHBOARD
# ------------------------------------------------

with tab1:

    fig = px.pie(df, names="Churn Prediction")
    st.plotly_chart(fig)

    if "MonthlyCharges" in df.columns:

        fig = px.box(df, x="Churn Prediction", y="MonthlyCharges")
        st.plotly_chart(fig)

    if "tenure" in df.columns:

        fig = px.histogram(df, x="tenure", color="Churn Prediction")
        st.plotly_chart(fig)

# ------------------------------------------------
# PREDICTIONS
# ------------------------------------------------

with tab2:

    st.dataframe(df[
        ["Churn Prediction","Churn Probability","Risk Level","CLV"]
    ])

# ------------------------------------------------
# RETENTION STRATEGIES
# ------------------------------------------------

with tab3:

    st.dataframe(df[
        ["Risk Level","Churn Probability","Retention Strategy"]
    ])

# ------------------------------------------------
# CUSTOMER SEGMENTS
# ------------------------------------------------

with tab4:

    if "MonthlyCharges" in df.columns and "tenure" in df.columns:

        fig = px.scatter(
            df,
            x="MonthlyCharges",
            y="tenure",
            color="Cluster",
            title="Customer Segments"
        )

        st.plotly_chart(fig)

# ------------------------------------------------
# HIGH RISK CUSTOMERS
# ------------------------------------------------

st.subheader("High Risk Customers")

high_risk = df[df["Risk Level"] == "High Risk"]

st.dataframe(high_risk)

# ------------------------------------------------
# CUSTOMER CHURN SIMULATOR
# ------------------------------------------------

st.subheader("Customer Churn Simulator")

st.write("Enter customer data to predict churn")

input_data = {}

cols = st.columns(3)

for i, feature in enumerate(features):

    with cols[i % 3]:

        input_data[feature] = st.number_input(
            feature,
            value=float(X[feature].mean())
        )

if st.button("Predict Customer Churn"):

    input_df = pd.DataFrame([input_data])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"Customer likely to CHURN ({probability:.2f}%)")
    else:
        st.success(f"Customer likely to STAY ({100-probability:.2f}%)")

# ------------------------------------------------
# DOWNLOAD RESULTS
# ------------------------------------------------

csv = df.to_csv(index=False)

st.download_button(
    label="Download Prediction Results",
    data=csv,
    file_name="churn_predictions.csv"
)


