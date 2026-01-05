
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection App")
st.write("This app predicts whether a credit card transaction is **Fraudulent** or **Legitimate**.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df = load_data()

# Sidebar
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Choose an option", ["Dataset Overview", "EDA", "Predict Fraud"])

# Dataset Overview
if option == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write("Shape of dataset:", df.shape)

# EDA Section
elif option == "EDA":
    st.subheader("Fraud vs Non-Fraud Distribution")
    st.bar_chart(df["Class"].value_counts())

    st.subheader("Transaction Amount Distribution")
    st.line_chart(df["Amount"].sample(1000))

# Train model (simple training inside app for demo)
elif option == "Predict Fraud":
    st.subheader("Fraud Prediction")

    # Feature scaling
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled["Amount"] = scaler.fit_transform(df_scaled[["Amount"]])
    df_scaled["Time"] = scaler.fit_transform(df_scaled[["Time"]])

    X = df_scaled.drop("Class", axis=1)
    y = df_scaled["Class"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    st.write("Enter transaction details:")

    input_data = []
    for col in X.columns:
        value = st.number_input(col, value=0.0)
        input_data.append(value)

    if st.button("Predict"):
        prediction = model.predict([input_data])[0]
        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")
