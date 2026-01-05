# Programming-for-AI-Project
project of credit card fraud detection data set

# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using **Exploratory Data Analysis (EDA)** and **Machine Learning techniques**.  
A **Streamlit-based web application** is developed to provide an interactive interface for fraud prediction.

The dataset used is highly imbalanced, representing a real-world fraud detection problem.

---

## 🎯 Objectives
- Perform Exploratory Data Analysis (EDA)
- Preprocess and scale transaction data
- Train machine learning models
- Detect fraudulent transactions effectively
- Deploy the model using Streamlit

---

## 📂 Dataset Information
- **Name:** Credit Card Fraud Detection Dataset
- **Source:** Kaggle
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492
- **Features:**
  - `V1` – `V28`: PCA transformed features
  - `Time`: Time elapsed since first transaction
  - `Amount`: Transaction amount
  - `Class`: Target variable  
    - `0` = Legitimate  
    - `1` = Fraudulent  

---

## 🧪 Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit

---

## ⚙️ Project Structure

---

## 🔍 Exploratory Data Analysis (EDA)
- Fraud vs non-fraud distribution
- Transaction amount analysis
- Statistical summary of features
- Dataset imbalance analysis

---

## 🤖 Machine Learning Models
- Logistic Regression
- Random Forest Classifier

👉 **Random Forest** was selected as the final model due to better **recall** and **F1-score**, which are crucial in fraud detection.

---

## 🖥️ Streamlit Application
### Features:
- Dataset overview
- Fraud distribution visualization
- Interactive fraud prediction
- Simple and user-friendly interface

### Run the Application:
```bash
pip install -r requirements.txt
streamlit run app.py
