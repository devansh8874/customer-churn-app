import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved models
try:
    rf_model = joblib.load('random_forest_churn_model.pkl')
    kmeans_model = joblib.load('kmeans_cluster_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please make sure .pkl files are in the same directory.")
    st.stop()


st.title("Customer Churn Prediction App")

st.write("""
### Enter Customer Details
This app uses **Unsupervised Learning (Clustering)** to segment the customer 
and **Supervised Learning (Random Forest)** to predict churn risk.
""")

# --- INPUTS ---
# We must create an input for ALL 10 features from your list
st.subheader("Customer Details")
tenure = st.number_input("Tenure (Months)", min_value=0)
warehouse_to_home = st.number_input("Warehouse to Home Distance (km)", min_value=0)
num_devices = st.number_input("Number of Devices Registered", min_value=0, step=1)
num_address = st.number_input("Number of Addresses", min_value=1, step=1)

st.subheader("Order & Satisfaction Details")
# Note: You may need to adjust the options for these select boxes
# based on what values are in your dataset
prefered_order_cat = st.selectbox("Preferred Order Category (Encoded)", [0, 1, 2, 3, 4, 5]) 
satisfaction_score = st.selectbox("Satisfaction Score", [1, 2, 3, 4, 5])
day_since_last_order = st.number_input("Days Since Last Order", min_value=0, step=1)
cashback = st.number_input("Cashback Amount", min_value=0.0)

st.subheader("Profile Details")
marital_status = st.selectbox("Marital Status (Encoded)", [0, 1, 2], help="0=Single, 1=Married, 2=Divorced/Other")
complain = st.selectbox("Complain Raised in Last Month?", [0, 1], help="0=No, 1=Yes")


# --- PREDICTION ---
if st.button("Predict Churn Risk"):
    
    # 1. Arrange input data in the EXACT SAME ORDER as your list
    input_data = [[
        tenure,
        warehouse_to_home,
        num_devices,
        prefered_order_cat,
        satisfaction_score,
        marital_status,
        num_address,
        complain,
        day_since_last_order,
        cashback
    ]]
    
    # 2. Scale the data (This will now work as it has 10 features)
    scaled_data = scaler.transform(input_data)
    
    # 3. Get Cluster (Unsupervised) - Prediction 1
    cluster = kmeans_model.predict(scaled_data)[0]
    st.info(f"This customer belongs to Segment (Cluster): {cluster}")
    
    # 4. Get Churn Prediction (Supervised) - Prediction 2
    #    USE THE *SAME* 10-FEATURE scaled_data
    #    This was the part causing your 11-feature error
    
    churn_prediction = rf_model.predict(scaled_data)[0]
    churn_probability = rf_model.predict_proba(scaled_data)[0][1] # Get probability of churn (class 1)
    
    st.subheader("Churn Prediction Result")
    
    if churn_prediction == 1:
        st.error(f"High Churn Risk (Prediction: 1)")
        st.write(f"**Probability of Churn: {churn_probability * 100:.2f}%**")
    else:
        st.success(f"Low Churn Risk (Prediction: 0)")
        st.write(f"**Probability of Churn: {churn_probability * 100:.2f}%**")