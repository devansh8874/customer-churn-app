import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved models
rf_model = joblib.load('random_forest_churn_model.pkl')
kmeans_model = joblib.load('kmeans_cluster_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Customer Churn Prediction App")

st.write("""
### Enter Customer Details
This app uses **Unsupervised Learning (Clustering)** to segment the customer 
and **Supervised Learning (Random Forest)** to predict churn risk.
""")

# --- INPUTS ---
# Ensure these match the columns used during training!
tenure = st.number_input("Tenure (Months)", min_value=0)
warehouse_dist = st.number_input("Distance from Warehouse", min_value=0)
complain = st.selectbox("Complain Raised?", [0, 1])
day_since_order = st.number_input("Days Since Last Order", min_value=0)
cashback = st.number_input("Cashback Amount", min_value=0.0)

# If you had more columns in your training data, add them here...

# --- PREDICTION ---
if st.button("Predict Churn Risk"):
    # 1. Arrange input data in the EXACT SAME ORDER as X_train
    input_data = [[tenure, warehouse_dist, complain, day_since_order, cashback]]
    
    # 2. Scale the data
    scaled_data = scaler.transform(input_data)
    
    # 3. Get Cluster (Unsupervised)
    cluster = kmeans_model.predict(scaled_data)[0]
    st.info(f"Customer belongs to Segment (Cluster): {cluster}")
    
    # 4. Add Cluster to input data (for Supervised Model)
    final_input = np.append(input_data, cluster).reshape(1, -1)
    
    # 5. Predict Churn
    prediction = rf_model.predict(final_input)
    
    if prediction[0] == 1:
        st.error("Prediction: High Risk of Churn! (Likely to leave)")
    else:
        st.success("Prediction: Low Risk. (Likely to stay)")