import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pickle

# Load the trained CatBoost model
# model = joblib.load("final_catboost_model.pkl")
with open(r"final_catboost_model.pkl","rb") as f:
    model=pickle.load(f)

# App Title
st.title("🏡 California House Price Prediction")

# Create two columns for better UI layout
col1, col2 = st.columns(2)

# Left Column - User Inputs
with col1:
    st.subheader("🏠 House Features")
    
    median_income = st.slider("Median Income", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    housing_median_age = st.slider("Housing Median Age", min_value=0, max_value=100, value=25, step=1)
    total_rooms = st.slider("Total Rooms", min_value=0, max_value=10000, value=500, step=50)
    total_bedrooms = st.slider("Total Bedrooms", min_value=0, max_value=5000, value=100, step=10)
    population = st.slider("Population", min_value=0, max_value=40000, value=3000, step=100)
    households = st.slider("Households", min_value=0, max_value=5000, value=500, step=10)

    # Ocean Proximity Options
    ocean_proximity = st.selectbox("Ocean Proximity", ["INLAND", "NEAR OCEAN", "<1H OCEAN", "NEAR BAY"])

# Feature Engineering (Derived Features)
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households
log_population = np.log(population + 1)

# Encoding Ocean Proximity (One-Hot Encoding)
ocean_mapping = {
    "INLAND": [1, 0, 0, 0],
    "NEAR OCEAN": [0, 1, 0, 0],
    "<1H OCEAN": [0, 0, 1, 0],
    "NEAR BAY": [0, 0, 0, 1]
}
ocean_encoded = ocean_mapping[ocean_proximity]

# Clustering (Dummy Cluster Assignment)
cluster = 3  # Adjust if you have a clustering logic in training

# Prepare input data for prediction
input_data = np.array([[median_income, 
                        ocean_encoded[0],  # INLAND
                        population_per_household, 
                        cluster, 
                        rooms_per_household, 
                        housing_median_age, 
                        bedrooms_per_room, 
                        ocean_encoded[1],  # NEAR OCEAN
                        log_population, 
                        ocean_encoded[2],  # <1H OCEAN
                        ocean_encoded[3]]  # NEAR BAY
                       ])

# Right Column - Prediction Output
with col2:
    st.subheader("💰 Predicted House Price")

    if st.button("🏡 Predict Price"):
        # Make Prediction
        prediction = model.predict(input_data)[0]
        
        # Display Result
        st.success(f"🏠 The estimated house price is: **${prediction:,.2f}**")
