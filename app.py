# Import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Judul Utama
st.title('Housing Price Predictor')
st.text('This web can be used to predict housing prices')

# Menambahkan sidebar
st.sidebar.header("Please input the housing features")

def create_user_input():
    # Numerical Features
    longitude = st.sidebar.slider('Longitude', min_value=-124.35, max_value=-114.31, value=-119.0, step=0.01)
    latitude = st.sidebar.slider('Latitude', min_value=32.54, max_value=41.95, value=37.0, step=0.01)
    housing_median_age = st.sidebar.slider('Housing Median Age', min_value=1, max_value=52, value=30)
    total_rooms = st.sidebar.number_input('Total Rooms', min_value=2, max_value=32054, value=2000)
    total_bedrooms = st.sidebar.number_input('Total Bedrooms', min_value=2, max_value=5290, value=400)
    population = st.sidebar.number_input('Population', min_value=3, max_value=15507, value=3000)
    households = st.sidebar.number_input('Households', min_value=2, max_value=5050, value=800)
    median_income = st.sidebar.number_input('Median Income', min_value=4999, max_value=150001, value=50000, step=1)

    # Categorical Features
    ocean_proximity_group = st.sidebar.radio('Ocean Proximity', ['INLAND', 'NEAR BAY', '<1H_OCEAN_SOUTH', 
                                                            '<1H_OCEAN_NORTH', 'NEAR_OCEAN_NORTH', 'NEAR_OCEAN_SOUTH', 'ISLAND'])

    # Creating a dictionary with user input
    user_data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity_group': ocean_proximity_group
    }
    
    # Convert dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get customer data
data_customer = create_user_input()

# Membuat 2 kolom
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Housing Features")
    st.write(data_customer.transpose().rename(columns={'0':'feature_value'}))

# Load model
with open('final_catboost_model.pkl', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict to data
predicted_price = model_loaded.predict(data_customer)

# Menampilkan hasil prediksi
with col2:
    st.subheader('Prediction Result')
    st.write(f'Predicted House Price: ${predicted_price[0]:,.2f}')
