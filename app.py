import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

# Define FeatureEngineer class (MUST match training code)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.median_values = None

    def fit(self, X, y=None):
        X = X.copy()
        X_numeric = X.drop(columns=['ocean_proximity'])
        
        # Derived features
        X_numeric['rooms_per_household'] = X_numeric['total_rooms'] / X_numeric['households']
        X_numeric['bedrooms_per_room'] = X_numeric['total_bedrooms'] / X_numeric['total_rooms']
        X_numeric['population_per_household'] = X_numeric['population'] / X_numeric['households']
        X_numeric['log_population'] = np.log(X_numeric['population'] + 1)
        
        X_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.median_values = X_numeric.median()
        self.kmeans.fit(X_numeric[['longitude', 'latitude']])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        ocean_proximity = X[['ocean_proximity']]
        X = X.drop(columns=['ocean_proximity'])
        
        # Derived features
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
        X['population_per_household'] = X['population'] / X['households']
        X['log_population'] = np.log(X['population'] + 1)
        
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(self.median_values, inplace=True)  # Now uses stored medians
        
        X['cluster'] = self.kmeans.predict(X[['longitude', 'latitude']])
        X.drop(columns=['total_rooms', 'total_bedrooms', 'population', 
                       'households', 'longitude', 'latitude'], inplace=True)
        
        return pd.concat([X, ocean_proximity], axis=1)

# Load pipeline
pipeline = joblib.load('california_housing_model_final.sav')

# Streamlit UI
st.title('🏠 California House Price Prediction')

# Input widgets
longitude = st.number_input('Longitude', value=-118.49)
latitude = st.number_input('Latitude', value=34.26)
housing_median_age = st.number_input('Housing Median Age', value=28)
total_rooms = st.number_input('Total Rooms', value=2127)
total_bedrooms = st.number_input('Total Bedrooms', value=434)
population = st.number_input('Population', value=1152)
households = st.number_input('Households', value=434)
median_income = st.number_input('Median Income', value=4.99)
ocean_proximity = st.selectbox('Ocean Proximity', 
    ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])

# Prediction
if st.button('Predict Price'):
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })
    
    try:
        pred = pipeline.predict(input_data)[0]
        st.success(f'Predicted Median House Value: **${pred*100000:,.0f}**')
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
