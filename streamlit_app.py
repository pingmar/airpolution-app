import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class AirQualityPredictor:
    def __init__(self):
        """Initialize the Air Quality Prediction project with simulated data"""
        self.raw_data = self._generate_sample_data()
        self.processed_data = None
        self.model = None
        
    def _generate_sample_data(self, n_samples=1000):
        """
        Generate synthetic air quality dataset
        
        Args:
            n_samples (int): Number of data points to generate
        
        Returns:
            pd.DataFrame: Simulated air quality data
        """
        np.random.seed(42)
        data = {
            'temperature': np.random.uniform(0, 40, n_samples),
            'humidity': np.random.uniform(20, 90, n_samples),
            'wind_speed': np.random.uniform(0, 20, n_samples),
            'pollutant_levels': np.random.uniform(0, 500, n_samples),
            'latitude': np.random.uniform(-90, 90, n_samples),
            'longitude': np.random.uniform(-180, 180, n_samples),
            'datetime': pd.date_range(start='2023-01-01', periods=n_samples),
            'air_quality_index': np.random.uniform(0, 500, n_samples)
        }
        return pd.DataFrame(data)
    
    def preprocess_data(self):
        """
        Clean and prepare data for modeling
        
        Returns:
            pd.DataFrame: Processed and cleaned dataset
        """
        # Ensure we have data
        if self.raw_data is None:
            self.raw_data = self._generate_sample_data()
        
        # Feature engineering
        self.processed_data = self.raw_data.copy()
        self.processed_data['hour'] = self.processed_data['datetime'].dt.hour
        self.processed_data['month'] = self.processed_data['datetime'].dt.month
        
        return self.processed_data
    
    def exploratory_analysis(self):
        """
        Perform exploratory data analysis
        
        Returns:
            dict: Analysis visualizations and insights
        """
        # Correlation analysis
        correlation_matrix = self.processed_data.select_dtypes(include=[np.number]).corr()
        
        # Visualizations
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        
        plt.figure(figsize=(12, 6))
        plt.scatter(self.processed_data['temperature'], 
                    self.processed_data['air_quality_index'])
        plt.xlabel('Temperature')
        plt.ylabel('Air Quality Index')
        plt.title('Temperature vs Air Quality Index')
        plt.savefig('temperature_vs_aqi.png')
        
        return {
            'correlation_matrix': correlation_matrix,
            'heatmap_file': 'correlation_heatmap.png',
            'scatter_plot_file': 'temperature_vs_aqi.png'
        }
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Train machine learning model for AQI prediction
        
        Returns:
            dict: Model performance metrics
        """
        # Select features
        features = ['temperature', 'humidity', 'wind_speed', 
                    'pollutant_levels', 'hour', 'month', 
                    'latitude', 'longitude']
        X = self.processed_data[features]
        y = self.processed_data['air_quality_index']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions and evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred)
        }
    
    def streamlit_dashboard(self):
        """
        Create interactive Streamlit dashboard for AQI predictions
        """
        st.title('Air Quality Index Predictor')
        
        # Input features
        temperature = st.slider('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
        humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=50)
        wind_speed = st.slider('Wind Speed (m/s)', min_value=0.0, max_value=20.0, value=5.0)
        
        # Prediction
        if st.button('Predict AQI'):
            # Prepare input for prediction
            input_data = np.array([[
                temperature, humidity, wind_speed, 
                200, 12, 6,  # default values for other features
                0, 0  # default lat/long
            ]])
            
            # Scale input
            scaler = StandardScaler()
            scaler.fit(self.processed_data[['temperature', 'humidity', 'wind_speed', 
                                            'pollutant_levels', 'hour', 'month', 
                                            'latitude', 'longitude']])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = self.model.predict(input_scaled)
            st.write(f'Predicted Air Quality Index: {prediction[0]:.2f}')

# Uncomment for standalone script execution
if __name__ == '__main__':
    predictor = AirQualityPredictor()
    predictor.preprocess_data()
    predictor.exploratory_analysis()
    model_performance = predictor.train_model()
    print("Model Performance:", model_performance)