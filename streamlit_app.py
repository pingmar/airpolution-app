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
    def __init__(self, data_sources=None):
        """
        Initialize the Air Quality Prediction project
        
        Args:
            data_sources (list): List of API/data source URLs
        """
        self.data_sources = data_sources or [
            'https://openaq.org/api/v2/measurements',
            'https://api.openweathermap.org/data/2.5/air_pollution'
        ]
        self.raw_data = None
        self.processed_data = None
        self.model = None
        
    def collect_data(self):
        """
        Collect data from multiple sources
        
        Returns:
            pd.DataFrame: Collected air quality and weather data
        """
        dataframes = []
        
        # Simulated data collection (replace with actual API calls)
        for source in self.data_sources:
            try:
                # Placeholder for actual API request
                response = requests.get(source)
                df = pd.DataFrame(response.json().get('results', []))
                dataframes.append(df)
            except Exception as e:
                print(f"Error collecting data from {source}: {e}")
        
        self.raw_data = pd.concat(dataframes, ignore_index=True)
        return self.raw_data
    
    def preprocess_data(self):
        """
        Clean and prepare data for modeling
        
        Returns:
            pd.DataFrame: Processed and cleaned dataset
        """
        if self.raw_data is None:
            self.collect_data()
        
        # Handle missing values
        self.processed_data = self.raw_data.dropna()
        
        # Feature engineering
        self.processed_data['datetime'] = pd.to_datetime(self.processed_data['datetime'])
        self.processed_data['hour'] = self.processed_data['datetime'].dt.hour
        self.processed_data['month'] = self.processed_data['datetime'].dt.month
        
        # Select relevant features
        features = ['temperature', 'humidity', 'wind_speed', 
                    'pollutant_levels', 'hour', 'month', 'latitude', 'longitude']
        target = 'air_quality_index'
        
        return self.processed_data[features + [target]]
    
    def exploratory_analysis(self):
        """
        Perform exploratory data analysis
        
        Returns:
            dict: Analysis visualizations and insights
        """
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.processed_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        
        # Scatter plot of key features
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
        
        Args:
            test_size (float): Proportion of test dataset
            random_state (int): Seed for reproducibility
        
        Returns:
            dict: Model performance metrics
        """
        X = self.processed_data.drop('air_quality_index', axis=1)
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
        temperature = st.slider('Temperature', min_value=0.0, max_value=50.0, value=25.0)
        humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=50)
        
        # Prediction
        if st.button('Predict AQI'):
            prediction = self.model.predict([[temperature, humidity]])
            st.write(f'Predicted Air Quality Index: {prediction[0]:.2f}')

# Example usage
predictor = AirQualityPredictor()
predictor.collect_data()
predictor.preprocess_data()
predictor.exploratory_analysis()
model_performance = predictor.train_model()
print("Model Performance:", model_performance)