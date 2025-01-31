import streamlit as st
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class AirQualityPredictor:
    def __init__(self):
        self.raw_data = self._generate_sample_data()
        self.processed_data = None
        self.model = None
        self.model_performance = None
    
    def _generate_sample_data(self, n_samples=1000):
        """Generate synthetic air quality dataset"""
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
        """Clean and prepare data for modeling"""
        self.processed_data = self.raw_data.copy()
        self.processed_data['hour'] = self.processed_data['datetime'].dt.hour
        self.processed_data['month'] = self.processed_data['datetime'].dt.month
        return self.processed_data
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
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
        
        return correlation_matrix
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train machine learning model for AQI prediction"""
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
        self.model_performance = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        return self.model_performance

def main():
    st.title('Air Quality Prediction Dashboard')
    
    # Initialize predictor
    predictor = AirQualityPredictor()
    
    # Display processing stages
    st.header('Project Processing Stages')
    
    # Data Preprocessing
    st.subheader('1. Data Preprocessing')
    processed_data = predictor.preprocess_data()
    st.write(f'Total Records: {len(processed_data)}')
    st.write(f'Features Added: hour, month')
    
    # Exploratory Analysis
    st.subheader('2. Exploratory Data Analysis')
    correlation_matrix = predictor.exploratory_analysis()
    st.write('Correlation Matrix Shape:', correlation_matrix.shape)
    
    # Model Training
    st.subheader('3. Model Training')
    model_performance = predictor.train_model()
    
    # Display Model Performance
    st.write('Model Performance Metrics:')
    st.write(f"Mean Absolute Error (MAE): {model_performance['mae']:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {model_performance['rmse']:.2f}")
    st.write(f"R² Score: {model_performance['r2_score']:.2f}")
    
    # Visualization Display
    st.subheader('Visualizations')
    col1, col2 = st.columns(2)
    
    with col1:
        st.image('correlation_heatmap.png', caption='Correlation Heatmap')
    
    with col2:
        st.image('temperature_vs_aqi.png', caption='Temperature vs Air Quality Index')

if __name__ == '__main__':
    main()