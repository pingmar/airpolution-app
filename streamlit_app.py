import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('air_quality_prediction.log'),
        logging.StreamHandler()
    ]
)

class AirQualityPredictor:
    def __init__(self):
        """Initialize the Air Quality Prediction project with simulated data"""
        self.logger = logging.getLogger(__name__)
        self.raw_data = self._generate_sample_data()
        self.processed_data = None
        self.model = None
        
        # Log initialization
        self._log_process_start()
    
    def _log_process_start(self):
        """Log project initialization details"""
        self.logger.info("Air Quality Prediction Project Initialized")
        self.logger.info(f"Sample Data Generated: {len(self.raw_data)} records")
    
    def _log_process_complete(self, stage, details=None):
        """
        Log completion of a processing stage
        
        Args:
            stage (str): Name of the completed stage
            details (dict, optional): Additional details to log
        """
        self.logger.info(f"{stage} Processing Completed")
        if details:
            for key, value in details.items():
                self.logger.info(f"{key}: {value}")
    
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
        self.logger.info("Starting Data Preprocessing")
        
        # Ensure we have data
        if self.raw_data is None:
            self.raw_data = self._generate_sample_data()
        
        # Feature engineering
        self.processed_data = self.raw_data.copy()
        self.processed_data['hour'] = self.processed_data['datetime'].dt.hour
        self.processed_data['month'] = self.processed_data['datetime'].dt.month
        
        # Log preprocessing details
        self._log_process_complete('Preprocessing', {
            'Total Records': len(self.processed_data),
            'Features Added': ['hour', 'month']
        })
        
        return self.processed_data
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        self.logger.info("Starting Exploratory Data Analysis")
        
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
        
        # Log analysis completion
        self._log_process_complete('Exploratory Analysis', {
            'Correlation Matrix Shape': correlation_matrix.shape,
            'Visualization Files': ['correlation_heatmap.png', 'temperature_vs_aqi.png']
        })
        
        return {
            'correlation_matrix': correlation_matrix,
            'heatmap_file': 'correlation_heatmap.png',
            'scatter_plot_file': 'temperature_vs_aqi.png'
        }
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train machine learning model for AQI prediction"""
        self.logger.info("Starting Model Training")
        
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
        performance = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        # Log model training details
        self._log_process_complete('Model Training', {
            'Model Type': 'Random Forest',
            'Training Samples': len(X_train),
            'Test Samples': len(X_test),
            **performance
        })
        
        return performance

# Execution logging
if __name__ == '__main__':
    try:
        predictor = AirQualityPredictor()
        predictor.preprocess_data()
        predictor.exploratory_analysis()
        model_performance = predictor.train_model()
        logging.info("Air Quality Prediction Project Completed Successfully")
    except Exception as e:
        logging.error(f"Project Execution Failed: {e}", exc_info=True)