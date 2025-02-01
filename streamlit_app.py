import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s: %(message)s',
                   handlers=[logging.FileHandler('air_quality_prediction.log'),
                            logging.StreamHandler()])

class AirQualityPredictor:
    def __init__(self, filepath='AirQualityUCI.csv'):
        self.logger = logging.getLogger(__name__)
        self.raw_data = self._load_data(filepath)
        self.processed_data = None
        self.model = None
        self.logger.info("Air Quality Prediction Project Initialized")
        
    def _load_data(self, filepath):
        """Load and preprocess the UCI Air Quality dataset"""
        # Read CSV with proper decimal separator
        df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8')
        
        # Convert Date and Time columns to datetime with explicit format
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                      format='%d/%m/%Y %H.%M.%S')
        
        # Drop original Date and Time columns
        df = df.drop(['Date', 'Time'], axis=1)
        
        # Replace error values
        df = df.replace(-200, np.nan)
        
        self.logger.info(f"Data loaded: {len(df)} records")
        return df
    
    def preprocess_data(self):
        """Clean and prepare data for modeling"""
        self.logger.info("Starting Data Preprocessing")
        
        self.processed_data = self.raw_data.copy()
        self.processed_data['Hour'] = self.processed_data['DateTime'].dt.hour
        self.processed_data['Month'] = self.processed_data['DateTime'].dt.month
        
        # Drop columns with too many missing values
        self.processed_data = self.processed_data.dropna(thresh=len(self.processed_data)*0.8, axis=1)
        self.processed_data = self.processed_data.dropna()
        
        # Remove unnamed and non-numeric columns except Hour and Month
        keep_cols = [col for col in self.processed_data.columns 
                    if ('Hour' in col or 'Month' in col or 
                        self.processed_data[col].dtype in ['float64', 'int64']) 
                    and 'Unnamed' not in col]
        
        self.processed_data = self.processed_data[keep_cols]
        
        self.logger.info(f"Preprocessing completed. Records remaining: {len(self.processed_data)}")
        return self.processed_data
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        self.logger.info("Starting Exploratory Analysis")
        
        correlation_matrix = self.processed_data.corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        
        self.logger.info("Exploratory Analysis completed")
        return correlation_matrix
    
    def train_model(self, target='CO(GT)', test_size=0.2):
        """Train Random Forest model for air quality prediction"""
        self.logger.info(f"Starting Model Training for {target}")
        
        features = [col for col in self.processed_data.columns 
                   if col != target]
        
        X = self.processed_data[features]
        y = self.processed_data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.model = RandomForestRegressor(n_estimators=100)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        self.logger.info(f"Model Training completed. Metrics: {metrics}")
        return metrics

def main():
    st.title('Air Quality Prediction Dashboard')
    
    predictor = AirQualityPredictor()
    processed_data = predictor.preprocess_data()
    correlation_matrix = predictor.exploratory_analysis()
    metrics = predictor.train_model()
    
    st.header('Dataset Information')
    st.write(f"Total records: {len(processed_data)}")
    st.write(f"Features: {', '.join(processed_data.columns)}")
    
    st.header('Model Performance')
    st.write(f"MAE: {metrics['MAE']:.3f}")
    st.write(f"RMSE: {metrics['RMSE']:.3f}")
    st.write(f"R² Score: {metrics['R2']:.3f}")
    
    st.header('Correlation Heatmap')
    st.image('correlation_heatmap.png')

if __name__ == '__main__':
    main()