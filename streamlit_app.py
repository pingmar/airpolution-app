import pandas as pd
import numpy as np
import requests
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    handlers=[logging.FileHandler('air_quality_prediction.log'),
                              logging.StreamHandler()])

class AirQualityPredictor:
    def __init__(self, filepath=None, data_sources=None):
        self.logger = logging.getLogger(__name__)
        self.filepath = filepath
        self.data_sources = data_sources or [
            'https://openaq.org/api/v2/measurements',
            'https://api.openweathermap.org/data/2.5/air_pollution'
        ]
        self.raw_data = self._load_data()
        self.processed_data = None
        self.model = None
        self.scaler = StandardScaler()
        self.logger.info("Air Quality Prediction Project Initialized")

    def _load_data(self):
        """Load data from CSV or API sources."""
        if self.filepath:
            try:
                df = pd.read_csv(self.filepath, sep=';', decimal=',', encoding='utf-8')
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
                df = df.drop(['Date', 'Time'], axis=1)
                df = df.replace(-200, np.nan)
                self.logger.info(f"CSV data loaded: {len(df)} records")
                return df
            except Exception as e:
                self.logger.error(f"Error loading CSV file: {e}")
                return pd.DataFrame()
        
        dataframes = []
        for source in self.data_sources:
            try:
                response = requests.get(source)
                df = pd.DataFrame(response.json().get('results', []))
                dataframes.append(df)
            except Exception as e:
                self.logger.error(f"Error collecting data from {source}: {e}")
        
        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        return pd.DataFrame()
    
    def preprocess_data(self):
        """Clean and prepare data for modeling."""
        self.logger.info("Starting Data Preprocessing")
        if self.raw_data.empty:
            self.logger.warning("No data available for preprocessing.")
            return pd.DataFrame()
        
        self.processed_data = self.raw_data.copy()
        if 'DateTime' in self.processed_data.columns:
            self.processed_data['Hour'] = self.processed_data['DateTime'].dt.hour
            self.processed_data['Month'] = self.processed_data['DateTime'].dt.month
        
        self.processed_data = self.processed_data.dropna(thresh=len(self.processed_data) * 0.8, axis=1)
        self.processed_data = self.processed_data.dropna()
        
        numeric_cols = [col for col in self.processed_data.columns if self.processed_data[col].dtype in ['float64', 'int64']]
        self.processed_data = self.processed_data[numeric_cols]
        
        self.logger.info(f"Preprocessing completed. Records remaining: {len(self.processed_data)}")
        return self.processed_data
    
    def train_model(self, target='CO(GT)', test_size=0.2):
        """Train a machine learning model."""
        self.logger.info(f"Starting Model Training for {target}")
        if target not in self.processed_data.columns:
            self.logger.error(f"Target variable {target} not found in data.")
            return {}
        
        features = [col for col in self.processed_data.columns if col != target]
        X = self.processed_data[features]
        y = self.processed_data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
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
    
    def streamlit_dashboard(self):
        """Create an interactive Streamlit dashboard."""
        st.title('Air Quality Prediction Dashboard')

        # Define all input features used during training
        feature_names = [col for col in self.processed_data.columns if col != 'CO(GT)']

        # User input sliders
        input_data = {}
        for feature in feature_names:
            if feature in ['Temperature', 'Humidity']:  # Example features
                input_data[feature] = st.slider(f'{feature}', min_value=0.0, max_value=100.0, value=25.0)
            else:
                input_data[feature] = 0  # Default missing features to zero

        # Convert input to DataFrame with the correct feature order
        input_df = pd.DataFrame([input_data]).reindex(columns=feature_names, fill_value=0)

        # Scale the input data using the same scaler
        scaled_input = self.scaler.transform(input_df)

        # Prediction
        if st.button('Predict AQI'):
            prediction = self.model.predict(scaled_input)
            st.write(f'Predicted Air Quality Index: {prediction[0]:.2f}')


if __name__ == '__main__':
    predictor = AirQualityPredictor(filepath='AirQualityUCI.csv')
    predictor.preprocess_data()
    metrics = predictor.train_model()
    print("Model Performance:", metrics)
    predictor.streamlit_dashboard()
