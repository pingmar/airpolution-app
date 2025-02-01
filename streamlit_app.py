import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    handlers=[logging.FileHandler('air_quality_prediction.log'),
                              logging.StreamHandler()])

class AirQualityPredictor:
    def __init__(self, dataframe):
        self.logger = logging.getLogger(__name__)
        self.raw_data = dataframe
        self.processed_data = None
        self.model = None
        self.logger.info("Air Quality Prediction Project Initialized")

    def preprocess_data(self, drop_na=True):
        """Clean and prepare data for modeling"""
        self.logger.info("Starting Data Preprocessing")
        self.processed_data = self.raw_data.copy()
        
        if 'Date' in self.processed_data.columns and 'Time' in self.processed_data.columns:
            self.processed_data['DateTime'] = pd.to_datetime(
                self.processed_data['Date'] + ' ' + self.processed_data['Time'], 
                format='%d/%m/%Y %H.%M.%S', errors='coerce')
            self.processed_data = self.processed_data.drop(['Date', 'Time'], axis=1)
            self.processed_data['Hour'] = self.processed_data['DateTime'].dt.hour
            self.processed_data['Month'] = self.processed_data['DateTime'].dt.month
        
        self.processed_data = self.processed_data.replace(-200, np.nan)
        if drop_na:
            self.processed_data = self.processed_data.dropna()
        
        self.logger.info(f"Preprocessing completed. Records remaining: {len(self.processed_data)}")
        return self.processed_data
    
    def train_model(self, target, test_size=0.2, n_estimators=100):
        """Train Random Forest model for air quality prediction"""
        self.logger.info(f"Starting Model Training for {target}")
        
        features = [col for col in self.processed_data.columns if col != target]
        X = self.processed_data[features]
        y = self.processed_data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, [test_size, 1-test_size], random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.model = RandomForestRegressor(n_estimators=n_estimators)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        self.logger.info(f"Model Training completed. Metrics: {metrics}")
        return metrics, y_test, y_pred

def main():
    st.title('Interactive Air Quality Prediction')
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8')
        st.write("Preview of Dataset:")
        st.dataframe(df.head())
        
        predictor = AirQualityPredictor(df)
        drop_na = st.checkbox("Drop missing values?", value=True)
        processed_data = predictor.preprocess_data(drop_na)
        
        target_variable = st.selectbox("Select Target Variable", processed_data.columns)
        test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        n_estimators = st.slider("Number of Trees in Random Forest", min_value=50, max_value=300, value=100, step=10)
        
        if st.button("Train Model"):
            metrics, y_test, y_pred = predictor.train_model(target_variable, test_size, n_estimators)
            
            st.write("## Model Performance")
            st.write(f"MAE: {metrics['MAE']:.3f}")
            st.write(f"RMSE: {metrics['RMSE']:.3f}")
            st.write(f"R² Score: {metrics['R2']:.3f}")
            
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs. Predicted Values")
            st.pyplot(fig)

if __name__ == '__main__':
    main()
