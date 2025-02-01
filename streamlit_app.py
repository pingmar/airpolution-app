import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import streamlit as st
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def fetch_air_quality_data(capitals, parameter="pm25", limit=100, filename="air_quality_data.csv"):
    """
    Fetch air quality data from OpenAQ API for multiple capital cities and store results.
    """
    # Check if data file exists and is less than 1 day old
    if os.path.exists(filename):
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filename))
        if file_age.days < 1:
            print("Loading recent data from CSV file...")
            return pd.read_csv(filename)
    
    records = []
    for city in capitals:
        # Updated API endpoint for v2
        url = f"https://api.openaq.org/v3/measurements"
        headers = {'X-API-Key': st.secrets["AQ_API"]}
        params = {
            'city': city,
            'parameter': parameter,
            'limit': limit,
            'order_by': 'datetime',
            'sort': 'desc'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            for item in results:
                record = {
                    "city": item.get("city"),
                    "country": item.get("country"),
                    "location": item.get("location"),
                    "parameter": item.get("parameter"),
                    "value": item.get("value"),
                    "unit": item.get("unit"),
                    "latitude": item.get("coordinates", {}).get("latitude"),
                    "longitude": item.get("coordinates", {}).get("longitude"),
                    "date_utc": item.get("date", {}).get("utc"),
                }
                if all(v is not None for v in record.values()):
                    records.append(record)
                    
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch data for {city}: {str(e)}")
    
    if not records:
        st.error("No data was retrieved from the API")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    df['date_utc'] = pd.to_datetime(df['date_utc'])
    df.to_csv(filename, index=False)
    return df

def prepare_data_for_prediction(df, city):
    """Prepare data for prediction model."""
    city_data = df[df['city'] == city].copy()
    if city_data.empty:
        return None, None, None
    
    city_data = city_data.sort_values('date_utc')
    city_data['hour'] = city_data['date_utc'].dt.hour
    city_data['day_of_week'] = city_data['date_utc'].dt.dayofweek
    city_data['month'] = city_data['date_utc'].dt.month
    
    X = city_data[['hour', 'day_of_week', 'month']]
    y = city_data['value']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_prediction_model(X_train, y_train):
    """Train a simple prediction model."""
    if X_train is None or y_train is None:
        return None
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def generate_future_predictions(model, city, parameter, scaler=None):
    """Generate predictions for the next 7 days."""
    if model is None:
        return pd.DataFrame()
    
    future_dates = pd.date_range(start=datetime.now(), periods=7*24, freq='H')
    future_data = pd.DataFrame({
        'date_utc': future_dates,
        'hour': future_dates.hour,
        'day_of_week': future_dates.dayofweek,
        'month': future_dates.month
    })
    
    X_future = future_data[['hour', 'day_of_week', 'month']]
    predictions = model.predict(X_future)
    
    future_data['predicted_value'] = np.maximum(predictions, 0)  # Ensure no negative predictions
    future_data['city'] = city
    future_data['parameter'] = parameter
    
    return future_data

def visualize_air_quality(df):
    """Creates an interactive map displaying air quality data with predictions."""
    st.title("Global Air Quality Map with Predictions")
    st.write("Showing air pollution levels and predictions from OpenAQ data")
    
    # Sidebar Filters
    st.sidebar.header("Filters")
    country_filter = st.sidebar.multiselect("Select Countries", df["country"].unique())
    parameter_filter = st.sidebar.selectbox("Select Pollutant", df["parameter"].unique())
    
    # Filter data
    filtered_df = df.copy()
    if country_filter:
        filtered_df = filtered_df[filtered_df["country"].isin(country_filter)]
    if parameter_filter:
        filtered_df = filtered_df[filtered_df["parameter"] == parameter_filter]
    
    # Create map
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Add markers to map
    for _, row in filtered_df.iterrows():
        if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]):
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=10,
                popup=f"{row['city']} ({row['country']})<br>{row['parameter'].upper()}: {row['value']:.1f} {row['unit']}<br>Date: {row['date_utc']}",
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(m)
    
    folium_static(m)
    
    # Predictions section
    st.header("Air Quality Predictions")
    if not filtered_df.empty:
        selected_city = st.selectbox("Select City for Predictions", filtered_df["city"].unique())
        
        # Prepare and train model
        X_train, X_test, y_train, y_test = prepare_data_for_prediction(filtered_df, selected_city)
        model = train_prediction_model(X_train, y_train)
        
        if model is not None:
            predictions_df = generate_future_predictions(model, selected_city, parameter_filter)
            
            if not predictions_df.empty:
                st.subheader(f"Predicted {parameter_filter.upper()} levels for {selected_city}")
                fig_data = pd.DataFrame({
                    'Date': predictions_df['date_utc'],
                    'Predicted Value': predictions_df['predicted_value']
                }).set_index('Date')
                
                st.line_chart(fig_data)
                
                # Display accuracy metrics if test data is available
                if X_test is not None and y_test is not None:
                    test_score = model.score(X_test, y_test)
                    st.write(f"Model R² Score: {test_score:.2f}")
        else:
            st.warning(f"Insufficient data to make predictions for {selected_city}")
    
    # Historical trends
    st.header("Historical Trends")
    if not filtered_df.empty:
        filtered_df['date'] = pd.to_datetime(filtered_df['date_utc']).dt.date
        daily_avg = filtered_df.groupby(['date', 'city'])['value'].mean().reset_index()
        
        for city in filtered_df['city'].unique():
            city_data = daily_avg[daily_avg['city'] == city]
            if not city_data.empty:
                st.subheader(f"{city} - Daily Average {parameter_filter.upper()}")
                city_chart_data = pd.DataFrame({
                    'Date': city_data['date'],
                    'Value': city_data['value']
                }).set_index('Date')
                st.line_chart(city_chart_data)

if __name__ == "__main__":
    st.set_page_config(page_title="Air Quality Predictions", page_icon="🌍", layout="wide")
    
    capitals = [
        "Washington, D.C.", "London", "Paris", "Berlin", "Tokyo",
        "Beijing", "Delhi", "Moscow", "Rome", "Madrid"
    ]
    
    # Add a loading spinner while fetching data
    with st.spinner('Fetching air quality data...'):
        df = fetch_air_quality_data(capitals)
    
    if not df.empty:
        visualize_air_quality(df)
    else:
        st.error("No data available. Please check your API key and internet connection.")