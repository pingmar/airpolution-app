import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import streamlit as st
import folium
from streamlit_folium import folium_static

def fetch_locations_by_parameter(parameter_id=2, limit=1000):
    """
    Fetch locations that measure a specific parameter (default: PM2.5).
    
    Args:
        parameter_id (int): Parameter ID (2 for PM2.5)
        limit (int): Number of results per page
    """
    url = f"https://api.openaq.org/v3/locations"
    params = {
        "parameters_id": parameter_id,
        "limit": limit
    }
    headers = {'X-API-Key': st.secrets["AQ_API"]}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get("results", []))
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch locations: {str(e)}")
        return pd.DataFrame()

def fetch_latest_measurements(parameter_id=2, limit=1000):
    """
    Fetch latest measurements for a specific parameter (default: PM2.5).
    
    Args:
        parameter_id (int): Parameter ID (2 for PM2.5)
        limit (int): Number of results per page
    """
    url = f"https://api.openaq.org/v3/parameters/{parameter_id}/latest"
    params = {"limit": limit}
    headers = {'X-API-Key': st.secrets["AQ_API"]}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get("results", []))
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch measurements: {str(e)}")
        return pd.DataFrame()

def fetch_measurements_by_coordinates(lat, lon, radius=12000, limit=1000):
    """
    Fetch locations within a radius of specified coordinates.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        radius (int): Radius in meters
        limit (int): Number of results per page
    """
    url = "https://api.openaq.org/v3/locations"
    params = {
        "coordinates": f"{lon},{lat}",
        "radius": radius,
        "limit": limit
    }
    headers = {'X-API-Key': st.secrets["AQ_API"]}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get("results", []))
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch locations by coordinates: {str(e)}")
        return pd.DataFrame()

def fetch_sensor_measurements(sensor_id, limit=1000):
    """
    Fetch measurements for a specific sensor.
    
    Args:
        sensor_id (int): Sensor ID
        limit (int): Number of results per page
    """
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
    params = {"limit": limit}
    headers = {'X-API-Key': st.secrets["AQ_API"]}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get("results", []))
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch sensor measurements: {str(e)}")
        return pd.DataFrame()

def create_map(df):
    """Create a Folium map with location markers."""
    if df.empty:
        return None
    
    # Calculate center of map from data
    center_lat = df['coordinates.latitude'].mean()
    center_lon = df['coordinates.longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    
    for _, row in df.iterrows():
        try:
            lat = row['coordinates.latitude']
            lon = row['coordinates.longitude']
            name = row.get('name', 'Unknown Location')
            value = row.get('value', 'No data')
            unit = row.get('unit', '')
            
            popup_text = f"""
                <b>{name}</b><br>
                Value: {value} {unit}<br>
                Lat: {lat}<br>
                Lon: {lon}
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=popup_text,
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(m)
        except (TypeError, ValueError) as e:
            continue
    
    return m

def main():
    st.set_page_config(page_title="Air Quality Monitor", page_icon="🌍", layout="wide")
    st.title("Air Quality Monitoring Dashboard")
    
    # Sidebar for search options
    st.sidebar.title("Search Options")
    search_type = st.sidebar.radio(
        "Search Type",
        ["Latest PM2.5 Measurements", "Search by Location", "Specific Sensor"]
    )
    
    if search_type == "Latest PM2.5 Measurements":
        with st.spinner('Fetching latest PM2.5 measurements...'):
            df = fetch_latest_measurements()
            if not df.empty:
                st.success(f"Found {len(df)} locations with PM2.5 measurements")
                m = create_map(df)
                if m:
                    folium_static(m)
                
                # Show data table
                st.subheader("Latest Measurements")
                st.dataframe(df)
    
    elif search_type == "Search by Location":
        col1, col2 = st.sidebar.columns(2)
        lat = col1.number_input("Latitude", value=35.14942)
        lon = col2.number_input("Longitude", value=136.90610)
        radius = st.sidebar.slider("Radius (meters)", 1000, 50000, 12000)
        
        if st.sidebar.button("Search"):
            with st.spinner('Searching locations...'):
                df = fetch_measurements_by_coordinates(lat, lon, radius)
                if not df.empty:
                    st.success(f"Found {len(df)} locations within {radius}m radius")
                    m = create_map(df)
                    if m:
                        folium_static(m)
                    
                    # Show data table
                    st.subheader("Location Details")
                    st.dataframe(df)
    
    elif search_type == "Specific Sensor":
        sensor_id = st.sidebar.number_input("Sensor ID", value=3917, min_value=1)
        if st.sidebar.button("Fetch Data"):
            with st.spinner('Fetching sensor measurements...'):
                df = fetch_sensor_measurements(sensor_id)
                if not df.empty:
                    st.success(f"Found {len(df)} measurements for sensor {sensor_id}")
                    
                    # Show time series plot
                    st.subheader("Measurement Time Series")
                    chart_data = df[['datetime', 'value']].copy()
                    chart_data['datetime'] = pd.to_datetime(chart_data['datetime'])
                    chart_data = chart_data.set_index('datetime')
                    st.line_chart(chart_data)
                    
                    # Show data table
                    st.subheader("Raw Measurements")
                    st.dataframe(df)

if __name__ == "__main__":
    main()