import requests
import pandas as pd
import json
import os
import streamlit as st
import folium
from streamlit_folium import folium_static
from requests.auth import HTTPBasicAuth

def fetch_air_quality_data(capitals, parameter="pm25", limit=100, filename="air_quality_data.csv"):
    """
    Fetch air quality data from OpenAQ API for multiple capital cities and store results.
    If data already exists in a CSV file, load from it instead of making API requests.
    
    Args:
        capitals (list): List of capital cities to fetch data for.
        parameter (str): Pollutant type (e.g., pm25, pm10, co, no2).
        limit (int): Number of records to fetch per city.
        filename (str): File to store/retrieve data.
    
    Returns:
        pd.DataFrame: DataFrame containing air quality data.
    """
    # Check if data file exists
    if os.path.exists(filename):
        print("Loading data from CSV file...")
        return pd.read_csv(filename)
    
    records = []
    for city in capitals:
        url = f"https://api.openaq.org/v3/measurements?city={city}&parameter={parameter}&limit={limit}&format=json"
        headers = {'X-API-Key': st.secrets["AQ_API"]}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            for item in results:
                records.append({
                    "city": item.get("city"),
                    "country": item.get("country"),
                    "location": item.get("location"),
                    "parameter": item.get("parameter"),
                    "value": item.get("value"),
                    "unit": item.get("unit"),
                    "latitude": item.get("coordinates", {}).get("latitude"),
                    "longitude": item.get("coordinates", {}).get("longitude"),
                    "date_utc": item.get("date", {}).get("utc"),
                })
        else:
            print(f"Failed to fetch data for {city}: {response.status_code}")
    
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print("Data saved to CSV file.")
    return df

def visualize_air_quality(df):
    """Creates an interactive map displaying air quality data."""
    st.title("Global Air Quality Map")
    st.write("Showing air pollution levels from OpenAQ API")
    
    # Initialize map
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    for _, row in df.iterrows():
        if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]):
            folium.Marker(
                [row["latitude"], row["longitude"]],
                popup=f"{row['city']} ({row['country']})\n{row['parameter'].upper()}: {row['value']} {row['unit']}\nDate: {row['date_utc']}",
                tooltip=f"{row['city']} - {row['parameter'].upper()} {row['value']} {row['unit']}"
            ).add_to(m)
    
    folium_static(m)

# Streamlit App
if __name__ == "__main__":
    capitals = ["Washington, D.C.", "London", "Paris", "Berlin", "Tokyo"]
    df = fetch_air_quality_data(capitals)
    visualize_air_quality(df)
