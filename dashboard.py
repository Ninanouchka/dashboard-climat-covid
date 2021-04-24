import streamlit as st
import pandas as pd
from datetime import datetime
import folium
import re
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
from dms2dec.dms_convert import dms2dec
import pydeck as pdk
import plotly.express as px


def geoloc_convert(loc_degree):
    """ Convert degrees minutes seconds coordinate to decimal latitude or longitude  """
    loc_decimal = dms2dec(loc_degree)
    if loc_degree[-1] == 'O':
        loc_decimal = -loc_decimal
    return loc_decimal

def scatter_map(df):
    """ Build scatter map with color gradient for iptcc """
    mark_size = [100 for i in df.index]
    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_data=["iptcc"],
                    color="iptcc", color_continuous_scale=['#7BD150', '#F6E626', '#FC9129', '#FF1B00', '#6E1E80'], size=mark_size, size_max=10, zoom=4, height=450)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@st.cache
def load_data():
    """ Load the cleaned data with latitudes, longitudes & timestamps """
    df = pd.read_csv("IPTCC-20210423-153416.csv", sep="|", parse_dates=['DATE'])
    df.columns = [col_name.lower() for col_name in df.columns]
    df['latitude'] = df['latitude'].apply(geoloc_convert)
    df['longitude'] = df['longitude'].apply(geoloc_convert)
    df['iptcc'] = df['iptcc'].str.replace(',', '.').astype(float)
    return df


st.title("Meteo Covid - IPTCC")

# Load the dataset
df = load_data()

# Calculate the timerange for the slider
min_ts = min(df["date"]).to_pydatetime()
max_ts = max(df["date"]).to_pydatetime()

#slider to chose date
st.sidebar.subheader("Inputs")
day_date = pd.to_datetime(st.sidebar.slider("Date to chose", min_value=min_ts, max_value=max_ts, value=max_ts))

# day = st.sidebar.text_input("Day", value='22')
# month = st.sidebar.text_input("Month", value='04')
# year = st.sidebar.text_input("Year", value='2021')
show_heatmap = st.sidebar.checkbox("Show date range")

if show_heatmap:
    min_selection, max_selection = st.sidebar.slider("Timeline", min_value=min_ts, max_value=max_ts, value=[min_ts, max_ts])

    # Filter data for timeframe
    st.write(f"Filtering between {min_selection.date()} & {max_selection.date()}")
    df = df[(df["date"] >= min_selection) & (df["date"] <= max_selection)].groupby('station').mean()[['iptcc', 'latitude', 'longitude']]
    st.write(f"Stations: {len(df)}")

else:
    
    # Get last day data 
#     day_date = pd.to_datetime(year + month + day, format='%Y%m%d')
    st.write(f"Data for {day_date.date()}")
    df = df[(df["date"] == day_date)]
    st.write(f"Data Points: {len(df)}")

# Plot the stations on the map
# st.map(df)
st.plotly_chart(scatter_map(df), use_container_width=True)