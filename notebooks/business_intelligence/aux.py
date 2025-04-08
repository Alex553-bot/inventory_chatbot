import streamlit as st
import pandas as pd
import pydeck as pdk

# List of countries to highlight with their latitudes and longitudes
highlighted_countries = {
    'United States': {'lat': 37.0902, 'lon': -95.7129},
    'Canada': {'lat': 56.1304, 'lon': -106.3468},
    'United Kingdom': {'lat': 51.5074, 'lon': -0.1278},
    'France': {'lat': 46.6034, 'lon': 1.8883},
    'Germany': {'lat': 51.1657, 'lon': 10.4515},
    'Australia': {'lat': -25.2744, 'lon': 133.7751},
    'India': {'lat': 20.5937, 'lon': 78.9629},
    'Japan': {'lat': 36.2048, 'lon': 138.2529},
    'Brazil': {'lat': -14.2350, 'lon': -51.9253},
    'Mexico': {'lat': 23.6345, 'lon': -102.5528}
}

# Convert highlighted countries into a DataFrame for easy processing
data = pd.DataFrame.from_dict(highlighted_countries, orient='index').reset_index()
data.columns = ['Country', 'Latitude', 'Longitude']

# Create a PyDeck map with highlighted markers
deck = pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=20,  # Latitude of the center of the map
        longitude=0,  # Longitude of the center of the map
        zoom=2,       # Zoom level
        pitch=0       # Camera pitch
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=data,
            get_position='[Longitude, Latitude]',
            get_radius=200000,  # Size of the highlighted marker
            get_color='[255, 0, 0, 160]',  # Color for the highlighted countries (red)
            pickable=True,
            auto_highlight=True
        ),
    ],
    tooltip={
        'html': '<b>Country:</b> {Country}',
        'style': {
            'backgroundColor': 'steelblue',
            'color': 'white',
            'fontSize': '16px'
        }
    }
)

# Render the map in Streamlit
st.pydeck_chart(deck)
