import os
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import webbrowser
import pandas as pd
import geopandas as gpd

plate_boundaries = gpd.read_file(os.path.join(os.path.dirname(__file__), "data/TectonicPlateBoundaries.shp"))
plate_boundaries['coordinates'] = plate_boundaries.apply(lambda x: [(b,a) for (a,b) in list(x.geometry.coords)], axis='columns')
plate_boundaries.drop('geometry', axis=1, inplace=True)

countries_boundaries = gpd.read_file(os.path.join(os.path.dirname(__file__), "data/countries.shp"))

earthquakes = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/earthquakes1970-2014.csv"), parse_dates=["DateTime"])

def color_producer(magnitude):
    if magnitude > 6.5:
        return 'red'
    else:
        return 'green'

my_map = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)

for i in range(len(plate_boundaries)):
    folium.PolyLine(locations=plate_boundaries.coordinates.iloc[i], weight=2, color='black').add_to(my_map)

for idx, row in countries_boundaries.iterrows():
    folium.GeoJson(row['geometry']).add_to(my_map)

for i in range(0,len(earthquakes)):
    folium.Circle(
        location=[earthquakes.iloc[i]['Latitude'], earthquakes.iloc[i]['Longitude']],
        popup=("{} ({})").format(
            earthquakes.iloc[i]['Magnitude'],
            earthquakes.iloc[i]['DateTime'].year),
        radius=earthquakes.iloc[i]['Magnitude']**5.5,
        color=color_producer(earthquakes.iloc[i]['Magnitude'])).add_to(my_map)

html_file_path = 'my_map.html'
my_map.save(html_file_path)

# Open the HTML file in the default web browser
webbrowser.open(html_file_path)
