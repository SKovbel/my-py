import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import webbrowser

my_map = folium.Map(location=[42.32,-71.0589], tiles='openstreetmap', zoom_start=10)

Marker((42.30, -71.10)).add_to(my_map)
Circle(location=(42.10, -71.30), radius=20, color='forestgreen').add_to(my_map)

mc = MarkerCluster()
mc.add_child(Marker((42.30607218, -71.08273260)))
mc.add_child(Marker((42.35779134, -71.13937053)))
mc.add_child(Circle(location=(42.30607218, -71.13937053), radius=20, color='darkred'))

my_map.add_child(mc)

# Save the map to an HTML file
html_file_path = 'my_map.html'

HeatMap(data=[(42.30607218, -71.08273260), (42.20607218, -71.08273260), (42.30607218, -71.02273260)], radius=30).add_to(my_map)

#Choropleth(geo_data=}...., 
#key_on="feature.id", 
#fill_color='YlGnBu', 
#legend_name='Major criminal incidents (Jan-Aug 2018)'
#).add_to(my_map)


my_map.save(html_file_path)

# Open the HTML file in the default web browser
webbrowser.open(html_file_path)
