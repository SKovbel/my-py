import geopandas as gpd
import matplotlib.pyplot as plt

world_filepath = gpd.datasets.get_path('naturalearth_lowres')
world = gpd.read_file(world_filepath)
world.head()

ax = world.plot(figsize=(20,20), color='whitesmoke', linestyle=':', edgecolor='black')

# zoom
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])

plt.show()

map2 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)
