import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

path=os.path.join(os.path.dirname(__file__),  "purple_martin.csv")

birds_df = pd.read_csv(path, parse_dates=['timestamp'])
birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df["location-long"], birds_df["location-lat"]))

# GeoDataFrame showing path for each bird
path_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: LineString(x)).reset_index()
path_gdf = gpd.GeoDataFrame(path_df, geometry=path_df.geometry)
path_gdf.crs = {'init' :'epsg:4326'}

# GeoDataFrame showing starting point for each bird
start_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[0]).reset_index()
start_gdf = gpd.GeoDataFrame(start_df, geometry=start_df.geometry)
start_gdf.crs = {'init' :'epsg:4326'}

# GeoDataFrame showing ending point for each bird
end_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[-1]).reset_index()
end_gdf = gpd.GeoDataFrame(end_df, geometry=end_df.geometry)
end_gdf.crs = {'init': 'epsg:4326'}


world_filepath = gpd.datasets.get_path('naturalearth_lowres')
world = gpd.read_file(world_filepath)
world.head()

#ax = world.plot(figsize=(20,20), color='whitesmoke', linestyle=':', edgecolor='black')
#plt.show()

america = world.loc[world['continent'].isin(['North America', 'South America'])]
south_america = world.loc[world['continent'].isin(['South America'])]
ax = south_america.plot(figsize=(20,20), color='whitesmoke', linestyle=':', edgecolor='black')
birds.plot(ax=ax, markersize=10)
start_gdf.plot(ax=ax, markersize=10)
end_gdf.plot(ax=ax, markersize=10)
path_gdf.plot(ax=ax, markersize=10)

protected_areas[protected_areas['MARINE']!='2'].plot(ax=ax, alpha=0.4, zorder=1)
birds[birds.geometry.y < 0].plot(ax=ax, color='red', alpha=0.6, markersize=10, zorder=2)

# zoom
ax.set_xlim([-110, -30])
ax.set_ylim([-30, 60])
plt.show()


totalArea = sum(south_america.geometry.to_crs(epsg=3035).area) / 10**6
print(totalArea)