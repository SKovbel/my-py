import pandas as pd
import geopandas as gpd
#full_data = gpd.read_file("../input/geospatial-learn-course-data/DEC_lands/DEC_lands/DEC_lands.shp")

print(gpd.datasets.available)

# geopandas.datasets.get_path("naturalearth_lowres")  '.../python3.8/site-packages/geopandas/datasets/naturalearth_lowres/naturalearth_lowres.shp'
# gpd.GeoDataFrame

birds_df = pd.read_csv("../input/geospatial-learn-course-data/purple_martin.csv", parse_dates=['timestamp'])
birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df["location-long"], birds_df["location-lat"]))
birds.crs = {'init': 'epsg:4326'}