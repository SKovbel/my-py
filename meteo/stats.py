from meteostat import Stations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


Stations.cache_dir = './cache'

stations = Stations()

# Get the list of stations
station_list = stations.fetch()
station_list['daily_start'] = pd.to_datetime(station_list['daily_start'])
station_list['daily_start_year'] = station_list['daily_start'].dt.year


stations_year = station_list.groupby('daily_start_year').size().reset_index(name='count')

# Print the list of stations
print(station_list)
print(station_list.columns)

plt.xticks(range(int(stations_year['daily_start_year'].min()), int(stations_year['daily_start_year'].max()) + 1, 10))
sns.scatterplot(x=stations_year['daily_start_year'], y=stations_year['count'])
plt.show()
