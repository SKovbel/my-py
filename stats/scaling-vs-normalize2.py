# modules we'll use
import numpy as np
import pandas as pd
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
data = np.random.normal(size=1000)
df = pd.DataFrame(data, columns=['value'])
index_of_positive = df.value > 0

original_data = df.value.loc[index_of_positive]
scaled_data = minmax_scaling(original_data.to_numpy(), columns=[0])
normalized_data = stats.boxcox(original_data)

fig, ax = plt.subplots(1, 3, figsize=(15, 3))

ax[0].set_title("Original Data")
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)

ax[1].set_title("Scaled data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)

ax[2].set_title("Normalized data")
sns.histplot(normalized_data, ax=ax[2], kde=True, legend=False)

plt.show()