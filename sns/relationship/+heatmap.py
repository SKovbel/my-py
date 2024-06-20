import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.plotting.register_matplotlib_converters()

data = pd.DataFrame({
    'y1': 2 + np.random.randn(10).cumsum(),
    'y2': 2 + np.random.randn(10).cumsum(),
    'y3': 2 + np.random.randn(10).cumsum(),
    'y4': 2 + np.random.randn(10).cumsum(),
    'y5': 2 + np.random.randn(10).cumsum()
}, index=np.arange(0, 10, 1))

sns.heatmap(data=data, annot=True)
plt.show()
