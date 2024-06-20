import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.DataFrame({
    'y1': np.random.randn(10).cumsum(),
    'y2': np.random.randn(10).cumsum(),
    'category': np.repeat(['Line 1', 'Line 2'], 5)
}, index=np.arange(0, 10, 1))

sns.lineplot(data=data)
plt.show()