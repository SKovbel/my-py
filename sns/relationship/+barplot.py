import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.plotting.register_matplotlib_converters()
# hist - range, bar - categorical

data = pd.DataFrame({
    'y1': 2 + np.random.randn(10).cumsum(),
    'y2': 2 + np.random.randn(10).cumsum(),
    'category': np.repeat(['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5'], 2)
}, index=np.arange(0, 10, 1))

sns.barplot(x=data['category'], y=data['y1'])
plt.show()
