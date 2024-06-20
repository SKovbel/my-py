import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.plotting.register_matplotlib_converters()

cnt = 100

data = pd.DataFrame({
    'y1': np.random.randn(cnt),
    'y2': np.random.choice([0, 1], size=100)
}, index=np.arange(0, cnt, 1))

sns.scatterplot(x=data.index, y=data['y1'], hue=data['y2'])
plt.show()

