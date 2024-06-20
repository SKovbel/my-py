import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cnt=10
# hist - range, bar - categorical
data = pd.DataFrame({
    'x': np.arange(0, cnt, 1),
    'y': np.random.randn(cnt).cumsum(),
    'z': np.random.choice(['A', 'B', 'C'], size=cnt)
})

sns.histplot(data=data, x='y', hue='z')
plt.show()