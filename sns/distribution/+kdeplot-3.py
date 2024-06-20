import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# hist - range, bar - categorical
# kde - you can think of it as a smoothed histogram.

cnt=10
data = pd.DataFrame({
    'x': np.arange(0, cnt, 1),
    'y': np.random.randn(cnt).cumsum(),
    'z': np.random.choice(['A', 'B', 'C'], size=cnt)
})

sns.kdeplot(data=data, shade=True, x='x', hue='z')
plt.show()