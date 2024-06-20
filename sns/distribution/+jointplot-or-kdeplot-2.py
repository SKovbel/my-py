import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# hist - range, bar - categorical
# kde - you can think of it as a smoothed histogram.

cnt=10
data = pd.DataFrame({
    'x': np.arange(0, cnt, 1),
    'y1': np.random.randn(cnt),
    'y2': np.random.randn(cnt)
})

sns.jointplot(x=data['y1'], y=data['y2'], kind='kde')
plt.show()