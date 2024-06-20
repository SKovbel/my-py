import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# hist - range, bar - categorical
# kde - you can think of it as a smoothed histogram.

data = pd.DataFrame({
    'y1': np.random.randn(10).cumsum()
}, index=np.arange(0, 10, 1))

sns.kdeplot(data=data, shade=True)
plt.show()