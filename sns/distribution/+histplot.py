import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# hist - range, bar - categorical
data = pd.DataFrame({
    'y1': np.random.randn(10).cumsum()
}, index=np.arange(0, 10, 1))

sns.histplot(data=data)
plt.show()