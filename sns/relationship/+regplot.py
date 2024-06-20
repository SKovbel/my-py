import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.plotting.register_matplotlib_converters()

# like scatterplot but with regression lines

cnt = 100

data = pd.DataFrame({
    'y': np.random.randn(cnt),
}, index=np.arange(0, cnt, 1))

sns.regplot(x=data.index, y=data['y'])
plt.show()
