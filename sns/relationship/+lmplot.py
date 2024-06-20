import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.plotting.register_matplotlib_converters()

# like scatterplot-3 but with regression lines

cnt = 100

data = pd.DataFrame({
    'idx': np.arange(0, cnt, 1),
    'y1': np.random.randn(cnt),
    'y2': np.random.choice([0, 1], size=100)
})

sns.lmplot(x='idx', y='y1', hue='y2', data=data, markers=['o', 'x'], palette='Set1')


plt.title('Example')
plt.xlabel('Index')
plt.ylabel('y1')
plt.show()
