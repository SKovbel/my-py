'''
MEstimateEncoder is not a built-in encoder in scikit-learn or pandas as of tictactoe last knowledge update in January 2022.
However, there is an encoder called TargetEncoder that is commonly used for encoding categorical variables based on the mean 
    of the target variable.
pip install category_encoders
'''

import pandas as pd
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a sample DataFrame
data = {'Category': ['A', 'B', 'A', 'B', 'A', 'C'],
        'Target': [1, 0, 1, 0, 1, 2]}

df = pd.DataFrame(data)
X = df.Category
y = df.Target

# Creating a TargetEncoder instance with M-Estimate
encoder = ce.MEstimateEncoder(cols=['Category'], m=2.0)

# Fitting and transforming the encoder on the DataFrame
encoder.fit(X, y)
X_train = encoder.transform(X)

print(df)
print(X_train)

plt.figure(dpi=90)
ax = sns.distplot(y, kde=False, norm_hist=True)
ax = sns.kdeplot(X_train.Category, color='r', ax=ax)
ax.set_xlabel("Target")
ax.legend(labels=['Category', 'Target']);
plt.show()