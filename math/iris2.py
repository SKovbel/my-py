import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import norm



iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
labels = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, column in enumerate(iris.feature_names):
    short = column.replace(" (cm)", "")
    stats = iris_df.groupby('species')[column].agg(['mean', 'std']).reset_index()
    stats.columns = ['species', 'mean', 'std']

    std = np.std(iris_df[column])
    mean = np.mean(iris_df[column])
    xspace = np.linspace(mean - 3 * std, mean + 3 * std, len(iris_df))

    pdf = norm.pdf(xspace, mean, std)
    marginal_pdf = np.zeros_like(xspace)
    codition_pdfs = {}
    for index, row in stats.iterrows():
        codition_pdf = norm.pdf(xspace, row['mean'], row['std'])
        marginal_pdf += codition_pdf * (len(iris_df[iris_df['species'] == index]) / len(iris_df))
        codition_pdfs[labels[row['species']]] = codition_pdf

    axs[i].plot(xspace, pdf, color='red', label=f'PDF({short})', linewidth=3)
    axs[i].plot(xspace, marginal_pdf, color='blue', label=f'Marginal({short})', linewidth=2)
    for label in codition_pdfs:
        axs[i].plot(xspace, codition_pdfs[label], label=f"Cond({short}|{label})", linewidth=1)
    #axs[i].set_title(short)
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Density')
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.show()
