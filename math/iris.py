import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import norm



iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
labels = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

column = 'petal length (cm)'
stats = iris_df.groupby('species')[column].agg(['mean', 'std']).reset_index()
stats.columns = ['species', 'mean', 'std']

std = np.std(iris_df[column])
mean = np.mean(iris_df[column])
xspace = np.linspace(mean - 3 * std, mean + 3 * std, 100)

pdf = norm.pdf(xspace, mean, std)
marginal_pdf = np.zeros_like(xspace)
codition_pdfs = {}
for index, row in stats.iterrows():
    codition_pdf = norm.pdf(xspace, row['mean'], row['std'])
    marginal_pdf += codition_pdf * (len(iris_df[iris_df['species'] == index]) / len(iris_df))
    codition_pdfs[labels[row['species']]] = codition_pdf


plt.figure(figsize=(10, 6))
plt.plot(xspace, pdf, color='red', label='Global PDF(pental length)')
for label in codition_pdfs:
    plt.plot(xspace, codition_pdfs[label], label=f"Conditional  PDF(pental length|{label})")
plt.plot(xspace, marginal_pdf, color='blue', linestyle='--', label='Marginal PDF(pental length)', linewidth=2)
plt.title('Gaussian Distributions for Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()
