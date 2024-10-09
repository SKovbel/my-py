import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import gaussian_kde

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
labels = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Create a 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, column in enumerate(iris.feature_names):
    short = column.replace(" (cm)", "")
    
    # Overall KDE
    overall_kde = gaussian_kde(iris_df[column])
    xspace = np.linspace(iris_df[column].min(), iris_df[column].max(), len(iris_df))
    pdf = overall_kde(xspace)

    # Marginal PDF and conditional PDFs
    marginal_pdf = np.zeros_like(xspace)
    condition_pdfs = {}

    stats = iris_df.groupby('species')[column].agg(['mean', 'std']).reset_index()
    stats.columns = ['species', 'mean', 'std']

    for index, row in stats.iterrows():
        # KDE for each species
        species_data = iris_df[iris_df['species'] == index][column]
        condition_kde = gaussian_kde(species_data)
        condition_pdfs[labels[row['species']]] = condition_kde(xspace)
        
        # Update marginal PDF based on the species proportion
        marginal_pdf += condition_pdfs[labels[row['species']]] * (len(species_data) / len(iris_df))

    # Plotting
    axs[i].plot(xspace, pdf, color='red', label=f'Global KDE({short})', linewidth=3)
    axs[i].plot(xspace, marginal_pdf, color='blue', label=f'Marginal({short})', linewidth=2)
    
    for label in condition_pdfs:
        axs[i].plot(xspace, condition_pdfs[label], label=f"Cond({short}|{label})", linewidth=1)

    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Density')
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.show()
