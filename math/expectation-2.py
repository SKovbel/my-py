import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sum_counts = np.zeros(2*6+1)
for die1 in range(1, 6+1):
    for die2 in range(1, 6+1):
        sum_counts[die1 + die2] += 1
total_outcomes = 6 ** 2
probabilities = sum_counts / total_outcomes

expected_value = np.sum(np.arange(len(probabilities)) * probabilities)
df = pd.DataFrame({
    'Sum': np.arange(0, 2*6 + 1),
    'Probability': probabilities
})

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(df['Sum'], df['Probability'], color='skyblue')
plt.xlabel('Sum of Dice')
plt.axvline(x=expected_value, color='red', linestyle='--', label=f'Expected Value = {expected_value:.2f}')
plt.ylabel('Probability')
plt.title('Probability Distribution of the Sum of Two Dice')
plt.xticks(df['Sum'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
