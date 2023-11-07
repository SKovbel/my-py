import numpy as np
from sklearn.linear_model import LinearRegression

def desired_marketing_expenditure(marketing_expenditure, units_sold, desired_units_sold):
    """
    :param marketing_expenditure: (list) A list of integers with the expenditure for each previous campaign.
    :param units_sold: (list) A list of integers with the number of units sold for each previous campaign.
    :param desired_units_sold: (integer) Target number of units to sell in the new campaign.
    :returns: (float) Required amount of money to be invested.
    """

    # Create a NumPy array from the provided data
    X = np.array(marketing_expenditure).reshape(-1, 1)
    y = np.array(units_sold)

    print(marketing_expenditure, X)
    print(units_sold, y)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)

    print(model.intercept_,  model.coef_[0])
    # Predict the marketing expenditure required for the desired units sold
    required_expenditure = (desired_units_sold - model.intercept_) / model.coef_[0]

    return required_expenditure

# For example, with the parameters below, the function should return 250000.0
print(desired_marketing_expenditure(
    [300000, 200000, 400000, 300000, 100000],
    [60000, 50000, 90000, 80000, 30000],
    60000))
print(desired_marketing_expenditure(
    [300000, 200000, 400000, 300000, 100000],
    [60000, 50000, 90000, 80000, 30000],
    110000))