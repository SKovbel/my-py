from sklearn.metrics import confusion_matrix

y_test = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]

print(confusion_matrix(y_test, y_pred))


# The diagonal elements represent the number of correctly classified instances for each class.
# Off-diagonal elements represent misclassifications.
# Rows represent the actual classes, and columns represent the predicted classes.