from sklearn.model_selection import train_test_split


X = [1,2,3,4,5,6,7,8,10]
y = [1,2,3,4,5,6,7,8,10]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

print(X_train)
print(y_train)

print(X_test)
print(y_test)