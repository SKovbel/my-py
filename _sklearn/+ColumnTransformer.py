'''
ColumnTransformer is a powerful tool in scikit-learn that allows you to apply different transformers
 to different columns of your dataset. This is especially useful when you have a dataset
   with a mix of numerical and categorical features that require different preprocessing steps.
'''

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a sample dataset
data = {
    'numeric_feature': [1, 2, None, 4, 5],
    'categorical_feature': ['A', 'B', 'A', 'B', 'B'],
    'target': [0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Define the columns and transformers
numeric_features = ['numeric_feature']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['categorical_feature']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create the final pipeline with preprocessing and RandomForestClassifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', RandomForestClassifier(random_state=42))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')