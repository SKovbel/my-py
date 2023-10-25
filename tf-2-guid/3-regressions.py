import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses

from dexa.chart import train

print(tf.__version__)

chart = False
debug = True
epochs = 100
validation = 0.2 # 20%
save_file = os.path.join(os.path.dirname(__file__), '../var/model.keras')

## The Auto MPG dataset
def load_data():
    ### Get the data
    raw_dataset = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',
        names=['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin'],
        na_values='?',
        comment='\t',
        sep=' ',
        skipinitialspace=True
    )
    dataset = raw_dataset.copy()

    ### Clean the data
    dataset.isna().sum()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='', dtype='float32')
    return dataset

def preprocess_data(dataset):
    ### Split the data into training and test sets
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    ### Split features from labels
    train_features = train_dataset.copy()
    train_labels = train_features.pop('MPG')

    test_features = test_dataset.copy()
    test_labels = test_features.pop('MPG')

    ### Inspect the data
    if chart:
        sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
        plt.show()

    if debug:
        print(train_dataset.describe().transpose())

    return train_dataset, train_features, train_labels, test_dataset, test_features, test_labels

## Normalization
def normalize_layer(dataset, features, labels):
    ### The Normalization layer
    normalizer = layers.Normalization(name="our_normalization", axis=-1)
    normalizer.adapt(np.array(features).astype(np.float32))

    if debug:
        print("\nlabels:\n", labels.describe()[['count', 'mean', 'std']], sep="")
        print("\nDataset:\n", dataset.describe().transpose(), sep="")
        #print("\nNormalizer meaans:\n", normalizer.mean.numpy(), sep="")
        first = np.array(features[:1])
        with np.printoptions(precision=2, suppress=True):
            print()
            print('First example:', first)
            print()
            #?print('Normalized:', normalizer(first).numpy())

    return normalizer

## Linear regression
### Linear regression with one variable
def liniear_regression(dataset, feature, labels):
    data = np.array(feature).astype(np.float32)

    normalizer = layers.Normalization(input_shape=[1,], axis=None)
    normalizer.adapt(data)

    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss=losses.MeanAbsoluteError(), #'mean_absolute_error'
        metrics=['accuracy']
    )

    result = model.fit(
        feature,
        labels,
        epochs=epochs,
        validation_split = validation, # Calculate validation results on 20% of the training data.
        verbose=2 if debug else 1
    )

    if debug:
        model.summary()

    if chart:
        sns.displot(labels, kde=True)
        plt.show()

        fit_chart = train.FitChart()
        fit_chart.chart(model, result)
        fit_chart.print(model, result)

    return model, result


### Linear regression with multiple inputs
def multi_liniear_regression(dataset, feature, labels):
    normalizer = normalize_layer(dataset, feature, labels)

    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss=losses.MeanAbsoluteError(), #'mean_absolute_error'
        metrics=['accuracy']
    )

    result = model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        validation_split = validation,
        verbose=2 if debug else 1
    )

    if debug:
        model.summary()

    if chart:
        sns.displot(labels, kde=True)
        plt.show()

        fit_chart = train.FitChart()
        fit_chart.chart(model, result)
        fit_chart.print(model, result)

    return model, result


## Regression with a deep neural network (DNN)
### Regression using a DNN and a single input
def dnn_liniear_regression(dataset, feature, labels):
    data = np.array(feature).astype(np.float32)

    normalizer = layers.Normalization(input_shape=[1,], axis=None)
    normalizer.adapt(data)

    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(units=1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=losses.MeanAbsoluteError(), #'mean_absolute_error'
        metrics=['accuracy']
    )

    result = model.fit(
        feature,
        labels,
        epochs=epochs,
        validation_split = validation, # Calculate validation results on 20% of the training data.
        verbose=2 if debug else 1
    )

    if debug:
        model.summary()

    if chart:
        sns.displot(labels, kde=True)
        plt.show()

        fit_chart = train.FitChart()
        fit_chart.chart(model, result)
        fit_chart.print(model, result)

    return model, result

### Regression using a DNN and multiple inputs
def ddn_multi_liniear_regression(dataset, feature, labels):
    normalizer = normalize_layer(dataset, feature, labels)

    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(units=1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=losses.MeanAbsoluteError(), #'mean_absolute_error'
        metrics=['accuracy']
    )

    result = model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        validation_split = validation,
        verbose=2 if debug else 1
    )

    if debug:
        model.summary()

    if chart:
        sns.displot(labels, kde=True)
        plt.show()

        fit_chart = train.FitChart()
        fit_chart.chart(model, result)

    return model, result


dataset = load_data()
train_dataset, train_features, train_labels, test_dataset, test_features, test_labels = preprocess_data(dataset)
features = test_features

case=3
if case==0:
    model, fit_result = multi_liniear_regression(train_dataset, train_features, train_labels)
elif case==1:
    model, fit_result = ddn_multi_liniear_regression(train_dataset, train_features, train_labels)
elif case==2:
    model, fit_result = liniear_regression(train_dataset, train_features['Horsepower'], train_labels)
    test_features = test_features['Horsepower']
else:
    model, fit_result = dnn_liniear_regression(train_dataset, train_features['Horsepower'], train_labels)
    test_features = test_features['Horsepower']

## Performance
evaluate = model.evaluate(
    test_features,
    test_labels,
    verbose=2 if debug else 1
)

print(pd.DataFrame([evaluate], index=['Mean absolute error [MPG]']).T)

test_predictions = model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()

model.save(save_file)
#model = tf.keras.models.load_model(save_file)

evaluate = model.evaluate(
    test_features, test_labels, verbose=0)

print(pd.DataFrame([evaluate], index=['Mean absolute error [MPG]']).T)

