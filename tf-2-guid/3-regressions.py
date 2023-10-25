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

chart = True
debug = True
epochs = 100
validation = 0.2 # 20%
learning_rate = 0.1

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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

    return model


### Linear regression with multiple inputs
def multi_liniear_regression(dataset, feature, labels):
    normalizer = normalize_layer(dataset, feature, labels)

    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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

    return model

train_dataset, train_features, train_labels, test_dataset, test_features, test_labels = load_data()
#model = liniear_regression(train_dataset, train_features['Horsepower'], train_labels)
model = multi_liniear_regression(train_dataset, train_features, train_labels)
