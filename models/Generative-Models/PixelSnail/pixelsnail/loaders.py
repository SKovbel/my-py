import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_gemstone_data(preprocess, data_dir, batch_size=32):
    datagen = ImageDataGenerator(preprocessing_function=preprocess)

    datagen_train = datagen.flow_from_directory(
        f"{data_dir}/train/",
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode=None,
        shuffle=True)

    datagen_test = datagen.flow_from_directory(
        f"{data_dir}/test/",
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode=None,
        shuffle=True)

    x_train = np.zeros((0, 32, 32, 3))
    for data in datagen_train:
        if datagen_train.total_batches_seen > len(datagen_train):
            break
        x_train = np.concatenate([x_train, data])

    print(x_train.max(), x_train.min())

    x_test = np.zeros((0, 32, 32, 3))
    for data in datagen_test:
        if datagen_test.total_batches_seen > len(datagen_test):
            break
        x_test = np.concatenate([x_test, data])

    cap = x_train.shape[0] // batch_size
    x_train = x_train[:cap * batch_size, ...]

    cap = x_test.shape[0] // batch_size
    x_test = x_test[:cap * batch_size, ...]

    datagen = ImageDataGenerator()
    datagen_train = datagen.flow(
        x_train.astype(float), x_train.astype(float),
        #save_to_dir='tmp/gem_gen',
        batch_size=batch_size)

    datagen_test = datagen.flow(
        x_test.astype(float), x_test.astype(float),
        #save_to_dir='tmp/gem_gen',
        batch_size=batch_size)

    return datagen_train, datagen_test
