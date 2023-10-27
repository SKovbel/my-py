import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

print(tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    model.summary()

    return model

class CustomCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # This method is called at the beginning of each epoch
        print(f"Starting epoch {epoch}")

    def on_epoch_end(self, epoch, logs=None):
        # This method is called at the end of each epoch
        print(f"Finished epoch {epoch}")

    def on_train_begin(self, logs=None):
        # This method is called at the beginning of training
        print("Training is starting.")

    def on_train_end(self, logs=None):
        # This method is called at the end of training
        print("Training is complete.")

##
# example 1 - save & load after each epoch to single file
##
checkpoint_path = os.path.join(os.path.dirname(__file__), '../tmp/save-load/example-1/cp.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)

model = create_model()

save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

model.fit(
    train_images, 
    train_labels,  
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[save_callback, CustomCallback()]
)

model = create_model()
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model1, accuracy: {:5.2f}%".format(100 * acc))

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model1, accuracy: {:5.2f}%".format(100 * acc))

##
# example 2 - save every 5 epochs to separate file
##
checkpoint_path = os.path.join(os.path.dirname(__file__), '../tmp/save-load/example-2/cp-{epoch:04d}.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

n_batches = len(train_images) / batch_size
n_batches = math.ceil(n_batches)    # round up the number of batches to the nearest whole integer

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path, 
    verbose = 1, 
    save_weights_only = True,
    save_freq = 5 * n_batches
)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))

model.fit(
    train_images, 
    train_labels,
    epochs=50, 
    batch_size=batch_size, 
    callbacks=[cp_callback],
    validation_data=(test_images, test_labels),
    verbose=0
)

model = create_model()

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model2, accuracy: {:5.2f}%".format(100 * acc))

##
# example 3 - save manually
##
checkpoint_path = os.path.join(os.path.dirname(__file__), '../tmp/save-load/example-3/my_checkpoint')

model.save_weights(checkpoint_path)

model = create_model()

model.load_weights(checkpoint_path)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model3, accuracy: {:5.2f}%".format(100 * acc))


##
# example 4 - save full model
##
checkpoint_path = os.path.join(os.path.dirname(__file__), '../tmp/save-load/example-4/cp-{epoch:04d}.ckpt')

model = create_model()
model.fit(train_images, train_labels, epochs=5)

model.save(checkpoint_path)

new_model = tf.keras.models.load_model(checkpoint_path)

new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)

##
# example 5 - save format
##
checkpoint_path = os.path.join(os.path.dirname(__file__), '../tmp/save-load/example-5/model.save')
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save(checkpoint_path)

new_model = tf.keras.models.load_model(checkpoint_path)
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)

##
# example 6 - save in HDF5
##
checkpoint_path = os.path.join(os.path.dirname(__file__), '../tmp/save-load/example-6/model.h5')

model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save(checkpoint_path)

new_model = tf.keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


##
# example 7 - save custom object
##
'''
https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing

Define a get_config method in your object, and optionally a from_config classmethod.
get_config(self) returns a JSON-serializable dictionary of parameters needed to recreate the object.
'''