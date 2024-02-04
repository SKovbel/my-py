'''
https://keras.io/examples/generative/gpt2_text_generation_with_kerasnlp/

source ../../tmp/random_walks_with_stable_diffusion.venv/bin/activate
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install keras_cv
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install tensorflow
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install matplotlib
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install git+https://github.com/keras-team/keras-nlp.git -q

../../tmp/random_walks_with_stable_diffusion.venv/bin/python gpt2_text_generation_with_kerasnlp.fine-tuning.py

'''

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax" "tensorflow" or "torch"

import keras_nlp
import keras
import tensorflow as tf
import time
import tensorflow_datasets as tfds

keras.mixed_precision.set_global_policy("mixed_float16")

# To speed up training and generation, we use preprocessor of length 128
# instead of full length 1024.
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)


reddit_ds = tfds.load(
    "reddit_tifu",
    split="train",
    as_supervised=True,
    ignore_verifications=True
)

for document, title in reddit_ds:
    print(document.numpy())
    print(title.numpy())
    break

train_ds = (
    reddit_ds.map(lambda document, _: document)
    .batch(32)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

train_ds = train_ds.take(500)
num_epochs = 1

# Linearly decaying learning rate.
learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-5,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

gpt2_lm.fit(train_ds, epochs=num_epochs)


start = time.time()
output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)
end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

