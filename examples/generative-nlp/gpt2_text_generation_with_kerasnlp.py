'''
https://keras.io/examples/generative/gpt2_text_generation_with_kerasnlp/

source ../../tmp/random_walks_with_stable_diffusion.venv/bin/activate
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install keras_cv
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install tensorflow
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install matplotlib
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install git+https://github.com/keras-team/keras-nlp.git -q

../../tmp/random_walks_with_stable_diffusion.venv/bin/python gpt2_text_generation_with_kerasnlp.py

'''

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax" "tensorflow" or "torch"

import keras_nlp
import keras
import tensorflow as tf
import time

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


start = time.time()
output = gpt2_lm.generate("My trip to Yosemite was", max_length=200)
print("\nGPT-2 output:")
print(output)
end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

start = time.time()
output = gpt2_lm.generate("That Italian restaurant is", max_length=200)
print("\nGPT-2 output:")
print(output)
end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

start = time.time()
output = gpt2_lm.generate("What happance if the Europe fall down to the Mars", max_length=200)
print("\nGPT-2 output:")
print(output)
end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

