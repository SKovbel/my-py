'''
https://keras.io/examples/generative/random_walks_with_stable_diffusion/

python3.9 -m venv ../../tmp/random_walks_with_stable_diffusion.venv
source ../../tmp/random_walks_with_stable_diffusion.venv/bin/activate
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install keras_cv
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install tensorflow
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install matplotlib
../../tmp/random_walks_with_stable_diffusion.venv/bin/pip install cuda-python

../../tmp/random_walks_with_stable_diffusion.venv/bin/python random_walks_with_stable_diffusion.py
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
import keras_cv
import keras
import matplotlib.pyplot as plt
#from tensorflow.keras import ops
import numpy as np
import math
from PIL import Image

# Enable mixed precision
# (only do this if you have a recent NVIDIA GPU)
keras.mixed_precision.set_global_policy("mixed_float16")

# Instantiate the Stable Diffusion model
model = keras_cv.models.StableDiffusion(jit_compile=True)



prompt_1 = "Mark came from school to home"
prompt_2 = "Maria has eatean"
interpolation_steps = 5
interpolation_steps = 1

encoding_1 = tf.squeeze(model.encode_text(prompt_1))
encoding_2 = tf.squeeze(model.encode_text(prompt_2))

interpolated_encodings = tf.linspace(encoding_1, encoding_2, interpolation_steps)

# Show the size of the latent manifold
print(f"Encoding shape: {encoding_1.shape}")


seed = 12345
noise = tf.random.normal((512 // 8, 512 // 8, 4), seed=seed)

images = model.generate_image(
    interpolated_encodings,
    batch_size=interpolation_steps,
    num_steps=10,
    diffusion_noise=noise,
)

def export_as_gif(filename, images, frames_per_second=10, rubber_band=False):
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )


export_as_gif(
    "doggo-and-fruit-5.gif",
    [Image.fromarray(img) for img in images],
    frames_per_second=2,
    rubber_band=True,
)

interpolation_steps = 150
batch_size = 3
batches = interpolation_steps // batch_size

interpolated_encodings = ops.linspace(encoding_1, encoding_2, interpolation_steps)
batched_encodings = ops.split(interpolated_encodings, batches)

images = []
for batch in range(batches):
    images += [
        Image.fromarray(img)
        for img in model.generate_image(
            batched_encodings[batch],
            batch_size=batch_size,
            num_steps=25,
            diffusion_noise=noise,
        )
    ]

export_as_gif("doggo-and-fruit-150.gif", images, rubber_band=True)

