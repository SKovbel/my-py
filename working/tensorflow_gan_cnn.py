import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2

# Generator model
def generator(input_shape):
    input_layer = Input(shape=input_shape)
    # Encoder (640x480 -> 320x240)
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    
    # Decoder (320x240 -> 640x480)
    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='sigmoid')(x)
    
    generator = Model(inputs=input_layer, outputs=x)
    return generator

# Discriminator model
def discriminator(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', activation=LeakyReLU(0.2))(input_layer)
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation=LeakyReLU(0.2))(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(inputs=input_layer, outputs=x)
    return discriminator

# GAN model
def gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=generator.input.shape[1:])
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = Model(inputs=gan_input, outputs=gan_output)
    return gan

# Image preprocessing function
def preprocess_image(image_path, target_shape):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (target_shape[1], target_shape[0]))
    image = image / 255.0  # Normalize pixel values
    return image

# Training parameters
image_shape = (640, 480, 3)
input_image_path = './images/vg_night_640_x_480.png'
output_image_path = './images/win_xp_640_x_480.png'
epochs = 200
batch_size = 1

# Load and preprocess input and output images
input_image = preprocess_image(input_image_path, image_shape)
output_image = preprocess_image(output_image_path, image_shape)

# Create and compile the GAN
generator = generator(image_shape)
discriminator = discriminator(image_shape)
gan = gan(generator, discriminator)
discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')
gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

real_labels = np.expand_dims(np.ones((1, 1)), axis=0)
fake_labels = np.expand_dims(np.zeros((batch_size, 1)), axis=0)
output_image = np.expand_dims(output_image, axis=0)  # Add batch dimension

# Training loop
for epoch in range(epochs):
    noise = np.random.randn(batch_size, *image_shape)
    #noise = tf.random.normal([batch_size, *image_shape])
    generated_images = generator.predict(noise)
    real_loss = discriminator.train_on_batch(output_image, real_labels)
    fake_loss = discriminator.train_on_batch(generated_images, fake_labels)
    discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
    
    # Train the GAN
    gan_labels = np.ones((batch_size, 1))
    gan_loss = gan.train_on_batch(noise, gan_labels)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, D Loss: {discriminator_loss}, G Loss: {gan_loss}")

# After training, you can use the generator to modify an image
input_image = input_image[np.newaxis, ...]  # Add a batch dimension
modified_image = generator.predict(input_image)[0]

# Display the modified image
cv2.imshow('Modified Image', modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
