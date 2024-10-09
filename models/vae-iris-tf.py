import keras
from keras import backend as K
from keras import layers
from keras import optimizers
from keras.datasets import mnist
from keras.models import Model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

original_dim = 4
latent_dim = 2

def encoder():
    encoder_input=layers.Input(shape=(original_dim,))
    x = layers.Dense(latent_dim)(encoder_input)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    return Model(encoder_input, [z_mean, z_log_var], name='encoder')

def decoder():
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(original_dim,activation='relu')(decoder_input)
    return Model(decoder_input, x, name='decoder')

def sampling(args):
    z_mean,z_log_var = args
    z_sigma = K.sqrt(K.exp(z_log_var))
    epsilon = K.random_normal(shape=(K.shape(z_mean)),mean=0,stddev=1)
    return z_mean + z_sigma*epsilon

def create_sampler():
    return layers.Lambda(sampling, name='sampler')

encoder = encoder()
decoder = decoder()
sampler = create_sampler()

x = layers.Input(shape=(original_dim,))
z_mean, z_log_var = encoder(x)
z = sampling([z_mean, z_log_var])
z_decoded = decoder(z)
vae = Model(x, z_decoded, name='vae')

rc_loss = keras.losses.mean_squared_error(x,z_decoded)
kl_loss = -0.5 * K.sum(1+z_log_var-K.square(z_mean)-K.exp(z_log_var), axis=-1)
vae_loss = K.mean(rc_loss+kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='sgd')

iris = pd.read_csv("~/Documents/bioresource eng proj/iris.csv")
x_train, x_test, y_train, y_test = train_test_split(iris[['sepal.length','sepal.width','petal.length','petal.width']],iris['variety'],test_size=0.1, random_state=1)
vae.fit(x_train, x_train,epochs=100,batch_size=135,validation_data=(x_test,x_test))

x_test_encoded = encoder.predict(x_test)
x_test_decoded = decoder.predict(x_test_encoded)

print('Original Datapoints :')
print(x_test)
print('Reconstructed Datapoints :')