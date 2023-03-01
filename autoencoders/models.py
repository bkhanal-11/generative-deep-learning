import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Model, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import numpy as np 

IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 100
BUFFER_SIZE = 1000
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 2
EPOCHS = 3

# Encoder
encoder_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input")
x = Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]  # the decoder will need this!

x = Flatten()(x)
encoder_output = Dense(EMBEDDING_DIM, name="encoder_output")(x)

encoder = Model(encoder_input, encoder_output)
encoder.summary()

# Decoder
decoder_input = Input(shape=(EMBEDDING_DIM,), name="decoder_input")
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)
x = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
decoder_output = Conv2D(
    CHANNELS, (3, 3), strides=1, activation="sigmoid", padding="same", name="decoder_output"
)(x)

decoder = Model(decoder_input, decoder_output)
decoder.summary()

# Autoencoder
autoencoder = Model(encoder_input, decoder(encoder_output))  # decoder(encoder_output)
autoencoder.summary()
