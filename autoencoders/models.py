import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Model, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import numpy as np 

class Autoencoder(tf.keras.layers.Layer):
    def __init__(self, image_size, channels, embedding_dim,  **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.image_size = image_size
        self.channels = channels
        self.embedding_dim = embedding_dim

    def encoder(self):
        # Encoder
        self.encoder_input = Input(shape=(self.image_size,
                                     self.image_size,
                                     self.channels), name="encoder_input")
        x = Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(self.encoder_input)
        x = Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
        self.encoder_out_dim = K.int_shape(x)[1:]  # the decoder will need this!

        x = Flatten()(x)
        encoder_output = Dense(self.encoder_out_dim, name="encoder_output")(x)

        encoder = Model(self.encoder_input, encoder_output)

        return encoder
    
    def decoder(self):
        # Decoder
        decoder_input = Input(shape=(self.embedding_dim, ), name="decoder_input")
        x = Dense(np.prod(self.encoder_out_dim))(decoder_input)
        x = Reshape(self.encoder_out_dim)(x)
        x = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        self.decoder_output = Conv2D(self.channels, (3, 3), strides=1, activation="sigmoid", padding="same", name="decoder_output"
        )(x)

        decoder = Model(decoder_input, self.decoder_output)

        return decoder

    def call(self):
        # Autoencoder
        autoencoder = Model(self.encoder_input, self.decoder(self.encoder_output))  # decoder(encoder_output)
        
        return autoencoder
