import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import numpy as np 

class Autoencoder(tf.keras.models.Model):
    def __init__(self, image_size, channels, embedding_dim,  **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.image_size = image_size
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.encoder_output = None

        self.encoder()
        self.decoder()
    
    def encoder(self):
        self.encoder_input = Input(shape=(self.image_size, self.image_size, self.channels), name="encoder_input")
        x = Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(self.encoder_input)
        x = Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
        self.encoder_out_dim = x.shape[1:]
        x = Flatten()(x)
        self.encoder_output = Dense(self.embedding_dim, name="encoder_output")(x)
        self.encoder = tf.keras.models.Model(inputs=self.encoder_input, outputs=self.encoder_output)
    
    def decoder(self):
        decoder_input = Input(shape=(self.embedding_dim, ), name="decoder_input")
        x = Dense(np.prod(self.encoder_out_dim))(decoder_input)
        x = Reshape(self.encoder_out_dim)(x)
        x = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        self.decoder_output = Conv2D(self.channels, (3, 3), strides=1, activation="sigmoid", padding="same", name="decoder_output")(x)
        self.decoder = tf.keras.models.Model(inputs=decoder_input, outputs=self.decoder_output)
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
