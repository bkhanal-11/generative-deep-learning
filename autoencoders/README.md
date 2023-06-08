# Autoencoder

An autoencoder is a type of neural network that is trained to compress data into a low-dimensional representation and then reconstruct the original data from that representation. This can be useful for tasks such as data compression, anomaly detection, and image denoising.

The basic structure of an autoencoder consists of an encoder network that compresses the data into a lower-dimensional representation and a decoder network that reconstructs the original data from that representation. The goal is to train the network to minimize the difference between the input and output data.

Here is an example of an autoencoder architecture:

![Autoencoder Architecture](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fstarship-knowledge.com%2Fwp-content%2Fuploads%2F2020%2F10%2Fautoencoder-676x478.jpeg&f=1&nofb=1&ipt=58fa19346665631b026038a4891b8e5d3e4f2bb6ddea34c6574baab27a8c8958&ipo=images)

In this architecture, the input data is passed through the encoder network, which compresses it into a lower-dimensional representation in the latent space. The decoder network then takes this representation and reconstructs the original data.

During training, the network is fed pairs of input and target data, and the weights of the encoder and decoder networks are adjusted to minimize the difference between the output of the decoder and the target data.

### Key Discussion Points

1. Difference between generative and discriminative modelling


2. Motivation behind Autoencoder


3. Basic structure of Autoencoder


4. Dimensionality reduction, Feature Extraction with Encoder and Input generation from Decoder


5. Autoencoder vs PCA vs SVD


6. Analogy of Autoencoder with Art Exhibition


7. Visualization of MNIST dataset in 2-dimensional latent space


8. Limitation of Autoencoder and introduction to Variational Autoencoder


9. Comparison of U-Net and VAE