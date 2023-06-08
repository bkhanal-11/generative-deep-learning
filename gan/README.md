# Generative Adversarial Networks (GAN)

GAN is a type of neural network that consists of two main components: a generator and a discriminator. GANs were first introduced by Goodfellow et al. in their 2014 paper ["Generative Adversarial Networks"](https://arxiv.org/pdf/1406.2661.pdf).

The generator takes a random noise vector as input and generates a sample that is intended to look like it was drawn from the same distribution as the training data. The discriminator takes as input a sample (either real or generated) and outputs a probability indicating whether the sample is real or fake.

During training, the generator tries to generate samples that will fool the discriminator into thinking they are real, while the discriminator tries to distinguish between real and generated samples. The two components are trained together in an adversarial setting, with the generator trying to maximize the probability of fooling the discriminator, and the discriminator trying to correctly classify samples as real or fake.

The loss function used in GANs is called the adversarial loss, which is based on the idea of a two-player minimax game. The generator's loss is the negative log-probability that the discriminator assigns to its generated samples, while the discriminator's loss is the negative log-probability that it assigns to real samples plus the negative log-probability that it assigns to the generator's samples.

One important aspect of training GANs is to ensure that the generator and discriminator are not too powerful relative to each other. If the generator is too powerful, it can generate samples that perfectly mimic the training data, causing the discriminator to be unable to distinguish between real and generated samples. On the other hand, if the discriminator is too powerful, it can easily distinguish between real and generated samples, causing the generator to generate low-quality samples. This is known as the "mode collapse" problem in GANs.

To address this issue, several techniques have been proposed, such as adding noise to the input of the discriminator or using a technique called "batch normalization" to stabilize the training process. Additionally, more recent GAN architectures, such as Wasserstein GANs and StyleGANs, have been developed that address some of the limitations of the original GAN architecture.

### Key Discussion Points

1. Batch Normalization


2. Recalling Classification Problem and Decoder of VAE


3. Architecture of vanilla GAN


4. Binary Cross Entropy function and its preference for classification problem


5. Generator and Discriminator Losses in GAN


6. Preference of logarithmic function in Neural Network


7. Limitation of GAN [Mode Collapse, Convergence, Vanishing Gradient]


8. K-Means clustering and Vector Quantization (VQ)

