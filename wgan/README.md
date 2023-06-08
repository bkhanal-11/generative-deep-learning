W# asserstein Generative Adversarial Network (WGAN)

Wasserstein Generative Adversarial Network (WGAN) is a variant of the Generative Adversarial Network (GAN) algorithm that introduces the Wasserstein distance as a metric for training the generator and discriminator models. WGAN aims to improve stability and address some of the limitations of traditional GANs, such as mode collapse and training convergence issues.
## Key Concepts

1. Generative Adversarial Network (GAN)

GANs consist of two main components: a generator network and a discriminator network. The generator learns to generate synthetic data samples that resemble real data, while the discriminator tries to distinguish between real and generated data. The two models are trained simultaneously in a competitive manner, where the generator aims to fool the discriminator, and the discriminator aims to correctly classify the real and generated samples.

2. Wasserstein Distance

The Wasserstein distance, also known as the Earth Mover's distance (EM distance), is a measure of the dissimilarity between two probability distributions. In the context of WGAN, it is used to quantify the difference between the real data distribution and the generated data distribution. Unlike the Jensen-Shannon divergence or the Kullback-Leibler divergence used in traditional GANs, the Wasserstein distance has desirable properties that lead to more stable training.

3. Wasserstein-1 Distance and Kantorovich-Rubinstein Duality

WGAN uses the Wasserstein-1 distance, which is a variation of the Wasserstein distance. The Wasserstein-1 distance measures the minimum cost required to transform one probability distribution into another. It is defined as the maximum difference between the expected values of a function (called a critic or discriminator function) applied to samples drawn from the two distributions. This formulation allows the discriminator to provide a more informative gradient signal during training.

The Kantorovich-Rubinstein duality theorem is leveraged to estimate the Wasserstein-1 distance efficiently. It states that the Wasserstein-1 distance can be computed as the maximum difference between the expected values of the critic function over all 1-Lipschitz functions. This enables the use of a critic network with weight clipping or gradient penalty techniques to enforce Lipschitz continuity and approximate the Wasserstein-1 distance.

4. Critic Network and Weight Clipping / Gradient Penalty

In WGAN, the discriminator is replaced by a critic network, which acts as an approximate evaluator of the Wasserstein distance. The critic is a neural network that takes input samples and outputs a scalar value representing the estimated Wasserstein distance between the real and generated distributions. During training, the critic is optimized to minimize the Wasserstein distance.

Two common techniques used to enforce the Lipschitz continuity constraint in WGAN are weight clipping and gradient penalty. Weight clipping involves constraining the weights of the critic network to a fixed range. Gradient penalty, on the other hand, penalizes the magnitude of the gradients of the critic network with respect to its inputs. These techniques help ensure that the critic satisfies the Lipschitz constraint, which is necessary to estimate the Wasserstein distance accurately.

5. Training Procedure

The training procedure in WGAN involves iteratively updating the critic and generator networks. During each iteration, the critic is trained to maximize the difference between the expected critic values for real and generated samples, while the generator is trained to minimize the expected critic value for generated samples. The updates to the critic and generator are performed using gradient descent techniques.

The key steps in the WGAN training procedure are as follows:

- **Step 1:** Sample a batch of real data samples.
- **Step 2:** Generate a batch of synthetic data samples using the generator.
- **Step 3:** Compute the Wasserstein-1 distance between the real and generated samples using the critic.
- **Step 4:** Update the critic parameters to minimize the negative Wasserstein distance.
- **Step 5:** Clip the weights of the critic network or apply gradient penalty to enforce Lipschitz continuity.
- **Step 6:** Update the generator parameters to minimize the expected critic value for generated samples.
- **Step 7:** Repeat steps 1-6 for multiple iterations.

### Key Discussion Points

1. Why vanilla GAN suffers from Mode collapse and Vanishing Gradient and we need a different loss function

2. Comparison of KL, JS, TV divergences, Earth Mover's Distance

3. Earth Mover's Distance

4. W-Loss Derived from Earth Mover's Distance

5. How enforcing 1-Lipschitz Continuity on Critic forces W-Loss to be valid as Earth Mover's Distance (linear condition)

6. 1-Lipschitz Continuity, how 1 came to be for W-Loss and how it forces critic function to be linear

7. How to limit gradients of critic using weight clipping and gradient penalty

8. Conditional and Controllable Generation
