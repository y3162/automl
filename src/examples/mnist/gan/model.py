"""Two small adversarial models with NumPy CPU gradients."""
import numpy as np

from examples.mnist.model import ArrayModel


def sigmoid(logits):
    return np.exp(-np.logaddexp(0, -logits))


class Generator(ArrayModel):
    """Map Gaussian noise to 784 grayscale intensities with a sigmoid layer."""

    def __init__(self, latent_dim: int = 16, seed: int = 1):
        self.latent_dim = latent_dim
        rng = np.random.default_rng(seed)
        self.parameters = {
            "weight": rng.normal(0, latent_dim ** -0.5, (latent_dim, 784)).astype(np.float32),
            "bias": np.zeros(784, dtype=np.float32),
        }

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        if type(values["latent_dim"]) is not int or values["latent_dim"] < 1:
            raise ValueError("latent_dim must be a positive integer")
        if type(values["seed"]) is not int or values["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        return values

    def generate(self, noise):
        return sigmoid(noise @ self.parameters["weight"] + self.parameters["bias"])

    def update(self, noise, images, image_gradient, rate):
        delta = image_gradient * images * (1 - images)
        self.parameters["weight"] -= rate * (noise.T @ delta)
        self.parameters["bias"] -= rate * delta.sum(axis=0)


class Discriminator(ArrayModel):
    """Binary logistic regression distinguishing MNIST images from generator output."""

    def __init__(self, seed: int = 2):
        rng = np.random.default_rng(seed)
        self.parameters = {
            "weight": rng.normal(0, 1 / 28, (784, 1)).astype(np.float32),
            "bias": np.zeros(1, dtype=np.float32),
        }

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        if type(values["seed"]) is not int or values["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        return values

    def logits(self, images):
        return images @ self.parameters["weight"] + self.parameters["bias"]

    def update(self, real, fake, rate):
        real_logits, fake_logits = self.logits(real), self.logits(fake)
        real_delta = (sigmoid(real_logits) - 1) / (2 * len(real))
        fake_delta = sigmoid(fake_logits) / (2 * len(fake))
        self.parameters["weight"] -= rate * (real.T @ real_delta + fake.T @ fake_delta)
        self.parameters["bias"] -= rate * (real_delta.sum() + fake_delta.sum())
        return float((np.logaddexp(0, -real_logits).mean() + np.logaddexp(0, fake_logits).mean()) / 2)

    def generator_gradient(self, fake):
        logits = self.logits(fake)
        delta = (sigmoid(logits) - 1) / len(fake)
        return delta @ self.parameters["weight"].T, float(np.logaddexp(0, -logits).mean())
