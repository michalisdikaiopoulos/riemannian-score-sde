import numpy as np
import jax.numpy as jnp
import geomstats.geometry.hypersphere as hypersphere

from score_sde.datasets import TensorDataset
from score_sde.utils import register_dataset


class KentSyntheticDataset(TensorDataset):
    """Synthetic mixture of von Mises-Fisher distributions on S²."""

    def __init__(self, n_samples=2000, n_components=3, kappa=30.0, seed=42, **kwargs):
        self.manifold = hypersphere.Hypersphere(2)
        rng = np.random.RandomState(seed)

        # Place means at well-separated points on S²
        default_means = np.array([
            [0.0, 0.0, 1.0],  # north pole
            [1.0, 0.0, 0.0],  # equator x
            [0.0, 1.0, 0.0],  # equator y
            [0.0, 0.0, -1.0],  # south pole
            [-1.0, 0.0, 0.0],  # equator -x
            [0.0, -1.0, 0.0],  # equator -y
        ])
        means = default_means[:n_components]
        weights = np.ones(n_components) / n_components

        # Sample from mixture
        samples = []
        labels = rng.choice(n_components, size=n_samples, p=weights)
        for i in range(n_samples):
            samples.append(self._sample_vmf(means[labels[i]], kappa, rng))

        self.data = jnp.array(np.stack(samples))

        # Store ground truth
        self.means = means
        self.kappa = kappa
        self.weights = weights
        self.labels = labels

    def _sample_vmf(self, mu, kappa, rng):
        """Wood's algorithm for von Mises-Fisher on S²."""
        d = 3
        b = (-2 * kappa + np.sqrt(4 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
        x0 = (1 - b) / (1 + b)
        c = kappa * x0 + (d - 1) * np.log(1 - x0 ** 2)

        while True:
            z = rng.beta((d - 1) / 2, (d - 1) / 2)
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = rng.uniform()
            if kappa * w + (d - 1) * np.log(1 - x0 * w) - c >= np.log(u):
                break

        v = rng.randn(d)
        v = v - mu * np.dot(v, mu)
        v = v / (np.linalg.norm(v) + 1e-10)

        sample = w * mu + np.sqrt(1 - w ** 2) * v
        return sample / np.linalg.norm(sample)


@register_dataset
class KentSynthetic(KentSyntheticDataset):
    def __init__(self, data_dir="data", **kwargs):
        super().__init__(**kwargs)