import numpy as np
import jax.numpy as jnp
import geomstats.geometry.hypersphere as hypersphere

from score_sde.datasets import TensorDataset
from score_sde.utils import register_dataset


class CheckerboardDataset(TensorDataset):
    """Checkerboard pattern on S² in (longitude, latitude) coordinates."""

    def __init__(self, n_samples=3000, n_bins=8, seed=42, **kwargs):
        """
        Args:
            n_bins: number of bins along each of the two angular axes
                    (longitude and latitude); higher = more, smaller squares.
        """
        self.manifold = hypersphere.Hypersphere(2)
        rng = np.random.RandomState(seed)

        samples = []
        while len(samples) < n_samples:
            phi = rng.uniform(0, 2 * np.pi)  # longitude in [0, 2pi)
            cos_theta = rng.uniform(-1, 1)  # uniform-on-sphere sampling
            theta = np.arccos(cos_theta)  # polar angle in [0, pi]

            bin_lon = int(phi / (2 * np.pi) * n_bins)
            bin_lat = int(theta / np.pi * n_bins)
            if (bin_lon + bin_lat) % 2 == 0:  # checkerboard mask
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                samples.append([x, y, z])

        data = jnp.array(np.stack(samples))
        super().__init__(data)

        self.n_bins = n_bins


@register_dataset
class CheckerboardSynthetic(CheckerboardDataset):
    def __init__(self, data_dir="data", **kwargs):
        super().__init__(**kwargs)
