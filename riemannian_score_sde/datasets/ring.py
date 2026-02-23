import numpy as np
import jax.numpy as jnp
import geomstats.geometry.hypersphere as hypersphere

from score_sde.datasets import TensorDataset
from score_sde.utils import register_dataset


class RingDataset(TensorDataset):
    """Points distributed in a band around a great circle on S²."""

    def __init__(self, n_samples=2000, latitude=0.0, concentration=20.0, seed=42, **kwargs):
        """
        Args:
            latitude: center of the band in radians.
                      0 = equator (great circle), pi/4 = 45°N, etc.
            concentration: how tight the band is.
                          Higher = thinner ring, lower = wider band.
        """
        self.manifold = hypersphere.Hypersphere(2)
        rng = np.random.RandomState(seed)

        # Sample azimuthal angle uniformly around the ring
        phi = rng.uniform(0, 2 * np.pi, size=n_samples)

        # Sample polar deviation from the latitude using von Mises
        # This concentrates points near the target latitude
        theta_center = np.pi / 2 - latitude  # convert latitude to polar angle
        theta = rng.vonmises(theta_center, concentration, size=n_samples)
        theta = np.clip(theta, 0.01, np.pi - 0.01)  # stay away from poles

        # Spherical to Cartesian
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        data = jnp.array(np.stack([x, y, z], axis=-1))
        super().__init__(data)

        self.latitude = latitude
        self.concentration = concentration


@register_dataset
class RingSynthetic(RingDataset):
    def __init__(self, data_dir="data", **kwargs):
        super().__init__(**kwargs)