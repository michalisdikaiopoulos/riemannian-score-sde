import numpy as np
import jax
import jax.numpy as jnp

from geomstats.geometry.euclidean import Euclidean
from score_sde.sde import SDE
from distrax import MultivariateNormalDiag


class UniformDistribution:
    """Uniform density on compact manifold"""

    def __init__(self, manifold, **kwargs):
        self.manifold = manifold

    def sample(self, rng, shape):
        return self.manifold.random_uniform(state=rng, n_samples=shape[0])

    def log_prob(self, z):
        return -jnp.ones([z.shape[0]]) * self.manifold.log_volume

    def grad_U(self, x):
        return jnp.zeros_like(x)


class MultivariateNormal(MultivariateNormalDiag):
    def __init__(self, dim, mean=None, scale=None, **kwargs):
        mean = jnp.zeros((dim)) if mean is None else mean
        scale = jnp.ones((dim)) if scale is None else scale
        super().__init__(mean, scale)

    def sample(self, rng, shape):
        return super().sample(seed=rng, sample_shape=shape)

    def log_prob(self, z):
        return super().log_prob(z)

    def grad_U(self, x):
        return x / (self.scale_diag**2)


class DefaultDistribution:
    def __new__(cls, manifold, flow, **kwargs):
        if isinstance(flow, SDE):
            return flow.limiting
        else:
            if isinstance(manifold, Euclidean):
                zeros = jnp.zeros((manifold.dim))
                ones = jnp.ones((manifold.dim))
                return MultivariateNormalDiag(zeros, ones)
            elif hasattr(manifold, "random_uniform"):
                return UniformDistribution(manifold)
            else:
                # TODO: WrappedNormal
                raise NotImplementedError(f"No default distribution for {manifold}")


class WrapNormDistribution:
    def __init__(self, manifold, scale=1.0, mean=None):
        self.manifold = manifold
        if mean is None:
            mean = self.manifold.identity
        self.mean = mean
        # NOTE: assuming diagonal scale
        self.scale = (
            jnp.ones((mean.shape)) * scale
            if isinstance(scale, float)
            else jnp.array(scale)
        )

    def sample(self, rng, shape):
        mean = self.mean[None, ...]
        tangent_vec = self.manifold.random_normal_tangent(
            rng, self.manifold.identity, np.prod(shape)
        )[1]
        # tangent_vec = self.manifold.random_normal_tangent(rng, mean, np.prod(shape))[1]
        tangent_vec *= self.scale
        tangent_vec = self.manifold.metric.transpfrom0(mean, tangent_vec)
        return self.manifold.metric.exp(tangent_vec, mean)

    def log_prob(self, z):
        tangent_vec = self.manifold.metric.log(z, self.mean)
        tangent_vec = self.manifold.metric.transpback0(self.mean, tangent_vec)
        zero = jnp.zeros((self.manifold.dim))
        # TODO: to refactor axis contenation / removal
        if self.scale.shape[-1] == self.manifold.dim:  # poincare
            scale = self.scale
        else:  # hyperboloid
            scale = self.scale[..., 1:]
        norm_pdf = MultivariateNormalDiag(zero, scale).log_prob(tangent_vec)
        logdetexp = self.manifold.metric.logdetexp(self.mean, z)
        return norm_pdf - logdetexp

    def grad_U(self, x):
        def U(x):
            sq_dist = self.manifold.metric.dist(x, self.mean) ** 2
            res = 0.5 * sq_dist / (self.scale[0] ** 2)  # scale must be isotropic
            logdetexp = self.manifold.metric.logdetexp(self.mean, x)
            return res + logdetexp

        # U = lambda x: -self.log_prob(x)  #NOTE: this does not work

        return self.manifold.to_tangent(self.manifold.metric.grad(U)(x), x)


class WrapNormDistributionSphere:
    """Wrapped normal distribution specifically for Hypersphere S^d."""

    def __init__(self, manifold, scale=1.0, mean=None):
        self.manifold = manifold
        self.dim = manifold.dim  # 2 for S²

        # Default mean: north pole
        if mean is None:
            mean = jnp.zeros(self.dim + 1).at[-1].set(1.0)  # [0, 0, 1]
        self.mean = mean
        self.north_pole = jnp.zeros(self.dim + 1).at[-1].set(1.0)

        self.scale = scale if isinstance(scale, float) else float(scale)

    def _to_tangent_at_north_pole(self, rng, n_samples):
        """Sample random tangent vectors at north pole.
        Tangent space at [0,0,1] is the xy-plane: [v1, v2, 0]."""
        v = jax.random.normal(rng, (n_samples, self.dim)) * self.scale
        # Pad with zero z-component
        return jnp.concatenate([v, jnp.zeros((n_samples, 1))], axis=-1)

    def _parallel_transport(self, tangent_vec, base_point, end_point):
        """Closed-form parallel transport on S^d from base_point to end_point.

        Formula: v_transported = v - <v, log_ab / ||log_ab||> *
                 (tan(||log_ab|| / 2)) * (log_ab / ||log_ab|| + log_ba / ||log_ba||)

        Simplified using: PT_a->b(v) = v - (dot(a+b, v) / (1 + dot(a, b))) * (a + b)
        """
        # Handle batch dimensions
        dot_ab = jnp.sum(base_point * end_point, axis=-1, keepdims=True)
        # Avoid division by zero when base == end
        denom = jnp.maximum(1.0 + dot_ab, 1e-8)
        ab = base_point + end_point
        coeff = jnp.sum(ab * tangent_vec, axis=-1, keepdims=True) / denom
        return tangent_vec - coeff * ab

    def _exp_map(self, tangent_vec, base_point):
        """Exponential map on S^d: base_point + tangent_vec -> point on sphere."""
        norm = jnp.linalg.norm(tangent_vec, axis=-1, keepdims=True)
        norm = jnp.maximum(norm, 1e-8)
        return base_point * jnp.cos(norm) + tangent_vec * (jnp.sin(norm) / norm)

    def _log_map(self, point, base_point):
        """Logarithmic map on S^d: inverse of exp map."""
        dot = jnp.sum(point * base_point, axis=-1, keepdims=True)
        dot = jnp.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = jnp.arccos(dot)
        direction = point - dot * base_point
        norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        norm = jnp.maximum(norm, 1e-8)
        return (theta / norm) * direction

    def sample(self, rng, shape):
        """Sample from wrapped normal: noise at north pole -> transport to mean -> exp map."""
        n_samples = np.prod(shape) if isinstance(shape, tuple) else shape
        tangent_vec = self._to_tangent_at_north_pole(rng, n_samples)
        tangent_vec = self._parallel_transport(
            tangent_vec, self.north_pole, self.mean
        )
        return self._exp_map(tangent_vec, self.mean)

    def log_prob(self, z):
        """Log probability of the wrapped normal."""
        tangent_vec = self._log_map(z, self.mean)
        # Transport back to north pole for density computation
        tangent_vec = self._parallel_transport(
            tangent_vec, self.mean, self.north_pole
        )
        # Only take the first d components (tangent space at north pole is xy-plane)
        v = tangent_vec[..., :self.dim]
        # Gaussian log prob
        log_prob = -0.5 * jnp.sum(v ** 2, axis=-1) / (self.scale ** 2)
        log_prob -= self.dim * 0.5 * jnp.log(2 * jnp.pi * self.scale ** 2)
        # Jacobian correction: log det of exp map
        log_prob -= self._logdetexp(z)
        return log_prob

    def _logdetexp(self, z):
        """Log determinant of the exponential map Jacobian at mean."""
        dot = jnp.sum(z * self.mean, axis=-1)
        dot = jnp.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = jnp.arccos(dot)
        # For S^d: logdet = (d-1) * log(sin(theta) / theta)
        sinc = jnp.where(theta < 1e-7, 1.0, jnp.sin(theta) / theta)
        return (self.dim - 1) * jnp.log(jnp.maximum(sinc, 1e-10))

    def grad_U(self, x):
        """Gradient of the potential U = -log_prob (without normalization constants)."""
        sq_dist = jnp.sum(self._log_map(x, self.mean) ** 2, axis=-1)
        # U(x) = 0.5 * dist² / scale² + logdetexp
        # grad U = (1/scale²) * (-log_map) + grad_logdetexp
        # Simplified: use the log map directly
        log_vec = self._log_map(x, self.mean)
        grad = -log_vec / (self.scale ** 2)
        # Project to tangent space (ensure it's tangent to sphere at x)
        grad = grad - jnp.sum(grad * x, axis=-1, keepdims=True) * x
        return grad