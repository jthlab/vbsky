from dataclasses import dataclass
from typing import TypeVar, Generic

import jax
from jax import numpy as jnp

T = TypeVar("T")


@dataclass
# want to use abc.ABC here but can't get it to cooperate with dataclass
class Distribution(Generic[T]):
    "A distribution is something you can sample from and evaluate likelihoods."
    dim: int  # the dimension of the returned samples

    @property
    def params(self) -> T:
        "The parameters for this distribution."
        pass

    def sample(self, rng: jax.random.PRNGKey, params: T, n: int = 1) -> jnp.ndarray:
        "Draw n samples from this distribution."
        pass

    def log_pdf(self, params: T, x: jnp.ndarray) -> float:
        pass

    def _check_x_1d(self, x):
        x = jnp.atleast_1d(x)
        if x.ndim != 1 or len(x) != self.dim:
            raise ValueError(
                f"When calling log_pdf(x), x should be a 1-d vector of length {self.dim=}. (For vectorized evaluation use jax.vmap)."
            )


@dataclass
class Beta(Distribution):
    @property
    def params(self):
        return {"a": 1, "b": 1}

    def sample(self, rng: jax.random.PRNGKey, params, n) -> jnp.ndarray:
        return jax.random.beta(rng, params["a"], params["b"], (n, self.dim))

    def log_pdf(self, params, x) -> float:
        self._check_x_1d(x)
        return jax.scipy.stats.beta.logpdf(
            jnp.atleast_1d(x), params["a"], params["b"]
        ).sum()


@dataclass
class MeanField(Distribution):
    """Mean field Gaussian (diagonal covariance matrix)."""

    @property
    def params(self):
        return {}

    def sample(self, rng: jax.random.PRNGKey, params, n) -> jnp.ndarray:
        return jax.random.normal(rng, (n, self.dim))

    def log_pdf(self, params, x) -> float:
        self._check_x_1d(x)
        return jax.scipy.stats.norm.logpdf(jnp.atleast_1d(x)).sum()


# @dataclass
# class FullRank(Distribution):
#     def _tril(self, L):
#         return jax.ops.index_update(
#             jnp.zeros([self.dim, self.dim]), jnp.tril_indices(self.dim), L
#         )
#
#     @property
#     def params(self):
#         I = jnp.eye(self.dim)
#         return {"L": I[jnp.tril_indices_from(I)], "mu": jnp.zeros(self.dim)}
#
#     def sample(self, rng: jax.random.PRNGKey, params, n) -> jnp.ndarray:
#         L = self._tril(params["L"])
#         return jax.random.multivariate_normal(rng, params["mu"], L @ L.T, shape=(n,))
#
#     def log_pdf(self, params, x):
#         self._check_x_1d(x)
#         L = self._tril(params["L"])
#         return jax.scipy.stats.multivariate_normal.logpdf(x, params["mu"], L @ L.T)
#


class PointMass(Distribution):
    @property
    def params(self) -> dict:
        return {"x": jnp.zeros(self.dim)}

    def sample(self, rng, params, n=1):
        return jnp.full((n, self.dim), params["x"])

    def log_pdf(self, params, x):
        self._check_x_1d(x)
        # after transforming, x may differ numerically from params['x']
        if len(x) == self.dim:
            return jnp.where(jnp.isclose(x, params["x"]).all(), 0.0, -jnp.inf)
        return -jnp.inf


@dataclass
class Constant:
    c: jnp.ndarray

    @property
    def params(self) -> dict:
        return {}

    @property
    def dim(self):
        return len(jnp.atleast_1d(self.c))

    def sample(self, rng, params, n=1):
        return jnp.full(shape=(n, self.dim), fill_value=self.c)

    def log_pdf(self, params, x):
        return jnp.where(jnp.isclose(x, self.c), 0.0, -jnp.inf).sum()
