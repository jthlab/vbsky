import abc
from dataclasses import dataclass
from typing import Type, Dict, Tuple, Union

import jax
import numpy as np
from jax import numpy as jnp, vmap
from jax._src.scipy.special import expit, logit

from vbsky.prob.distribution import Distribution, MeanField


@dataclass
class Transformation(abc.ABC):

    dim: int

    @property
    @abc.abstractmethod
    def params(self) -> dict:
        pass

    @abc.abstractmethod
    def direct(self, params, x):
        pass

    @abc.abstractmethod
    def log_det_jac(self, params, x):
        pass

    @abc.abstractmethod
    def inverse(self, params, x):
        pass


def Transform(
    f_or_dim: Union[int, Distribution], T: Type[Transformation]
) -> Distribution:
    if isinstance(f_or_dim, int):
        f = MeanField(f_or_dim)
    else:
        assert isinstance(f_or_dim, Distribution)
        f = f_or_dim

    tr = T(f.dim)

    @dataclass
    class TransformedDistribution(Distribution):
        @property
        def params(self):
            return {"base": f.params, "transform": tr.params}

        def sample(self, rng: jax.random.PRNGKey, params, n: int = 1):
            x = f.sample(rng, params["base"], n)
            return vmap(tr.direct, (None, 0))(params["transform"], x)

        def log_pdf(self, params, x):
            # y=f(x)
            # f_Y(y) = f_X(f^-1(y)) |Df^-1(y)| = f_X(x)/|Df(x)|
            x0 = tr.inverse(params["transform"], x)
            return f.log_pdf(params["base"], x0) - tr.log_det_jac(
                params["transform"], x0
            )

    return TransformedDistribution(f.dim)


class Shift(Transformation):
    @property
    def params(self):
        return {"c": jnp.zeros(self.dim)}

    def direct(self, params, x):
        return x + params["c"]

    def inverse(self, params, y):
        return y - params["c"]

    def log_det_jac(self, params, x):
        return 0.0


@dataclass
class Scale(Transformation):
    @property
    def params(self) -> dict:
        return {"log_sigma": jnp.zeros(self.dim)}

    def direct(self, params, x):
        return jnp.exp(params["log_sigma"]) * x

    def inverse(self, params, y):
        return jnp.exp(-params["log_sigma"]) * y

    def log_det_jac(self, params, x):
        return params["log_sigma"].sum()


@dataclass
class Affine(Transformation):
    """X -> L @ X + mu where L is lower triangular"""

    def _tril(self, vec_L) -> jnp.ndarray:
        return jax.ops.index_update(
            jnp.zeros([self.dim] * 2), jnp.tril_indices(self.dim), vec_L
        )

    @property
    def params(self) -> dict:
        I = jnp.eye(self.dim)
        return {
            "L": I[jnp.tril_indices_from(I)],
            "mu": jnp.zeros(self.dim),
        }

    def direct(self, params, x) -> jnp.ndarray:
        L = self._tril(params["L"])
        return jnp.dot(x, L.T) + params["mu"]

    def log_det_jac(self, params, x) -> jnp.ndarray:
        L = self._tril(params["L"])
        return jnp.log(abs(L.diagonal())).sum()

    def inverse(self, params, y):
        L = self._tril(params["L"])
        return jax.scipy.linalg.solve_triangular(L, y - params["mu"], lower=True)


@dataclass
class DiagonalAffine(Transformation):

    """X -> exp(log_sigma)*X + mu"""

    @property
    def params(self) -> dict:
        return {"log_sigma": jnp.zeros(self.dim), "mu": jnp.zeros(self.dim)}

    def direct(self, params, x):
        return jnp.exp(params["log_sigma"]) * x + params["mu"]

    def log_det_jac(self, params, x):
        return params["log_sigma"].sum()

    def inverse(self, params, y):
        return (y - params["mu"]) * jnp.exp(-params["log_sigma"])


@dataclass
class Exp(Transformation):
    "x -> exp(x)"

    @property
    def params(self) -> dict:
        return {}

    def direct(self, params, x):
        return jnp.exp(x)

    def log_det_jac(self, params, x):
        return jnp.sum(x)

    def inverse(self, params, y):
        return jnp.log(y)


@dataclass
class Softplus(Transformation):
    "x -> log(1 + exp(x))"

    @property
    def params(self) -> dict:
        return {}

    def direct(self, params, x):
        return jax.nn.softplus(x)

    def log_det_jac(self, params, x):
        return (x - self.direct(params, x)).sum()

    def inverse(self, params, y):
        return jnp.where(y > 10, y, jnp.log(jnp.expm1(y)))


Positive = Softplus


@dataclass
class ZeroOne(Transformation):
    @property
    def params(self) -> dict:
        return {}

    def direct(self, params, x):
        return jnp.clip(expit(x), 1e-7, 1 - 1e-7)

    def log_det_jac(self, params, x):
        return (jnp.log(expit(x)) + jnp.log1p(-expit(x))).sum()

    def inverse(self, params, y):
        yc = jnp.clip(y, 1e-7, 1 - 1e-7)
        return logit(yc)


def Blockwise(
    **kwargs: Union[int, Tuple[int, Transformation]],
) -> Distribution:
    def f(x):
        if isinstance(x, int):
            return (x, Identity)
        assert isinstance(x, tuple)
        return x

    blocks = {k: f(v) for k, v in kwargs.items()}
    block_sizes, transformations = zip(*blocks.values())
    splits = np.cumsum(block_sizes)[:-1]
    dim = sum(block_sizes)

    def split(A):
        return dict(zip(blocks, jnp.split(A, splits, axis=-1)))

    @dataclass
    class Block(Transformation):
        @property
        def params(self):
            return {k: T(b).params for k, (b, T) in blocks.items()}

        def direct(self, params, u):
            assert u.shape[-1] == dim
            arys = split(u)
            return {
                k: T(b).direct(params[k], arys[k])
                for k, T, b in zip(blocks, transformations, block_sizes)
            }

        def log_det_jac(self, params, u):
            arys = split(u)
            return sum(
                [
                    T(b).log_det_jac(params[k], arys[k])
                    for k, T, b in zip(blocks, transformations, block_sizes)
                ]
            )

        def inverse(self, params, d):
            return jnp.concatenate(
                [
                    T(b).inverse(params[k], d[k])
                    for k, T, b in zip(blocks, transformations, block_sizes)
                ],
                axis=-1,
            )

    return Block


def Concat(*args: Distribution):
    """Concatenate two distributions: given distributions d1, d2, returns a distribution of dimension d1.dim + d2.dim
    obtained by concatenating their outputs.
    """
    nd = len(args)
    splits_dim = np.cumsum([d.dim for d in args])
    splits = splits_dim[:-1]
    dim = splits_dim[-1]

    class ConcatenatedDistribution(Distribution):
        @property
        def params(self):
            return tuple(d.params for d in args)

        def sample(self, rng: jax.random.PRNGKey, params, n: int = 1) -> jnp.ndarray:
            rs = jax.random.split(rng, nd)
            ss = [f.sample(r, p, n) for r, f, p in zip(rs, args, params)]
            return jnp.concatenate(ss, axis=1)

        def log_pdf(self, params, x):
            arys = jnp.split(x, splits, axis=-1)
            log_pdfs = jnp.array(
                [d.log_pdf(p, a) for d, p, a in zip(args, params, arys)]
            )
            return jnp.where(
                (len(x) == self.dim),
                log_pdfs.sum(),
                -jnp.inf,
            )

    return ConcatenatedDistribution(dim)


def Repeat(f: Distribution, out_dim: int):
    """Given a d-dimension distribution, returns a distribution of dimension d * out_dim which repeats the distribution
    out_dim times.
    """

    class RepeatedDistribution(Distribution):
        @property
        def params(self):
            return f.params

        def sample(self, rng: jax.random.PRNGKey, params: T, n: int = 1) -> jnp.ndarray:
            s = f.sample(rng, params, n)  # [n, f.dim]
            return jnp.repeat(s, self.dim, axis=1)

        def log_pdf(self, params, x):
            return jnp.where(
                (len(x) == self.dim) & jnp.all(x[1:] == x[:-1]),
                f.log_pdf(params, x[0]),
                -jnp.inf,
            )

    return RepeatedDistribution(out_dim)


def _Compose2(
    T1: Type[Transformation], T2: Type[Transformation]
) -> Type[Transformation]:
    """Return a new transformation representing t2(t1(x)).

    Args:
        T1, T2: The transformations to compose.

    Returns:
        The transformed Flow.
    """

    @dataclass
    class Composition(Transformation):
        @property
        def _ts(self):
            return T1(self.dim), T2(self.dim)

        @property
        def params(self):
            t1, t2 = self._ts
            return {"t1": t1.params, "t2": t2.params}

        def direct(self, params, x):
            t1, t2 = self._ts
            y = t1.direct(params["t1"], x)
            z = t2.direct(params["t2"], y)
            return z

        def log_det_jac(self, params, x):
            # D(f2 o f1)(x) = f2(y)
            # J(f1 o f2)(z) = J(f2)_y * J(f1)_x
            # => log |J(f1 o f2)| = log|Jf2(y)| + log|Jf1(x)|
            t1, t2 = self._ts
            y = t1.direct(params["t1"], x)
            return t1.log_det_jac(params["t1"], x) + t2.log_det_jac(params["t2"], y)

        def inverse(self, params, z):
            t1, t2 = self._ts
            y = t2.inverse(params["t2"], z)
            x = t1.inverse(params["t1"], y)
            return x

    return Composition


@dataclass
class Identity(Transformation):
    @property
    def params(self):
        return {}

    def direct(self, params, x):
        return x

    def inverse(self, params, y):
        return y

    def log_det_jac(self, params, x):
        return 0.0


def Compose(*args: Type[Transformation]) -> Type[Transformation]:
    """Compose(T1, T2, ..., Tn) returns a transformation that maps x to Tn(Tn-1(...(T1(x))...)."""
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return _Compose2(*args)
    return _Compose2(Compose(*args[:-1]), args[-1])


def Householder(rank: int = 1) -> Type[Transformation]:
    # from jax.experimental import stax
    # from jax.experimental.stax import Dense, BatchNorm, Relu

    class HH(Transformation):
        r: int = rank

        # def __post_init(self):
        #     init_fun, self._nn = stax.serial(
        #         Dense(4 * self.dim), BatchNorm(), Relu, Dense(2 * self.dim)
        #     )
        #     rng = jax.random.PRNGKey(0)
        #     in_shape = (-1, self.dim)
        #     out_shape, self._nn_params = init_fun(rng, in_shape)

        @property
        def params(self):
            return {"V": jnp.ones((self.r, self.dim))}

        def direct(self, params, x):
            V = params["V"]
            return jax.lax.scan(
                lambda y, v: (y - 2 * v * jnp.dot(y, v) / jnp.dot(v, v), None), x, V
            )[0]

        def inverse(self, params, y):
            return self.direct({"V": params["V"][::-1]}, y)

        def log_det_jac(self, params, x):
            return 0.0

    return HH


def Bounded(low, high) -> Transform:
    return Compose(
        ZeroOne,
        lambda dim: Scale(dim=dim, c=jnp.log(high - low)),
        lambda dim: Shift(dim=dim, c=low),
    )
