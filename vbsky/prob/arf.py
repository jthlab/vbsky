import itertools
from dataclasses import dataclass

import jax
import numpy as np
from jax import numpy as jnp
from jax.example_libraries import stax
from jax.example_libraries.stax import Relu

from vbsky.prob.transform import Transformation


def get_masks(input_dim, hidden_dim, hidden_layers=1):
    masks = []
    input_degrees = np.arange(input_dim)
    degrees = [input_degrees]

    for n_h in range(hidden_layers + 1):
        degrees.append(np.arange(hidden_dim) % (input_dim - 1))
    degrees.append(input_degrees % input_dim - 1)

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks.append(
            np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(float)
        )

    return masks


def get_conditional_masks(input_dim, conditional_dim, hidden_dim, hidden_layers=1):
    masks = []
    input_degrees = np.concatenate(
        [np.zeros(conditional_dim), np.arange(1, input_dim + 1)]
    )
    degrees = [input_degrees]

    for n_h in range(hidden_layers + 1):
        degrees.append(np.arange(hidden_dim) % input_dim)
    degrees.append(np.arange(input_dim))

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks.append(
            np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(float)
        )

    return masks


def MaskedDense(mask):
    def init_fun(rng, input_shape):
        out_dim = mask.shape[-1]
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = jax.random.split(rng)
        bound = 1.0 / (input_shape[-1] ** 0.5)
        W = jax.random.uniform(
            k1, (input_shape[-1], out_dim), minval=-bound, maxval=bound
        )
        b = jax.random.uniform(k2, (out_dim,), minval=-bound, maxval=bound)
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return jnp.dot(inputs, W * mask) + b

    return init_fun, apply_fun


def nn_from_masks(masks, rng, input_dim):
    layers = itertools.chain.from_iterable(
        (MaskedDense(masks[i]), Relu) for i in range(len(masks) - 1)
    )
    layers = list(layers)
    layers.append(MaskedDense(np.tile(masks[-1], 2)))

    init_fun, apply_fun = stax.serial(*layers)
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun


def create_transform(hidden_layers=1, hidden_dim=64, conditional_dim=None):

    if conditional_dim is None:

        def masked_transform(rng, input_dim):
            masks = get_masks(
                input_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers
            )
            return nn_from_masks(masks, rng, input_dim)

    else:

        def masked_transform(rng, input_dim):
            hidden_dim = conditional_dim + input_dim // 2
            masks = get_conditional_masks(
                input_dim,
                conditional_dim=conditional_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
            )
            return nn_from_masks(masks, rng, input_dim + conditional_dim)

    return masked_transform


def MAF(transform, rng):
    @dataclass
    class MADE(Transformation):
        @property
        def params(self) -> dict:
            params, _ = transform(rng, self.dim)
            return params

        def forward(self, params, u):
            _, apply_fun = transform(rng, self.dim)

            init = jnp.zeros_like(u)

            def f(carry, i):
                log_weight, bias = apply_fun(params, carry).split(2, axis=-1)
                return (
                    jax.ops.index_update(
                        carry, i, u[i] * jnp.exp(log_weight[i]) + bias[i]
                    ),
                    log_weight,
                )

            x, log_weight = jax.lax.scan(f, init, jnp.arange(len(u)))
            log_det_jacobian = -log_weight.sum(-1)
            return x, log_det_jacobian

        def direct(self, params, u):
            x, _ = self.forward(params, u)
            return x

        def log_det_jac(self, params, u):
            _, log_det_jacobian = self.forward(params, u)
            return log_det_jacobian

        def inverse(self, params, x):
            _, apply_fun = transform(rng, self.dim)
            log_weight, bias = apply_fun(params, x).split(2)
            u = (x - bias) * jnp.exp(-log_weight)
            return u

    return MADE


def IAF(transform, rng):
    @dataclass
    class MADE(Transformation):
        @property
        def params(self) -> dict:
            params, _ = transform(rng, self.dim)
            return params

        def direct(self, params, u):
            _, apply_fun = transform(rng, self.dim)
            log_weight, bias = apply_fun(params, u).split(2)
            x = u * jnp.exp(log_weight) + bias
            return x

        def log_det_jac(self, params, u):
            _, apply_fun = transform(rng, self.dim)
            log_weight, bias = apply_fun(params, u).split(2)
            log_det_jacobian = log_weight.sum(-1)
            return log_det_jacobian

        def inverse(self, params, x):
            _, apply_fun = transform(rng, self.dim)

            init = jnp.zeros_like(x)

            def f(carry, i):
                log_weight, bias = apply_fun(params, carry).split(2)
                return (
                    jax.ops.index_update(
                        carry, i, (x[i] - bias[i]) * jnp.exp(-log_weight[i])
                    ),
                    None,
                )

            u, _ = jax.lax.scan(f, init, jnp.arange(len(x)))
            return u

    return MADE


def Conditional_IAF(transform, rng):
    @dataclass
    class MADE(Transformation):
        @property
        def params(self) -> dict:
            params, _ = transform(rng, self.dim)
            return params

        def direct(self, params, c, u):
            _, apply_fun = transform(rng, self.dim)
            v = jnp.concatenate([c, u], axis=-1)
            log_weight, bias = apply_fun(params, v).split(2, axis=1)
            x = u * jnp.exp(log_weight) + bias
            return x

        def log_det_jac(self, params, c, u):
            _, apply_fun = transform(rng, self.dim)
            v = jnp.concatenate([c, u], axis=-1)
            log_weight, bias = apply_fun(params, v).split(2)
            log_det_jacobian = log_weight.sum(-1)
            return log_det_jacobian

        def inverse(self, params, c, x):
            _, apply_fun = transform(rng, self.dim)
            v = jnp.concatenate([c, x], axis=-1)

            init = jnp.zeros_like(v)

            def f(carry, i):
                log_weight, bias = apply_fun(params, carry).split(2)
                return (
                    jax.ops.index_update(
                        carry, i, (x[i] - bias[i]) * jnp.exp(-log_weight[i])
                    ),
                    None,
                )

            u, _ = jax.lax.scan(f, init, jnp.arange(len(x)))
            return u[-self.dim :]

    return MADE


def get_arys(inds, u):
    arys = u.split(inds[:-1], axis=-1)
    return arys


def Conditional(block_sizes, transformations):
    assert len(block_sizes) == len(transformations)
    inds = tuple(jnp.cumsum(jnp.array(block_sizes)).astype(int))

    @dataclass
    class Conditional_Block(Transformation):
        @property
        def params(self):
            return [T(b).params for b, T in zip(block_sizes, transformations)]

        def direct(self, params, u):
            arys = get_arys(inds, u)
            xs = [transformations[0](block_sizes[0]).direct(params[0], arys[0])]
            for T, b, p, a in zip(
                transformations[1:], block_sizes[1:], params[1:], arys[1:]
            ):
                xs.append(T(b).direct(p, arys[0], a))
            return jnp.concatenate(xs, axis=-1)

        def log_det_jac(self, params, u):
            arys = get_arys(inds, u)
            log_det_jacs = [
                transformations[0](block_sizes[0]).log_det_jac(params[0], arys[0])
            ]
            for T, b, p, a in zip(
                transformations[1:], block_sizes[1:], params[1:], arys[1:]
            ):
                log_det_jacs.append(T(b).log_det_jac(p, arys[0], a))
            return jnp.sum(jnp.array(log_det_jacs))

        def inverse(self, params, x):
            arys = get_arys(inds, x)
            us = [transformations[0](block_sizes[0]).inverse(params[0], arys[0])]
            for T, b, p, a in zip(
                transformations[1:], block_sizes[1:], params[1:], arys[1:]
            ):
                us.append(T(b).inverse(p, arys[0], a))
            return jnp.concatenate(us, axis=-1)

    return Conditional_Block
